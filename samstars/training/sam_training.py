#!/usr/bin/env python3
"""Train the proposal decoder from image chips and individual crown masks.

Training uses crown boxes and foreground/background points derived from canopy
height. The image and prompt encoders remain frozen while the mask decoder is
optimized. Inputs are scaled to uint8 using supplied limits or per-chip scaling.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from samstars.data.scaling import load_scale_stats, scale_to_u8_per_channel


# ───────────── prompt settings (match inference style) ─────────────────────
FG_EXTRA_DEFAULT = 80
NEG_MAX_GND_DEFAULT = 150
NEG_MAX_CAN_DEFAULT = 300
CHM_GROUND_DEFAULT = 3.0

PAD_COORD = (-10.0, -10.0)  # same padding convention as inference
# ──────────────────────────────────────────────────────────────────────────


def stretch_u8(arr: np.ndarray) -> np.ndarray:
    """Fallback per-array min/max stretch to uint8."""
    arr = np.asarray(arr, np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.uint8)
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (arr - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def _bbox_from_mask(msk: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return XYXY bbox from a binary mask in pixel coordinates."""
    yy, xx = np.where(msk > 0)
    if yy.size == 0:
        return None
    x0 = float(xx.min())
    y0 = float(yy.min())
    x1 = float(xx.max())
    y1 = float(yy.max())
    return x0, y0, x1, y1


def _clamp_box_xyxy(box: tuple[float, float, float, float], W: int, H: int) -> tuple[float, float, float, float]:
    """Internal helper function."""
    x0, y0, x1, y1 = box
    x0 = float(np.clip(x0, 0, W - 1))
    x1 = float(np.clip(x1, 0, W - 1))
    y0 = float(np.clip(y0, 0, H - 1))
    y1 = float(np.clip(y1, 0, H - 1))
    # enforce non-degenerate
    if x1 <= x0:
        x1 = min(float(W - 1), x0 + 1.0)
    if y1 <= y0:
        y1 = min(float(H - 1), y0 + 1.0)
    return x0, y0, x1, y1


def _expand_box_xyxy(box: tuple[float, float, float, float], factor: float, W: int, H: int) -> tuple[float, float, float, float]:
    """Expand box about its center by a scale factor."""
    x0, y0, x1, y1 = box
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    hx = 0.5 * (x1 - x0) * factor
    hy = 0.5 * (y1 - y0) * factor
    out = (cx - hx, cy - hy, cx + hx, cy + hy)
    return _clamp_box_xyxy(out, W, H)


def _sample_points_from_chm(
    *,
    seed_xy: tuple[int, int],
    box_xyxy: tuple[float, float, float, float],
    chm: np.ndarray,
    fg_extra: int,
    neg_max_gnd: int,
    neg_max_can: int,
    chm_ground: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build points + labels like inference:
      - seed point (positive)
      - FG_EXTRA: top CHM pixels inside box with CHM > CHM_GROUND (positive)
      - NEG ground: random points where CHM < CHM_GROUND (negative)
      - NEG canopy outside box: random points where CHM > CHM_GROUND and outside box (negative)
    Returns:
      pts: (N,2) float32 in XY
      lbl: (N,) int64 in {1,0}
    """
    H, W = chm.shape
    sx, sy = seed_xy

    x0, y0, x1, y1 = box_xyxy
    # int region bounds (inclusive-ish)
    x0i = int(np.clip(np.floor(x0), 0, W - 1))
    y0i = int(np.clip(np.floor(y0), 0, H - 1))
    x1i = int(np.clip(np.ceil(x1), 0, W - 1))
    y1i = int(np.clip(np.ceil(y1), 0, H - 1))

    # --- foreground extra points (inside box) ---
    fg_pts = np.empty((0, 2), np.float32)
    if fg_extra > 0 and x1i >= x0i and y1i >= y0i:
        region = chm[y0i : y1i + 1, x0i : x1i + 1]
        m = np.isfinite(region) & (region > chm_ground)
        yy, xx = np.where(m)
        if yy.size:
            keep = min(fg_extra, yy.size)
            vals = region[yy, xx]
            # top-k CHM
            top_idx = np.argpartition(vals, -keep)[-keep:]
            fg_pts = np.column_stack([xx[top_idx] + x0i, yy[top_idx] + y0i]).astype(np.float32)

    # --- negative ground points (anywhere) ---
    neg_g = np.empty((0, 2), np.float32)
    if neg_max_gnd > 0:
        yg, xg = np.where(np.isfinite(chm) & (chm < chm_ground))
        if yg.size:
            if yg.size > neg_max_gnd:
                sel = np.random.choice(yg.size, neg_max_gnd, replace=False)
                yg, xg = yg[sel], xg[sel]
            neg_g = np.column_stack([xg, yg]).astype(np.float32)

    # --- negative canopy points OUTSIDE the box ---
    neg_c = np.empty((0, 2), np.float32)
    if neg_max_can > 0:
        yc, xc = np.where(np.isfinite(chm) & (chm > chm_ground))
        if yc.size:
            out_mask = (xc < x0i) | (xc > x1i) | (yc < y0i) | (yc > y1i)
            yc, xc = yc[out_mask], xc[out_mask]
            if yc.size:
                if yc.size > neg_max_can:
                    sel = np.random.choice(yc.size, neg_max_can, replace=False)
                    yc, xc = yc[sel], xc[sel]
                neg_c = np.column_stack([xc, yc]).astype(np.float32)

    # seed point first
    pts = [np.array([[float(sx), float(sy)]], np.float32)]
    lbl = [np.array([1], np.int64)]

    if fg_pts.size:
        pts.append(fg_pts)
        lbl.append(np.ones((fg_pts.shape[0],), np.int64))

    if neg_g.size:
        pts.append(neg_g)
        lbl.append(np.zeros((neg_g.shape[0],), np.int64))

    if neg_c.size:
        pts.append(neg_c)
        lbl.append(np.zeros((neg_c.shape[0],), np.int64))

    pts = np.vstack(pts).astype(np.float32)
    lbl = np.concatenate(lbl).astype(np.int64)
    return pts, lbl


class Chips(Dataset):
    def __init__(
        self,
        chip_dir: Path,
        img_size: int = 512,
        enc_size: int = 1024,
        rgb_lo_hi=None,
        *,
        fg_extra: int = FG_EXTRA_DEFAULT,
        neg_max_gnd: int = NEG_MAX_GND_DEFAULT,
        neg_max_can: int = NEG_MAX_CAN_DEFAULT,
        chm_ground: float = CHM_GROUND_DEFAULT,
        box_expand: float = 1.0,
    ):
        """Initialize the instance."""
        self.chip_dir = chip_dir
        self.ids = sorted((self.chip_dir / "images").glob("*.npy"))
        if not self.ids:
            raise FileNotFoundError(f"No chips in {self.chip_dir}")

        self.img_size = int(img_size)
        self.resize = ResizeLongestSide(enc_size)
        self.rgb_lo_hi = rgb_lo_hi

        self.fg_extra = int(fg_extra)
        self.neg_max_gnd = int(neg_max_gnd)
        self.neg_max_can = int(neg_max_can)
        self.chm_ground = float(chm_ground)
        self.box_expand = float(box_expand)

        # Fixed maximum number of points so DataLoader can stack tensors.
        # seed + fg + neg_gnd + neg_can
        self.max_points = 1 + self.fg_extra + self.neg_max_gnd + self.neg_max_can

    def __len__(self):
        """Return the number of items."""
        return len(self.ids)

    def __getitem__(self, i):
        """Return one item by index."""
        name = self.ids[i].stem

        # --- load RGB chip (C,H,W) ---
        img = np.load(self.chip_dir / "images" / f"{name}.npy")
        if img.ndim != 3:
            raise ValueError(f"{name}: expected img CHW, got {img.shape}")
        C, H, W = img.shape
        if H != self.img_size or W != self.img_size:
            # if your chips are padded/cropped, this keeps things safe
            H = min(H, self.img_size)
            W = min(W, self.img_size)

        # Convert RGB -> uint8 (global scaling if available)
        img_hwc = img.transpose(1, 2, 0).astype(np.float32)
        if self.rgb_lo_hi is not None and len(self.rgb_lo_hi) >= img_hwc.shape[2]:
            img_u8 = scale_to_u8_per_channel(img_hwc, self.rgb_lo_hi[: img_hwc.shape[2]])
        else:
            img_u8 = np.stack([stretch_u8(img_hwc[..., c]) for c in range(img_hwc.shape[2])], axis=-1)

        # Resize to encoder scale (1024 long side)
        img_u8 = self.resize.apply_image(img_u8)
        img_t = img_u8.transpose(2, 0, 1).astype(np.float32)  # CHW float, 0..255

        # --- load mask (H,W) ---
        msk = np.load(self.chip_dir / "masks" / f"{name}.npy")
        msk = (msk > 0).astype(np.float32)

        # --- load CHM (H,W) for prompt sampling ---
        chm_path = self.chip_dir / "chm" / f"{name}.npy"
        if not chm_path.exists():
            raise FileNotFoundError(f"{name}: missing CHM chip {chm_path}")
        chm = np.load(chm_path).astype(np.float32)

        # --- load json (point + optional box) ---
        jpath = self.chip_dir / "prompts" / f"{name}.json"
        j = json.load(open(jpath))
        pt = j.get("point", None)

        if pt is None:
            # fallback: centroid of mask
            yy, xx = np.where(msk > 0)
            if yy.size == 0:
                # should not happen, but keep safe
                sx, sy = self.img_size // 2, self.img_size // 2
            else:
                sx = int(np.round(xx.mean()))
                sy = int(np.round(yy.mean()))
        else:
            sx, sy = int(pt[0]), int(pt[1])

        # Box: prefer json["box"], else bbox from mask
        box = j.get("box", None)
        if box is not None and len(box) == 4:
            x0, y0, x1, y1 = map(float, box)
            box_xyxy = _clamp_box_xyxy((x0, y0, x1, y1), self.img_size, self.img_size)
        else:
            bb = _bbox_from_mask(msk)
            if bb is None:
                # fallback: small box around seed
                box_xyxy = _clamp_box_xyxy((sx - 8, sy - 8, sx + 8, sy + 8), self.img_size, self.img_size)
            else:
                box_xyxy = _clamp_box_xyxy(bb, self.img_size, self.img_size)

        if self.box_expand and self.box_expand != 1.0:
            box_xyxy = _expand_box_xyxy(box_xyxy, self.box_expand, self.img_size, self.img_size)

        # Build inference-style points from CHM
        pts, lbl = _sample_points_from_chm(
            seed_xy=(sx, sy),
            box_xyxy=box_xyxy,
            chm=chm,
            fg_extra=self.fg_extra,
            neg_max_gnd=self.neg_max_gnd,
            neg_max_can=self.neg_max_can,
            chm_ground=self.chm_ground,
        )

        # Pad to fixed length (max_points)
        pts_pad = np.full((self.max_points, 2), PAD_COORD, dtype=np.float32)
        lbl_pad = np.full((self.max_points,), -1, dtype=np.int64)
        n = min(self.max_points, pts.shape[0])
        pts_pad[:n] = pts[:n]
        lbl_pad[:n] = lbl[:n]

        # Transform coords/boxes to resized image coords (1024-long-side space)
        pts_pad = self.resize.apply_coords(pts_pad, (self.img_size, self.img_size)).astype(np.float32)
        box_np = np.array([box_xyxy], dtype=np.float32)
        box_rs = self.resize.apply_boxes(box_np, (self.img_size, self.img_size)).astype(np.float32)[0]

        return (
            torch.tensor(img_t).float(),
            torch.tensor(msk[None]).float(),
            torch.tensor(pts_pad).float(),     # (P,2)
            torch.tensor(lbl_pad).long(),      # (P,)
            torch.tensor(box_rs).float(),      # (4,)
        )


def train_decoder(
    chip_dir: Path,
    ckpt: Path,
    out_path: Path,
    img_size: int = 512,
    enc_size: int = 1024,
    epochs: int = 15,
    min_iters: int = 0,
    lr: float = 5e-5,
    device: str = "cpu",
    rgb_lo_hi=None,
    *,
    fg_extra: int = FG_EXTRA_DEFAULT,
    neg_max_gnd: int = NEG_MAX_GND_DEFAULT,
    neg_max_can: int = NEG_MAX_CAN_DEFAULT,
    chm_ground: float = CHM_GROUND_DEFAULT,
    box_expand: float = 1.0,
):
    """Train decoder."""
    ds = Chips(
        chip_dir,
        img_size=img_size,
        enc_size=enc_size,
        rgb_lo_hi=rgb_lo_hi,
        fg_extra=fg_extra,
        neg_max_gnd=neg_max_gnd,
        neg_max_can=neg_max_can,
        chm_ground=chm_ground,
        box_expand=box_expand,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    iters_per_epoch = max(1, len(loader))
    if min_iters > 0:
        min_epochs = int((min_iters + iters_per_epoch - 1) // iters_per_epoch)
        epochs = max(epochs, min_epochs)

    print(f"{chip_dir.name}: epochs={epochs} (iters/epoch={iters_per_epoch})")

    sam = sam_model_registry["vit_l"](checkpoint=str(ckpt)).to(device)
    sam.image_encoder.requires_grad_(False)
    sam.prompt_encoder.requires_grad_(False)

    opt = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=lr)
    base_pe = sam.prompt_encoder.get_dense_pe().to(device)

    def loss_fn(pred, truth):
        """Loss fn."""
        score = torch.sigmoid(pred)
        dice = 1 - (2 * (score * truth).sum((2, 3)) + 1) / (score.sum((2, 3)) + truth.sum((2, 3)) + 1)
        return (dice + F.binary_cross_entropy_with_logits(pred, truth)).mean()

    for ep in range(1, epochs + 1):
        tot = 0.0
        for img, msk, pts, lbl, box in loader:
            img = img.to(device)
            msk = msk.to(device)
            pts = pts.to(device)          # (B,P,2) after batching
            lbl = lbl.to(device)          # (B,P)
            box = box.to(device)          # (B,4)

            with torch.no_grad():
                emb = sam.image_encoder(sam.preprocess(img))

            # IMPORTANT: now we train with the same prompt types as inference (points + boxes)
            sparse, dense = sam.prompt_encoder(
                points=(pts, lbl),
                boxes=box,
                masks=None,
            )

            logit_low, _ = sam.mask_decoder(
                image_embeddings=emb,
                image_pe=base_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )

            # supervise against the original 512x512 mask
            logit = F.interpolate(logit_low, size=(img_size, img_size), mode="bilinear", align_corners=False)
            loss = loss_fn(logit, msk)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot += float(loss.item())

        print(f"{chip_dir.name} epoch {ep}/{epochs} loss={tot/len(loader):.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sam.state_dict(), out_path)
    print(f"✓ weights → {out_path}")


def main():
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(description="Train the proposal decoder from a prepared training-chip directory.")
    pa.add_argument(
        "--training-chips-dir",
        type=Path,
        required=True,
        help="Training-chip directory containing images/, masks/, chm/, and prompts/ subdirectories",
    )
    pa.add_argument("--out-path", type=Path, required=True, help="Path to write the trained decoder checkpoint")
    pa.add_argument("--ckpt", type=Path, required=True, help="Base SAM ViT-L checkpoint")
    pa.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    pa.add_argument("--min-iters", type=int, default=0, help="Minimum training iterations (converted to epochs)")
    pa.add_argument("--device", default="cpu", help="Training device")
    pa.add_argument("--scale-stats", type=Path, default=None, help="Optional global imagery scaling stats JSON")

    # New knobs (optional)
    pa.add_argument("--fg-extra", type=int, default=FG_EXTRA_DEFAULT, help="Additional positive prompt points sampled from CHM")
    pa.add_argument("--neg-max-gnd", type=int, default=NEG_MAX_GND_DEFAULT, help="Maximum negative ground prompt points")
    pa.add_argument("--neg-max-can", type=int, default=NEG_MAX_CAN_DEFAULT, help="Maximum negative canopy prompt points")
    pa.add_argument("--chm-ground", type=float, default=CHM_GROUND_DEFAULT, help="CHM threshold used when sampling prompts")
    pa.add_argument("--box-expand", type=float, default=1.0, help="Expand bbox about its center (1.0 = no expand)")

    args = pa.parse_args()

    rgb_lo_hi = None
    if args.scale_stats:
        stats = load_scale_stats(args.scale_stats)
        per_channel = stats.get("sam_rgb_u8", {}).get("per_channel", [])
        rgb_lo_hi = [(float(c["lo"]), float(c["hi"])) for c in per_channel]

    chip_dir = args.training_chips_dir
    if not chip_dir.exists():
        raise FileNotFoundError(f"Missing chips dir: {chip_dir}")

    train_decoder(
        chip_dir,
        args.ckpt,
        args.out_path,
        epochs=args.epochs,
        min_iters=args.min_iters,
        device=args.device,
        rgb_lo_hi=rgb_lo_hi,
        fg_extra=args.fg_extra,
        neg_max_gnd=args.neg_max_gnd,
        neg_max_can=args.neg_max_can,
        chm_ground=args.chm_ground,
        box_expand=args.box_expand,
    )


if __name__ == "__main__":
    main()
