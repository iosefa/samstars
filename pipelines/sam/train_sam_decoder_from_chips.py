#!/usr/bin/env python3
"""
Train SAM ViT-L mask decoder per size using chips built from manifests.

Upgrade: align training prompts with inference:
- Use box prompts (json["box"] if present, else derived from mask bbox)
- Use many fg/bg points derived from CHM, like inference
- Pad variable-length points to a fixed size with (-10,-10) and label -1

We still:
- freeze image_encoder + prompt_encoder
- train mask_decoder only
- use global RGB scaling to uint8 (via --scale-stats)
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

import geopandas as gpd  # kept because your script uses it in auto-epochs counting

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pipelines.shared.scale_stats import load_scale_stats, scale_to_u8_per_channel


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
        self.chip_dir = chip_dir
        self.ids = sorted((self.chip_dir / "img").glob("*.npy"))
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
        return len(self.ids)

    def __getitem__(self, i):
        name = self.ids[i].stem

        # --- load RGB chip (C,H,W) ---
        img = np.load(self.chip_dir / "img" / f"{name}.npy")
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
        msk = np.load(self.chip_dir / "msk" / f"{name}.npy")
        msk = (msk > 0).astype(np.float32)

        # --- load CHM (H,W) for prompt sampling ---
        chm_path = self.chip_dir / "chm" / f"{name}.npy"
        if not chm_path.exists():
            raise FileNotFoundError(f"{name}: missing CHM chip {chm_path}")
        chm = np.load(chm_path).astype(np.float32)

        # --- load json (point + optional box) ---
        jpath = self.chip_dir / "json" / f"{name}.json"
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

    print(f"{chip_dir.parent.name}: epochs={epochs} (iters/epoch={iters_per_epoch})")

    sam = sam_model_registry["vit_l"](checkpoint=str(ckpt)).to(device)
    sam.image_encoder.requires_grad_(False)
    sam.prompt_encoder.requires_grad_(False)

    opt = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=lr)
    base_pe = sam.prompt_encoder.get_dense_pe().to(device)

    def loss_fn(pred, truth):
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

        print(f"{chip_dir.parent.name} epoch {ep}/{epochs} loss={tot/len(loader):.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sam.state_dict(), out_path)
    print(f"✓ weights → {out_path}")


def main():
    pa = argparse.ArgumentParser(description="Train SAM decoder per size from chips_train.")
    pa.add_argument("--chips-root", type=Path, required=True, help="Root containing per-size/chips_train dirs")
    pa.add_argument("--ckpt", type=Path, required=True, help="SAM ViT-L checkpoint")
    pa.add_argument("--epochs", type=int, default=15)
    pa.add_argument("--min-iters", type=int, default=0, help="Minimum iterations (converted to epochs)")
    pa.add_argument("--auto-epochs-ref-size", type=str, default=None, help="Reference size (e.g., 0100) for auto-epochs")
    pa.add_argument("--auto-epochs-ref-epochs", type=float, default=None, help="Reference epochs for auto-epochs")
    pa.add_argument("--device", default="cpu")
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes to process")
    pa.add_argument("--scale-stats", type=Path, default=None, help="Global scaling stats JSON")

    # New knobs (optional)
    pa.add_argument("--fg-extra", type=int, default=FG_EXTRA_DEFAULT)
    pa.add_argument("--neg-max-gnd", type=int, default=NEG_MAX_GND_DEFAULT)
    pa.add_argument("--neg-max-can", type=int, default=NEG_MAX_CAN_DEFAULT)
    pa.add_argument("--chm-ground", type=float, default=CHM_GROUND_DEFAULT)
    pa.add_argument("--box-expand", type=float, default=1.0, help="Expand bbox about its center (1.0 = no expand)")

    args = pa.parse_args()

    rgb_lo_hi = None
    if args.scale_stats:
        stats = load_scale_stats(args.scale_stats)
        per_channel = stats.get("sam_rgb_u8", {}).get("per_channel", [])
        rgb_lo_hi = [(float(c["lo"]), float(c["hi"])) for c in per_channel]

    if args.sizes:
        sizes = {s.strip() for s in args.sizes.split(",") if s.strip()}
        size_dirs = [args.chips_root / s for s in sorted(sizes)]
    else:
        size_dirs = sorted(p for p in args.chips_root.iterdir() if p.is_dir())

    for size_dir in size_dirs:
        chip_dir = size_dir / "chips_train"
        if not chip_dir.exists():
            continue

        out_path = size_dir / "sam" / "sam_crowns_decoder.pth"

        # Keep your auto-epochs behavior exactly as before
        epochs = args.epochs
        if args.auto_epochs_ref_size and args.auto_epochs_ref_epochs:
            ref_manifest = args.chips_root / args.auto_epochs_ref_size / "chips_train" / "detectron_manifest.json"
            cur_manifest = size_dir / "chips_train" / "detectron_manifest.json"
            if not ref_manifest.exists() or not cur_manifest.exists():
                raise FileNotFoundError("Auto-epochs requires detectron_manifest.json for ref and current size.")

            def _count(mpath):
                data = json.load(open(mpath))
                n_chips = len(data)
                n_inst = 0
                for rec in data:
                    gpkg = Path(rec["gpkg_path"])
                    if gpkg.exists():
                        gdf = gpd.read_file(gpkg)
                        n_inst += len(gdf)
                return n_chips, n_inst

            ref_chips, ref_inst = _count(ref_manifest)
            cur_chips, cur_inst = _count(cur_manifest)
            if ref_inst <= 0 or cur_inst <= 0:
                raise RuntimeError("Auto-epochs requires non-zero instance counts.")
            updates_per_inst = (ref_chips * args.auto_epochs_ref_epochs) / ref_inst
            epochs = int(np.ceil(updates_per_inst * cur_inst / max(cur_chips, 1)))
            print(f"{size_dir.name}: auto-epochs -> {epochs}")

        train_decoder(
            chip_dir,
            args.ckpt,
            out_path,
            epochs=epochs,
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

# #!/usr/bin/env python3
# """
# Train SAM ViT-L mask decoder per size using chips built from manifests.
# For each size dir under --chips-root, trains decoder on chips_train and saves weights under sam/.
# """
# from __future__ import annotations
# import argparse, json
# from pathlib import Path
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from segment_anything import sam_model_registry
# from segment_anything.utils.transforms import ResizeLongestSide
# import geopandas as gpd
# from scale_stats import load_scale_stats, scale_to_u8_per_channel
#
#
# def stretch_u8(arr):
#     arr = np.asarray(arr, np.float32)
#     finite = np.isfinite(arr)
#     if not finite.any():
#         return np.zeros_like(arr, dtype=np.uint8)
#     lo = float(arr[finite].min())
#     hi = float(arr[finite].max())
#     if hi <= lo:
#         return np.zeros_like(arr, dtype=np.uint8)
#     out = (arr - lo) * 255.0 / (hi - lo)
#     return np.clip(out, 0, 255).astype(np.uint8)
#
#
# class Chips(Dataset):
#     def __init__(self, chip_dir: Path, img_size=512, enc_size=1024, rgb_lo_hi=None):
#         self.chip_dir = chip_dir
#         self.ids = sorted((self.chip_dir / "img").glob("*.npy"))
#         if not self.ids:
#             raise FileNotFoundError(f"No chips in {self.chip_dir}")
#         self.img_size = img_size
#         self.resize = ResizeLongestSide(enc_size)
#         self.rgb_lo_hi = rgb_lo_hi
#
#     def __len__(self):
#         return len(self.ids)
#
#     def __getitem__(self, i):
#         name = self.ids[i].stem
#         img = np.load(self.chip_dir / "img" / f"{name}.npy")
#         img_hwc = img.transpose(1, 2, 0)
#         if self.rgb_lo_hi is not None:
#             if len(self.rgb_lo_hi) >= img_hwc.shape[2]:
#                 img_u8 = scale_to_u8_per_channel(img_hwc, self.rgb_lo_hi[: img_hwc.shape[2]])
#             else:
#                 img_u8 = np.stack([stretch_u8(img_hwc[..., c]) for c in range(img_hwc.shape[2])], axis=-1)
#         else:
#             img_u8 = np.stack([stretch_u8(img_hwc[..., c]) for c in range(img_hwc.shape[2])], axis=-1)
#         img_u8 = self.resize.apply_image(img_u8)
#         img = img_u8.transpose(2, 0, 1).astype(np.float32)
#         msk = np.load(self.chip_dir / "msk" / f"{name}.npy")
#         pt = np.array(json.load(open(self.chip_dir / "json" / f"{name}.json"))["point"])
#         pt = self.resize.apply_coords(pt[None], (self.img_size, self.img_size))[0]
#         return torch.tensor(img).float(), torch.tensor(msk[None]).float(), torch.tensor(pt).float()
#
#
# def train_decoder(
#     chip_dir: Path,
#     ckpt: Path,
#     out_path: Path,
#     img_size=512,
#     enc_size=1024,
#     epochs=15,
#     min_iters=0,
#     lr=5e-5,
#     device="cpu",
#     rgb_lo_hi=None,
# ):
#     loader = DataLoader(Chips(chip_dir, img_size, enc_size, rgb_lo_hi=rgb_lo_hi), batch_size=1, shuffle=True)
#     iters_per_epoch = max(1, len(loader))
#     if min_iters > 0:
#         min_epochs = int((min_iters + iters_per_epoch - 1) // iters_per_epoch)
#         epochs = max(epochs, min_epochs)
#     print(f"{chip_dir.parent.name}: epochs={epochs} (iters/epoch={iters_per_epoch})")
#     sam = sam_model_registry["vit_l"](checkpoint=str(ckpt)).to(device)
#     sam.image_encoder.requires_grad_(False)
#     sam.prompt_encoder.requires_grad_(False)
#     opt = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=lr)
#     base_pe = sam.prompt_encoder.get_dense_pe().to(device)
#
#     def loss_fn(pred, truth):
#         score = torch.sigmoid(pred)
#         dice = 1 - (2 * (score * truth).sum((2, 3)) + 1) / (score.sum((2, 3)) + truth.sum((2, 3)) + 1)
#         return (dice + F.binary_cross_entropy_with_logits(pred, truth)).mean()
#
#     for ep in range(1, epochs + 1):
#         tot = 0
#         for img, msk, pt in loader:
#             img, msk, pt = img.to(device), msk.to(device), pt.to(device)
#             with torch.no_grad():
#                 emb = sam.image_encoder(sam.preprocess(img))
#             sparse, dense = sam.prompt_encoder(points=(pt[:, None, :], torch.ones((1, 1), device=device)), boxes=None, masks=None)
#             logit_low, _ = sam.mask_decoder(
#                 image_embeddings=emb,
#                 image_pe=base_pe,
#                 sparse_prompt_embeddings=sparse,
#                 dense_prompt_embeddings=dense,
#                 multimask_output=False,
#             )
#             logit = torch.nn.functional.interpolate(logit_low, size=(img_size, img_size), mode="bilinear", align_corners=False)
#             loss = loss_fn(logit, msk)
#             opt.zero_grad(); loss.backward(); opt.step()
#             tot += loss.item()
#         print(f"{chip_dir.parent.name} epoch {ep}/{epochs} loss={tot/len(loader):.4f}")
#
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     torch.save(sam.state_dict(), out_path)
#     print(f"✓ weights → {out_path}")
#
#
# def main():
#     pa = argparse.ArgumentParser(description="Train SAM decoder per size from chips_train.")
#     pa.add_argument("--chips-root", type=Path, required=True, help="Root containing per-size/chips_train dirs")
#     pa.add_argument("--ckpt", type=Path, required=True, help="SAM ViT-L checkpoint")
#     pa.add_argument("--epochs", type=int, default=15)
#     pa.add_argument("--min-iters", type=int, default=0, help="Minimum iterations (converted to epochs)")
#     pa.add_argument("--auto-epochs-ref-size", type=str, default=None, help="Reference size (e.g., 0100) for auto-epochs")
#     pa.add_argument("--auto-epochs-ref-epochs", type=float, default=None, help="Reference epochs for auto-epochs")
#     pa.add_argument("--device", default="cpu")
#     pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes to process")
#     pa.add_argument("--scale-stats", type=Path, default=None, help="Global scaling stats JSON")
#     args = pa.parse_args()
#
#     rgb_lo_hi = None
#     if args.scale_stats:
#         stats = load_scale_stats(args.scale_stats)
#         per_channel = stats.get("sam_rgb_u8", {}).get("per_channel", [])
#         rgb_lo_hi = [(float(c["lo"]), float(c["hi"])) for c in per_channel]
#
#     if args.sizes:
#         sizes = {s.strip() for s in args.sizes.split(",") if s.strip()}
#         size_dirs = [args.chips_root / s for s in sorted(sizes)]
#     else:
#         size_dirs = sorted(p for p in args.chips_root.iterdir() if p.is_dir())
#
#     for size_dir in size_dirs:
#         chip_dir = size_dir / "chips_train"
#         if not chip_dir.exists():
#             continue
#         out_path = size_dir / "sam" / "sam_crowns_decoder.pth"
#         epochs = args.epochs
#         if args.auto_epochs_ref_size and args.auto_epochs_ref_epochs:
#             ref_manifest = args.chips_root / args.auto_epochs_ref_size / "chips_train" / "detectron_manifest.json"
#             cur_manifest = size_dir / "chips_train" / "detectron_manifest.json"
#             if not ref_manifest.exists() or not cur_manifest.exists():
#                 raise FileNotFoundError("Auto-epochs requires detectron_manifest.json for ref and current size.")
#             def _count(mpath):
#                 data = json.load(open(mpath))
#                 n_chips = len(data)
#                 n_inst = 0
#                 for rec in data:
#                     gpkg = Path(rec["gpkg_path"])
#                     if gpkg.exists():
#                         gdf = gpd.read_file(gpkg)
#                         n_inst += len(gdf)
#                 return n_chips, n_inst
#             ref_chips, ref_inst = _count(ref_manifest)
#             cur_chips, cur_inst = _count(cur_manifest)
#             if ref_inst <= 0 or cur_inst <= 0:
#                 raise RuntimeError("Auto-epochs requires non-zero instance counts.")
#             updates_per_inst = (ref_chips * args.auto_epochs_ref_epochs) / ref_inst
#             epochs = int(np.ceil(updates_per_inst * cur_inst / max(cur_chips, 1)))
#             print(f"{size_dir.name}: auto-epochs -> {epochs}")
#
#         train_decoder(
#             chip_dir,
#             args.ckpt,
#             out_path,
#             epochs=epochs,
#             min_iters=args.min_iters,
#             device=args.device,
#             rgb_lo_hi=rgb_lo_hi,
#         )
#
#
# if __name__ == "__main__":
#     main()
