#!/usr/bin/env python3
"""
Train SAM ViT-L mask decoder per size using chips built from manifests.
For each size dir under --chips-root, trains decoder on chips_train and saves weights under sam/.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import geopandas as gpd


def stretch_u8(arr):
    arr = np.asarray(arr, np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.uint8)
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (arr - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def scale_u8(arr, lo, hi):
    arr = np.asarray(arr, np.float32)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return stretch_u8(arr)
    out = (arr - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def load_scale_stats(path: Path):
    data = json.load(open(path))
    low = data.get("low")
    high = data.get("high")
    if low is None or high is None:
        raise ValueError("scale stats must include 'low' and 'high'")
    return np.asarray(low, np.float32), np.asarray(high, np.float32)


class Chips(Dataset):
    def __init__(self, chip_dir: Path, img_size=512, enc_size=1024, scale_stats=None):
        self.chip_dir = chip_dir
        self.ids = sorted((self.chip_dir / "img").glob("*.npy"))
        if not self.ids:
            raise FileNotFoundError(f"No chips in {self.chip_dir}")
        self.img_size = img_size
        self.resize = ResizeLongestSide(enc_size)
        self.scale_stats = scale_stats

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        name = self.ids[i].stem
        img = np.load(self.chip_dir / "img" / f"{name}.npy")
        img_hwc = img.transpose(1, 2, 0)
        if self.scale_stats is not None:
            low, high = self.scale_stats
            if len(low) >= img_hwc.shape[2] and len(high) >= img_hwc.shape[2]:
                img_u8 = np.stack([scale_u8(img_hwc[..., c], low[c], high[c]) for c in range(img_hwc.shape[2])], axis=-1)
            else:
                img_u8 = np.stack([stretch_u8(img_hwc[..., c]) for c in range(img_hwc.shape[2])], axis=-1)
        else:
            img_u8 = np.stack([stretch_u8(img_hwc[..., c]) for c in range(img_hwc.shape[2])], axis=-1)
        img_u8 = self.resize.apply_image(img_u8)
        img = img_u8.transpose(2, 0, 1).astype(np.float32)
        msk = np.load(self.chip_dir / "msk" / f"{name}.npy")
        pt = np.array(json.load(open(self.chip_dir / "json" / f"{name}.json"))["point"])
        pt = self.resize.apply_coords(pt[None], (self.img_size, self.img_size))[0]
        return torch.tensor(img).float(), torch.tensor(msk[None]).float(), torch.tensor(pt).float()


def train_decoder(
    chip_dir: Path,
    ckpt: Path,
    out_path: Path,
    img_size=512,
    enc_size=1024,
    epochs=15,
    min_iters=0,
    lr=5e-5,
    device="cpu",
    scale_stats=None,
):
    loader = DataLoader(Chips(chip_dir, img_size, enc_size, scale_stats=scale_stats), batch_size=1, shuffle=True)
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
        tot = 0
        for img, msk, pt in loader:
            img, msk, pt = img.to(device), msk.to(device), pt.to(device)
            with torch.no_grad():
                emb = sam.image_encoder(sam.preprocess(img))
            sparse, dense = sam.prompt_encoder(points=(pt[:, None, :], torch.ones((1, 1), device=device)), boxes=None, masks=None)
            logit_low, _ = sam.mask_decoder(
                image_embeddings=emb,
                image_pe=base_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            logit = torch.nn.functional.interpolate(logit_low, size=(img_size, img_size), mode="bilinear", align_corners=False)
            loss = loss_fn(logit, msk)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
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
    args = pa.parse_args()

    scale_stats = load_scale_stats(args.scale_stats) if args.scale_stats else None

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
            scale_stats=scale_stats,
        )


if __name__ == "__main__":
    main()
