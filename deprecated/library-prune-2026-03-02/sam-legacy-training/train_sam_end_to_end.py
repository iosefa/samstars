#!/usr/bin/env python3
"""
Prepare training chips and fine‑tune the SAM‑ViT‑L mask decoder in one run.
Supports single raster + crowns or multiple tile/crowns pairs.
"""
import argparse, json, numpy as np, torch, torch.nn.functional as F
import rasterio, geopandas as gpd
from pathlib import Path
from shapely.geometry import box
from rasterio.features import rasterize
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

ROOT = Path(__file__).parent


# ─── chip creation ────────────────────────────────────────────────────────────
def prepare_chips_for_pair(image_path, crowns_path, outdir, img_size=512, overlap=480, rgb_bands=(6, 4, 2), prefix=None):
    """Create chips for one image/crowns pair."""
    outdir = Path(outdir)
    for sub in ("img", "msk", "json"):
        (outdir / sub).mkdir(parents=True, exist_ok=True)

    with rasterio.open(image_path) as src:
        rgb = np.stack([src.read(b + 1) for b in rgb_bands])
        rgb8 = ((rgb - rgb.min()) / (np.ptp(rgb) + 1e-9) * 255).astype("uint8")
        transform = src.transform
        height, width = src.height, src.width
        raster_bounds = box(*src.bounds)
        crs = src.crs

    crowns = gpd.read_file(crowns_path).to_crs(crs)
    crowns = crowns[crowns.geometry.is_valid & crowns.intersects(raster_bounds)]

    mask = rasterize(
        [(geom, 1) for geom in crowns.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )

    stride, tid, prefix = img_size - overlap, 0, (prefix or Path(image_path).stem)
    for y0 in range(0, height, stride):
        for x0 in range(0, width, stride):
            y1, x1 = min(y0 + img_size, height), min(x0 + img_size, width)
            chip = rgb8[:, y0:y1, x0:x1]
            msk = mask[y0:y1, x0:x1]
            if msk.sum() == 0:
                continue

            pad_h, pad_w = img_size - chip.shape[1], img_size - chip.shape[2]
            if pad_h or pad_w:
                chip = np.pad(chip, ((0, 0), (0, pad_h), (0, pad_w)))
                msk = np.pad(msk, ((0, pad_h), (0, pad_w)))

            ys, xs = np.where(msk)
            chip_id = f"{prefix}_{tid:06d}"
            json.dump(
                dict(
                    point=[int(xs.mean()), int(ys.mean())],
                    box=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                ),
                open(outdir / "json" / f"{chip_id}.json", "w"),
            )
            np.save(outdir / "img" / f"{chip_id}.npy", chip)
            np.save(outdir / "msk" / f"{chip_id}.npy", msk)
            tid += 1
    return tid


def prepare_chips(pairs, outdir, img_size=512, overlap=480, rgb_bands=(6, 4, 2)):
    """Prepare chips."""
    total = 0
    for image_path, crowns_path in pairs:
        count = prepare_chips_for_pair(image_path, crowns_path, outdir, img_size, overlap, rgb_bands)
        print(f"{Path(image_path).name}: {count} chips")
        total += count
    return total


# ─── dataset / training ───────────────────────────────────────────────────────
class Chips(Dataset):
    def __init__(self, chip_dir, img_size=512, enc_size=1024):
        """Initialize the instance."""
        self.chip_dir = Path(chip_dir)
        self.ids = sorted((self.chip_dir / "img").glob("*.npy"))
        if not self.ids:
            raise FileNotFoundError(f"No chips found under {self.chip_dir}")
        self.img_size = img_size
        self.resize = ResizeLongestSide(enc_size)

    def __len__(self):
        """Return the number of items."""
        return len(self.ids)

    def __getitem__(self, i):
        """Return one item by index."""
        name = self.ids[i].stem
        img = np.load(self.chip_dir / "img" / f"{name}.npy")
        msk = np.load(self.chip_dir / "msk" / f"{name}.npy")
        pt = np.array(json.load(open(self.chip_dir / "json" / f"{name}.json"))["point"])

        img_u8 = img.astype(np.uint8)
        img = self.resize.apply_image(img_u8.transpose(1, 2, 0)).transpose(2, 0, 1) / 255.0
        pt = self.resize.apply_coords(pt[None], (self.img_size, self.img_size))[0]
        return (
            torch.tensor(img).float(),
            torch.tensor(msk[None]).float(),
            torch.tensor(pt).float(),
        )


def train_decoder(chip_dir, ckpt_path, out_path, img_size=512, enc_size=1024, epochs=15, lr=5e-5, device="cpu"):
    """Train decoder."""
    loader = DataLoader(Chips(chip_dir, img_size, enc_size), batch_size=1, shuffle=True)

    sam = sam_model_registry["vit_l"](checkpoint=ckpt_path).to(device)
    sam.image_encoder.requires_grad_(False)
    sam.prompt_encoder.requires_grad_(False)

    optimizer = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=lr)
    base_pe = sam.prompt_encoder.get_dense_pe().to(device)

    def loss_fn(pred, truth):
        """Loss fn."""
        score = torch.sigmoid(pred)
        dice = 1 - (2 * (score * truth).sum((2, 3)) + 1) / (score.sum((2, 3)) + truth.sum((2, 3)) + 1)
        return (dice + F.binary_cross_entropy_with_logits(pred, truth)).mean()

    for ep in range(1, epochs + 1):
        tot = 0
        for img, msk, pt in loader:
            img, msk, pt = img.to(device), msk.to(device), pt.to(device)
            with torch.no_grad():
                emb = sam.image_encoder(sam.preprocess(img))
            point_labels = torch.ones((1, 1), device=device)
            sparse, _ = sam.prompt_encoder(points=(pt[:, None, :], point_labels), boxes=None, masks=None)
            logit_low, _ = sam.mask_decoder(
                image_embeddings=emb,
                image_pe=base_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=base_pe,
                multimask_output=False,
            )
            logit = torch.nn.functional.interpolate(logit_low, size=(img_size, img_size), mode="bilinear", align_corners=False)

            loss = loss_fn(logit, msk)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item()
        print(f"epoch {ep:02d}/{epochs}  loss={tot/len(loader):.4f}")

    torch.save(sam.state_dict(), out_path)


def resolve_pairs(args):
    """Resolve pairs."""
    pairs = []
    if args.tiles:
        pairs.extend([(Path(img), Path(crowns)) for img, crowns in args.tiles])
    if args.tiles_dir:
        for tile in sorted(Path(args.tiles_dir).iterdir()):
            if not tile.is_dir():
                continue
            img = tile / args.image_name
            crowns = tile / args.crowns_name
            if img.exists() and crowns.exists():
                pairs.append((img, crowns))
            else:
                missing = img if not img.exists() else crowns
                print(f"Skipping {tile}: missing {missing}")
    if not pairs:
        pairs.append((args.image, args.crowns))
    return pairs


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare chips and fine-tune the SAM decoder")
    parser.add_argument("--image", default=ROOT / "image.tif", type=Path, help="8-band WorldView raster")
    parser.add_argument("--crowns", default=ROOT / "crowns.gpkg", type=Path, help="Polygon crowns (same CRS as image)")
    parser.add_argument(
        "--tile",
        dest="tiles",
        nargs=2,
        action="append",
        metavar=("IMAGE", "CROWNS"),
        help="Repeatable image/crowns pair; use instead of --image/--crowns",
    )
    parser.add_argument("--tiles-dir", type=Path, help="Directory with tile subfolders")
    parser.add_argument("--image-name", default="image.tif", help="Image filename within each tile folder when using --tiles-dir")
    parser.add_argument("--crowns-name", default="crowns.gpkg", help="Crowns filename within each tile folder when using --tiles-dir")
    parser.add_argument("--chip-dir", default=ROOT / "data" / "chips", type=Path, help="Output directory for chips")
    parser.add_argument("--ckpt", default=ROOT / "sam_vit_l_0b3195.pth", type=Path, help="SAM ViT-L checkpoint")
    parser.add_argument("--out", default=ROOT / "sam_crowns_decoder.pth", type=Path, help="Destination for fine-tuned weights")
    parser.add_argument("--img-size", default=512, type=int, help="Chip size (pixels)")
    parser.add_argument("--overlap", default=480, type=int, help="Chip overlap (pixels)")
    parser.add_argument("--rgb-bands", default=(6, 4, 2), nargs=3, type=int, help="0-based RGB band indices")
    parser.add_argument("--epochs", default=3, type=int, help="Training epochs")
    parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate")
    parser.add_argument("--skip-chips", action="store_true", help="Skip chip generation and reuse existing chips")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu or cuda)")
    return parser.parse_args()


def main():
    """Run the command-line entry point."""
    args = parse_args()
    pairs = resolve_pairs(args)
    if not args.skip_chips:
        count = prepare_chips(pairs, args.chip_dir, args.img_size, args.overlap, tuple(args.rgb_bands))
        print(f"✓ {count} chips ➜ {args.chip_dir}")
    else:
        print(f"Skipping chip creation; using chips at {args.chip_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    train_decoder(args.chip_dir, args.ckpt, args.out, args.img_size, 1024, args.epochs, args.lr, device=args.device)
    print(f"✓ weights → {args.out}")


if __name__ == "__main__":
    main()
