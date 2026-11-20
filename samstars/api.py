"""Top-level runtime API for samstars."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from samstars._prompts import DEFAULT_FG_EXTRA, validate_fg_extra
from samstars._preprocessing import image_paths, inspect_images, training_contract, validate_contract

_BUNDLE_MANIFEST = "manifest.json"
_BUNDLE_FORMAT = "samstars_model_bundle"
_BUNDLE_VERSION = 1
_VALID_STORAGE = {"persistent", "temp"}


def _normalize_storage(storage: str) -> str:
    """Validate the requested storage policy."""
    if storage not in _VALID_STORAGE:
        raise ValueError(f"storage must be one of {sorted(_VALID_STORAGE)}, got {storage!r}")
    return storage


def _copy_or_symlink(src: Path, dst: Path) -> None:
    """Prefer symlink, fall back to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _find_stardist_weight(model_dir: Path) -> Path:
    """Pick a StarDist weights file from a trained model directory."""
    candidates = [
        model_dir / "weights_best.h5",
        model_dir / "weights_last.h5",
    ]
    for path in candidates:
        if path.exists():
            return path
    extra = sorted(model_dir.glob("*.h5"))
    if extra:
        return extra[0]
    raise FileNotFoundError(f"Missing StarDist weights in {model_dir}")


def _bundle_manifest_path(model: Path) -> Path:
    """Return the manifest path for a bundle directory or manifest path."""
    model = Path(model)
    if model.is_dir():
        return model / _BUNDLE_MANIFEST
    return model


def _resolve_training_tiles(
    *,
    tiles_dir: Path | None,
    img: Path | None,
    chm: Path | None,
    den: Path | None,
    crowns: Path | None,
    rgb_bands: tuple[int, int, int] | None,
    nir_band: int | None,
    texture_band: int | None,
    prep_root: Path,
    thin_seeds: bool,
    thin_dist: float,
    seed_d_min: float,
    seed_min_dist_px: int,
    seed_gauss_sigma: int,
    seed_min_samples: int,
    seed_merge_radius: float,
    seed_min_canopy_height: float,
    seed_rule_version: int = 1,
) -> Path:
    """Resolve training tiles from one raw labeled area, prepared tiles, or raw labeled tiles."""
    raw_inputs = (img, chm, den, crowns)
    if tiles_dir is not None and any(v is not None for v in raw_inputs):
        raise ValueError("Provide either tiles_dir or img/chm/den/crowns, not both.")
    if tiles_dir is None:
        if any(v is None for v in raw_inputs):
            raise ValueError("Provide either tiles_dir or all of img/chm/den/crowns.")
        from samstars.data import build_single_tile

        return Path(
            build_single_tile(
                img=Path(img),
                chm=Path(chm),
                den=Path(den),
                crowns=Path(crowns),
                rgb_bands=rgb_bands,
                nir_band=nir_band,
                texture_band=texture_band,
                out=prep_root / "scene",
                thin_seeds=thin_seeds,
                thin_dist=thin_dist,
                seed_d_min=seed_d_min,
                seed_min_dist_px=seed_min_dist_px,
                seed_gauss_sigma=seed_gauss_sigma,
                seed_min_samples=seed_min_samples,
                seed_merge_radius=seed_merge_radius,
                seed_min_canopy_height=seed_min_canopy_height,
                seed_rule_version=seed_rule_version,
                skip_existing=False,
            )
        )

    tiles_dir = Path(tiles_dir)
    if not tiles_dir.exists():
        raise FileNotFoundError(f"Missing training tiles directory: {tiles_dir}")
    if not tiles_dir.is_dir():
        raise NotADirectoryError(f"Training tiles path is not a directory: {tiles_dir}")

    tile_dirs = sorted(p for p in tiles_dir.iterdir() if p.is_dir())
    if not tile_dirs:
        raise ValueError(f"No tile directories found under {tiles_dir}")

    prepared_required = ("img.tif", "chm.tif", "cost.tif", "seeds.gpkg", "crowns.gpkg")
    raw_required = ("img.tif", "chm.tif", "den.tif", "crowns.gpkg")
    usable_prepared = []
    usable_raw = []
    missing_examples: list[str] = []
    for tile_dir in tile_dirs:
        missing_prepared = [name for name in prepared_required if not (tile_dir / name).exists()]
        if not missing_prepared:
            usable_prepared.append(tile_dir)
            continue
        missing_raw = [name for name in raw_required if not (tile_dir / name).exists()]
        if not missing_raw:
            usable_raw.append(tile_dir)
            continue
        if len(missing_examples) < 3:
            missing_examples.append(
                f"{tile_dir.name}: missing prepared [{', '.join(missing_prepared)}] and raw [{', '.join(missing_raw)}]"
            )

    if len(usable_prepared) == len(tile_dirs):
        return tiles_dir

    if usable_prepared and usable_raw:
        raise ValueError(
            "Training tiles must be consistently prepared or raw-labeled. "
            "Do not mix tile directories that already contain cost/seeds with ones that do not."
        )

    if len(usable_raw) == len(tile_dirs):
        from samstars.data import build_single_tile

        prepared_tiles_dir = prep_root / "tiles"
        for tile_dir in tile_dirs:
            build_single_tile(
                img=tile_dir / "img.tif",
                chm=tile_dir / "chm.tif",
                den=tile_dir / "den.tif",
                crowns=tile_dir / "crowns.gpkg",
                rgb_bands=rgb_bands,
                nir_band=nir_band,
                texture_band=texture_band,
                out=prepared_tiles_dir,
                tile_name=tile_dir.name,
                thin_seeds=thin_seeds,
                thin_dist=thin_dist,
                seed_d_min=seed_d_min,
                seed_min_dist_px=seed_min_dist_px,
                seed_gauss_sigma=seed_gauss_sigma,
                seed_min_samples=seed_min_samples,
                seed_merge_radius=seed_merge_radius,
                seed_min_canopy_height=seed_min_canopy_height,
                seed_rule_version=seed_rule_version,
                skip_existing=False,
            )
        return prepared_tiles_dir

    detail = "; ".join(missing_examples) if missing_examples else "no usable labeled tiles found"
    raise ValueError(
        "Training tiles must contain either prepared labeled tile contents "
        f"{prepared_required} or raw labeled tile contents {raw_required}: {detail}"
    )


@dataclass(slots=True)
class SamStarsResult:
    """Paths returned by a segmentation run.

    Attributes:
        out: Final output directory.
        crowns: Crown GeoPackage path, produced when refinement runs.
        labels_dir: Label-raster directory when exposed by the run, otherwise None.
        storage: Intermediate storage policy: persistent or temp.
        work_dir: Retained working directory for persistent runs, otherwise None.
    """

    out: Path
    crowns: Path
    labels_dir: Path | None
    storage: str
    work_dir: Path | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-serializable summary."""
        return {
            "out": str(self.out),
            "crowns": str(self.crowns),
            "labels_dir": str(self.labels_dir) if self.labels_dir is not None else None,
            "storage": self.storage,
            "work_dir": str(self.work_dir) if self.work_dir is not None else None,
        }


class SamStars:
    """Load a trained bundle and reuse its model files and stored defaults.

    Args:
        model: Bundle directory containing manifest.json and the trained artifacts.

    Use `train` to create a bundle and `segment` to apply it to source rasters
    or prepared tiles. Instance methods accept the same processing controls as
    the top-level functions.
    """

    def __init__(self, model: Path):
        self.root = Path(model)
        self.manifest_path = _bundle_manifest_path(self.root)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing model bundle manifest: {self.manifest_path}")
        self.root = self.manifest_path.parent
        self.manifest = json.load(open(self.manifest_path))
        if self.manifest.get("bundle_format") != _BUNDLE_FORMAT:
            raise ValueError(f"Unsupported model bundle format in {self.manifest_path}")

        self.decoder_path = self.root / self.manifest["decoder"]
        self.config_path = self.root / self.manifest["config"]
        self.weights_path = self.root / self.manifest["weights"]
        thresholds_name = self.manifest.get("thresholds")
        self.thresholds_path = self.root / thresholds_name if thresholds_name else None
        stats_name = self.manifest.get("scale_stats")
        self.scale_stats_path = self.root / stats_name if stats_name else None

        if not self.decoder_path.exists():
            raise FileNotFoundError(f"Missing decoder checkpoint: {self.decoder_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Missing StarDist config: {self.config_path}")
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Missing StarDist weights: {self.weights_path}")
        if self.scale_stats_path is not None and not self.scale_stats_path.exists():
            raise FileNotFoundError(f"Missing training scaling statistics: {self.scale_stats_path}")

        self.defaults = self.manifest.get("defaults", {})
        settings = self.manifest.get("preprocessing")
        self.preprocessing = validate_contract(settings) if settings is not None else None

    @classmethod
    def load(cls, model: Path | SamStars) -> SamStars:
        """Load a model bundle or return an existing SamStars object."""
        if isinstance(model, cls):
            return model
        return cls(Path(model))

    @staticmethod
    def _load_stardist_thresholds_from_dir(model_dir: Path) -> tuple[float | None, float | None]:
        """Read optimized StarDist thresholds when available."""
        thresholds_path = Path(model_dir) / "thresholds.json"
        if not thresholds_path.exists():
            return None, None
        data = json.load(open(thresholds_path))
        prob = data.get("prob")
        nms = data.get("nms")
        if prob is None:
            prob = data.get("prob_thresh")
        if nms is None:
            nms = data.get("nms_thresh")
        return prob, nms

    @staticmethod
    @contextmanager
    def _work_root(out: Path, storage: str, *, prefix: str) -> Iterator[Path]:
        """Yield the working root for a train or segment run."""
        storage = _normalize_storage(storage)
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        if storage == "persistent":
            yield out
            return

        with tempfile.TemporaryDirectory(prefix=prefix) as td:
            yield Path(td)

    @staticmethod
    def _write_bundle_manifest(
        *,
        out: Path,
        defaults: dict[str, Any],
        preprocessing: dict | None = None,
    ) -> Path:
        """Write the model-bundle manifest."""
        thresholds_name = "thresholds.json" if (out / "thresholds.json").exists() else None
        manifest = {
            "bundle_format": _BUNDLE_FORMAT,
            "bundle_version": _BUNDLE_VERSION,
            "decoder": "decoder.pth",
            "config": "config.json",
            "thresholds": thresholds_name,
            "weights": "weights.h5",
            "defaults": defaults,
        }
        if preprocessing is not None:
            manifest["preprocessing"] = validate_contract(preprocessing)
            if (out / "scale_stats.json").exists():
                manifest["scale_stats"] = "scale_stats.json"
        manifest_path = out / _BUNDLE_MANIFEST
        json.dump(manifest, open(manifest_path, "w"), indent=2)
        return manifest_path

    @staticmethod
    def _write_run_manifest(
        *,
        out: Path,
        storage: str,
        model: SamStars,
        work_dir: Path | None,
    ) -> Path:
        """Write a lightweight run manifest for segmentation outputs."""
        manifest = {
            "bundle_format": "samstars_run",
            "bundle_version": 1,
            "storage": storage,
            "model_bundle": str(model.root),
            "work_dir": str(work_dir) if work_dir is not None else None,
            "crowns": str(out / "crowns" / "crowns.gpkg"),
            "labels_dir": str(out / "labels") if (out / "labels").exists() else None,
        }
        if model.preprocessing is not None:
            manifest["preprocessing"] = model.preprocessing
        path = out / _BUNDLE_MANIFEST
        json.dump(manifest, open(path, "w"), indent=2)
        return path

    @staticmethod
    def _finalize_temp_outputs(
        *,
        temp_root: Path,
        out: Path,
    ) -> None:
        """Copy final user-facing outputs from a temp run root to the durable out directory."""
        crowns_src = temp_root / "crowns"
        labels_src = temp_root / "labels"
        crowns_dst = out / "crowns"
        labels_dst = out / "labels"

        if crowns_dst.exists():
            shutil.rmtree(crowns_dst)
        if crowns_src.exists():
            shutil.copytree(crowns_src, crowns_dst)

        if labels_dst.exists():
            shutil.rmtree(labels_dst)
        if labels_src.exists():
            shutil.copytree(labels_src, labels_dst)

    def _stage_stardist_model(self, stage_root: Path) -> Path:
        """Create the temporary StarDist directory used by refinement."""
        model_root = stage_root / "stardist_model"
        nested = model_root / "stardist"
        nested.mkdir(parents=True, exist_ok=True)
        _copy_or_symlink(self.config_path, nested / "config.json")
        _copy_or_symlink(self.weights_path, nested / "weights_best.h5")
        if self.thresholds_path is not None and self.thresholds_path.exists():
            _copy_or_symlink(self.thresholds_path, nested / "thresholds.json")
        return nested

    @classmethod
    def train(
        cls,
        *,
        tiles_dir: Path | None = None,
        img: Path | None = None,
        chm: Path | None = None,
        den: Path | None = None,
        crowns: Path | None = None,
        sam_base_ckpt: Path,
        out: Path,
        storage: str,
        rgb_bands: tuple[int, int, int] | None = None,
        nir_band: int | None = None,
        texture_band: int | None = None,
        device: str = "cpu",
        chip_size: int | None = None,
        chip_stride: int | None = None,
        sam_window_size_m: float | None = None,
        stardist_patch_size: int = 256,
        sam_epochs: int = 15,
        sam_min_iters: int = 0,
        sam_lr: float = 5e-5,
        fg_extra: int = DEFAULT_FG_EXTRA,
        stardist_epochs: int = 50,
        stardist_min_iters: int = 0,
        val_fraction: float = 0.2,
        crop_smooth_sigma: float | None = None,
        smooth_sigma: float | None = None,
        post_smooth_sigma: float | None = None,
        thin_seeds: bool = True,
        thin_dist: float = 0.8,
        seed_d_min: float = 4.5,
        seed_min_dist_px: int = 4,
        seed_gauss_sigma: int = 2,
        seed_min_samples: int = 1,
        seed_merge_radius: float = 3.0,
        seed_min_canopy_height: float = 2.0,
        stardist_input_key: str = "index",
        stardist_use_aux: bool = True,
        stardist_normalize_inputs: bool | None = None,
        stardist_augment: bool = True,
        stardist_window_size_m: float | None = None,
        stardist_window_overlap_m: float = 0.0,
        stardist_dedup_iou_thr: float = 0.0,
        stardist_cpu: bool = False,
        fast: bool = True,
        pipeline_mode: str = 'legacy',
        random_seed: int = 0,
        validation_tiles_dir: Path | None = None,
        stardist_steps_per_epoch: int = 100,
        embedding_cache_size: int = 64,
    ) -> SamStars:
        """Train a model bundle; fg_extra controls extra foreground prompts (0 = seed only)."""
        fg_extra = validate_fg_extra(fg_extra)
        from samstars._preprocessing import proposal_settings, positive_int
        crop_smooth_sigma, smooth_sigma, post_smooth_sigma = proposal_settings(
            pipeline_mode, crop_smooth_sigma, smooth_sigma, post_smooth_sigma)
        stardist_steps_per_epoch = positive_int(stardist_steps_per_epoch, 'stardist_steps_per_epoch')
        if isinstance(random_seed, bool) or not isinstance(random_seed, int) or not 0 <= random_seed < 2**32:
            raise ValueError('random_seed must be an integer in [0, 2**32).')
        if isinstance(embedding_cache_size, bool) or not isinstance(embedding_cache_size, int) or embedding_cache_size < 0:
            raise ValueError('embedding_cache_size must be a nonnegative integer.')
        if not 0 <= val_fraction < 1:
            raise ValueError('val_fraction must be in [0, 1).')
        if pipeline_mode == 'simple':
            stardist_input_key = 'file_name'
        from samstars._prompts import validate_canopy_height
        seed_min_canopy_height = validate_canopy_height(seed_min_canopy_height)
        seed_rule_version = 2 if pipeline_mode == 'simple' else 1
        prompt_height = seed_min_canopy_height if pipeline_mode == 'simple' else None
        if stardist_normalize_inputs is None:
            stardist_normalize_inputs = pipeline_mode != 'simple'
        preprocessing = training_contract(
            image_paths(tiles_dir, img), chip_size=chip_size, chip_stride=chip_stride,
            sam_window_size_m=sam_window_size_m, stardist_patch_size=stardist_patch_size,
        )
        chip_size = preprocessing["sam_window_size_px"]
        chip_stride = preprocessing["sam_stride_px"]
        from samstars.data import build_scale_stats_from_proposal_records
        from samstars.segmentation.engine import run_segmentation
        from samstars.training.chips import build_training_chips
        from samstars.training.records import build_stardist_training_records
        from samstars.training.sam_training import train_decoder

        storage = _normalize_storage(storage)
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for stale in (
            out / "decoder.pth",
            out / "config.json",
            out / "weights.h5",
            out / "thresholds.json",
            out / "scale_stats.json",
            out / _BUNDLE_MANIFEST,
        ):
            if stale.exists():
                stale.unlink()

        with cls._work_root(out, storage, prefix="samstars-train-") as work_root:
            training_root = work_root / "_training" if storage == "persistent" else work_root
            prep_root = training_root / "prep"
            training_chips_dir = training_root / "training_chips"
            training_run_dir = training_root / "training_run"
            records_dir = training_root / "stardist_training_records"
            scale_stats = training_root / "training_scale_stats.json"
            stardist_train_root = training_root / "_stardist_model"

            decoder_path = out / "decoder.pth"
            training_tiles_dir = _resolve_training_tiles(
                tiles_dir=tiles_dir,
                img=img,
                chm=chm,
                den=den,
                crowns=crowns,
                rgb_bands=rgb_bands,
                nir_band=nir_band,
                texture_band=texture_band,
                prep_root=prep_root,
                thin_seeds=thin_seeds,
                thin_dist=thin_dist,
                seed_d_min=seed_d_min,
                seed_min_dist_px=seed_min_dist_px,
                seed_gauss_sigma=seed_gauss_sigma,
                seed_min_samples=seed_min_samples,
                seed_merge_radius=seed_merge_radius,
                seed_min_canopy_height=seed_min_canopy_height,
                seed_rule_version=seed_rule_version,
            )

            from samstars.training.splits import split_training_tiles
            from samstars._preprocessing import inspect_images
            if validation_tiles_dir is not None:
                validation_tiles_dir = _resolve_training_tiles(
                    tiles_dir=validation_tiles_dir, img=None, chm=None, den=None, crowns=None,
                    rgb_bands=rgb_bands, nir_band=nir_band, texture_band=texture_band,
                    prep_root=prep_root/'validation', thin_seeds=thin_seeds, thin_dist=thin_dist,
                    seed_d_min=seed_d_min, seed_min_dist_px=seed_min_dist_px,
                    seed_gauss_sigma=seed_gauss_sigma, seed_min_samples=seed_min_samples,
                    seed_merge_radius=seed_merge_radius, seed_min_canopy_height=seed_min_canopy_height,
                    seed_rule_version=seed_rule_version)
                inspect_images(image_paths(validation_tiles_dir),
                    expected_resolution=preprocessing['resolution_m'], minimum_size=stardist_patch_size)
            split_root = training_root/'split'
            if split_root.exists():
                shutil.rmtree(split_root)
            training_tiles_dir, validation_tiles_dir, split_summary = split_training_tiles(
                training_tiles_dir, split_root, val_fraction=val_fraction, seed=random_seed,
                validation_tiles_dir=validation_tiles_dir)

            # Discard stale chips from earlier experiments in this working directory.
            if training_chips_dir.exists():
                shutil.rmtree(training_chips_dir)
            build_training_chips(
                tiles_dir=training_tiles_dir,
                out_dir=training_chips_dir,
                chip_size=chip_size,
                stride=chip_stride,
                shared_windows=pipeline_mode == 'simple',
                min_canopy_height=prompt_height,
            )

            train_decoder(
                chip_dir=training_chips_dir,
                ckpt=Path(sam_base_ckpt),
                out_path=decoder_path,
                img_size=chip_size,
                preprocessing=preprocessing,
                epochs=sam_epochs,
                min_iters=sam_min_iters,
                lr=sam_lr,
                device=device,
                fg_extra=fg_extra,
                pipeline_mode=pipeline_mode,
                random_seed=random_seed,
                embedding_cache_size=embedding_cache_size,
                min_canopy_height=prompt_height,
            )

            proposal_result = run_segmentation(
                tiles_dir=training_tiles_dir,
                out_dir=training_run_dir,
                sam_ckpt=decoder_path,
                device=device,
                run_sam=True,
                run_stardist=False,
                skip_existing_sam=False,
                preprocessing=preprocessing,
                fast=fast,
                pipeline_mode=pipeline_mode,
                random_seed=random_seed,
                min_canopy_height=prompt_height,
                fg_extra=fg_extra,
                crop_smooth_sigma=crop_smooth_sigma,
                smooth_sigma=smooth_sigma,
                post_smooth_sigma=post_smooth_sigma,
            )

            train_records, val_records = build_stardist_training_records(
                proposal_records=proposal_result["proposal_records"],
                tiles_dir=training_tiles_dir,
                out_dir=records_dir,
                val_fraction=0,
            )
            if validation_tiles_dir is not None:
                validation_result = run_segmentation(
                    tiles_dir=validation_tiles_dir, out_dir=training_root/'validation_run',
                    sam_ckpt=decoder_path, device=device, run_sam=True, run_stardist=False,
                    skip_existing_sam=False, preprocessing=preprocessing, fast=fast,
                    pipeline_mode=pipeline_mode, random_seed=random_seed, fg_extra=fg_extra,
                    min_canopy_height=prompt_height,
                    crop_smooth_sigma=crop_smooth_sigma, smooth_sigma=smooth_sigma,
                    post_smooth_sigma=post_smooth_sigma)
                val_records, _ = build_stardist_training_records(
                    proposal_records=validation_result['proposal_records'],
                    tiles_dir=validation_tiles_dir, out_dir=training_root/'validation_records', val_fraction=0)

            build_scale_stats_from_proposal_records(
                proposal_records=train_records,
                out=scale_stats,
                percentiles=(2.0, 98.0),
                chm_key="chm",
                nodata_values=(-9999.0,),
            )

            if stardist_cpu:
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "training" / "stardist_training.py"),
                    "--train-records",
                    str(train_records),
                    "--val-records",
                    str(val_records),
                    "--out-dir",
                    str(stardist_train_root),
                    "--model-name",
                    "stardist",
                    "--epochs",
                    str(stardist_epochs),
                    "--min-iters",
                    str(stardist_min_iters),
                    "--steps-per-epoch", str(stardist_steps_per_epoch),
                    "--seed", str(random_seed),
                    "--patch-size",
                    str(stardist_patch_size),
                    "--input-key",
                    stardist_input_key,
                ]
                if stardist_use_aux:
                    cmd.extend(["--use-aux", "--scale-stats", str(scale_stats)])
                if not stardist_normalize_inputs:
                    cmd.append("--no-normalize")
                if pipeline_mode == 'simple':
                    cmd.append('--geometric-only')
                if not stardist_augment:
                    cmd.append("--no-augment")
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = ""
                subprocess.run(cmd, check=True, env=env)
            else:
                from samstars.training.stardist_training import train_stardist_model

                train_stardist_model(
                    train_records=train_records,
                    val_records=val_records,
                    out_dir=stardist_train_root,
                    model_name="stardist",
                    epochs=stardist_epochs,
                    min_iters=stardist_min_iters,
                    steps_per_epoch=stardist_steps_per_epoch,
                    random_seed=random_seed,
                    intensity_augmentation=pipeline_mode != 'simple',
                    input_key=stardist_input_key,
                    use_aux=stardist_use_aux,
                    scale_stats_path=scale_stats if stardist_use_aux else None,
                    normalize_inputs=stardist_normalize_inputs,
                    augment=stardist_augment,
                    patch_size=stardist_patch_size,
                )

            stardist_dir = stardist_train_root / "stardist"
            shutil.copy2(scale_stats, out / "scale_stats.json")
            shutil.copy2(stardist_dir / "config.json", out / "config.json")
            metrics_path = stardist_dir/'training_metrics.json'
            if metrics_path.exists():
                shutil.copy2(metrics_path, out/'stardist_training.json')
            (out/'training_split.json').write_text(json.dumps(split_summary, indent=2))
            weight_path = _find_stardist_weight(stardist_dir)
            shutil.copy2(weight_path, out / "weights.h5")
            if (stardist_dir / "thresholds.json").exists():
                shutil.copy2(stardist_dir / "thresholds.json", out / "thresholds.json")
            elif (out / "thresholds.json").exists():
                (out / "thresholds.json").unlink()

        prob_thr, nms_thr = cls._load_stardist_thresholds_from_dir(out)
        cls._write_bundle_manifest(
            out=out,
            preprocessing=preprocessing,
            defaults={
                "rgb_bands": list(rgb_bands) if rgb_bands is not None else [1, 2, 3],
                "nir_band": nir_band,
                "texture_band": texture_band,
                "stardist_input_key": stardist_input_key,
                "stardist_use_aux": stardist_use_aux,
                "stardist_normalize_inputs": stardist_normalize_inputs,
                "stardist_prob_thr": prob_thr,
                "stardist_nms_thr": nms_thr,
                "stardist_window_size_m": stardist_window_size_m,
                "stardist_window_overlap_m": stardist_window_overlap_m,
                "stardist_dedup_iou_thr": stardist_dedup_iou_thr,
                "stardist_cpu": stardist_cpu,
                "crop_smooth_sigma": crop_smooth_sigma,
                "smooth_sigma": smooth_sigma,
                "post_smooth_sigma": post_smooth_sigma,
                "fast": fast,
                "fg_extra": fg_extra,
                "pipeline_mode": pipeline_mode,
                "canopy_rule_version": 1 if pipeline_mode == "simple" else None,
                "random_seed": random_seed,
                "seed_settings": dict(thin_seeds=thin_seeds, thin_dist=thin_dist,
                    seed_d_min=seed_d_min, seed_min_dist_px=seed_min_dist_px,
                    seed_gauss_sigma=seed_gauss_sigma, seed_min_samples=seed_min_samples,
                    seed_merge_radius=seed_merge_radius, seed_min_canopy_height=seed_min_canopy_height,
                    seed_rule_version=seed_rule_version),
            },
        )
        return cls(out)

    def segment(
        self,
        *,
        out: Path,
        storage: str,
        tiles_dir: Path | None = None,
        img: Path | None = None,
        chm: Path | None = None,
        den: Path | None = None,
        rgb_bands: tuple[int, int, int] | None = None,
        nir_band: int | None = None,
        texture_band: int | None = None,
        device: str = "cpu",
        stardist_input_key: str | None = None,
        stardist_use_aux: bool | None = None,
        stardist_prob_thr: float | None = None,
        stardist_nms_thr: float | None = None,
        stardist_window_size_m: float | None = None,
        stardist_window_overlap_m: float | None = None,
        stardist_dedup_iou_thr: float | None = None,
        stardist_cpu: bool = False,
        stardist_resume: bool = False,
        stardist_resume_from_labels: bool = False,
        stardist_skip_sources: str | None = None,
        stardist_skip_sources_file: Path | None = None,
        run_sam: bool = True,
        run_stardist: bool = True,
        skip_existing_sam: bool = True,
        log_refinement: bool = False,
        fast: bool | None = None,
        fg_extra: int | None = None,
        crop_smooth_sigma: float | None = None,
        smooth_sigma: float | None = None,
        post_smooth_sigma: float | None = None,
    ) -> SamStarsResult:
        """Run segmentation; fg_extra=None reuses the bundle count (80 for older bundles)."""
        resolved_fg_extra = validate_fg_extra(
            self.defaults.get("fg_extra", DEFAULT_FG_EXTRA) if fg_extra is None else fg_extra
        )
        if self.preprocessing is not None:
            inspect_images(image_paths(tiles_dir, img), expected_resolution=self.preprocessing["resolution_m"])
        from samstars.segmentation.engine import run_segmentation

        storage = _normalize_storage(storage)
        out = Path(out)
        if out.resolve() == self.root.resolve():
            raise ValueError("out must be different from the model bundle directory.")
        out.mkdir(parents=True, exist_ok=True)

        resolved_rgb_bands = rgb_bands
        if resolved_rgb_bands is None and self.defaults.get("rgb_bands") is not None:
            resolved_rgb_bands = tuple(self.defaults["rgb_bands"])
        resolved_nir_band = self.defaults.get("nir_band") if nir_band is None else nir_band
        resolved_texture_band = self.defaults.get("texture_band") if texture_band is None else texture_band

        resolved_stardist_input_key = stardist_input_key or self.defaults.get("stardist_input_key", "index")
        resolved_stardist_use_aux = (
            bool(self.defaults.get("stardist_use_aux", True))
            if stardist_use_aux is None
            else stardist_use_aux
        )
        resolved_prob_thr = self.defaults.get("stardist_prob_thr") if stardist_prob_thr is None else stardist_prob_thr
        resolved_nms_thr = self.defaults.get("stardist_nms_thr") if stardist_nms_thr is None else stardist_nms_thr
        resolved_window_size_m = (
            self.defaults.get("stardist_window_size_m")
            if stardist_window_size_m is None
            else stardist_window_size_m
        )
        resolved_window_overlap_m = (
            self.defaults.get("stardist_window_overlap_m", 0.0)
            if stardist_window_overlap_m is None
            else stardist_window_overlap_m
        )
        resolved_dedup_iou_thr = (
            self.defaults.get("stardist_dedup_iou_thr", 0.0)
            if stardist_dedup_iou_thr is None
            else stardist_dedup_iou_thr
        )
        resolved_fast = bool(self.defaults.get("fast", True)) if fast is None else fast
        resolved_crop_sigma = (
            float(self.defaults.get("crop_smooth_sigma", 2.0))
            if crop_smooth_sigma is None
            else crop_smooth_sigma
        )
        resolved_smooth_sigma = (
            float(self.defaults.get("smooth_sigma", 4.0))
            if smooth_sigma is None
            else smooth_sigma
        )
        resolved_post_sigma = (
            float(self.defaults.get("post_smooth_sigma", 0.0))
            if post_smooth_sigma is None
            else post_smooth_sigma
        )

        with self._work_root(out, storage, prefix="samstars-segment-") as work_root:
            with tempfile.TemporaryDirectory(prefix="samstars-stardist-") as model_stage:
                staged_stardist = self._stage_stardist_model(Path(model_stage))
                run_info = run_segmentation(
                    tiles_dir=tiles_dir,
                    img=img,
                    chm=chm,
                    den=den,
                    rgb_bands=resolved_rgb_bands,
                    nir_band=resolved_nir_band,
                    texture_band=resolved_texture_band,
                    out_dir=work_root,
                    sam_ckpt=self.decoder_path,
                    device=device,
                    stardist_model=staged_stardist,
                    stardist_input_key=resolved_stardist_input_key,
                    stardist_use_aux=resolved_stardist_use_aux,
                    stardist_prob_thr=resolved_prob_thr,
                    stardist_nms_thr=resolved_nms_thr,
                    stardist_window_size_m=resolved_window_size_m,
                    stardist_window_overlap_m=float(resolved_window_overlap_m or 0.0),
                    stardist_dedup_iou_thr=float(resolved_dedup_iou_thr or 0.0),
                    stardist_cpu=stardist_cpu,
                    stardist_resume=stardist_resume,
                    stardist_resume_from_labels=stardist_resume_from_labels,
                    stardist_skip_sources=stardist_skip_sources,
                    stardist_skip_sources_file=stardist_skip_sources_file,
                    run_sam=run_sam,
                    run_stardist=run_stardist,
                    skip_existing_sam=skip_existing_sam,
                    log_refinement=log_refinement,
                    fast=resolved_fast,
                    preprocessing=self.preprocessing,
                    stardist_normalize_inputs=bool(self.defaults.get("stardist_normalize_inputs", True)),
                    stardist_scale_stats=self.scale_stats_path,
                    fg_extra=resolved_fg_extra,
                    pipeline_mode=self.defaults.get('pipeline_mode', 'legacy'),
                    random_seed=self.defaults.get('random_seed', 0),
                    seed_settings=self.defaults.get('seed_settings'),
                    min_canopy_height=(self.defaults['seed_settings']['seed_min_canopy_height']
                        if self.defaults.get('canopy_rule_version') == 1 else None),
                    crop_smooth_sigma=resolved_crop_sigma,
                    smooth_sigma=resolved_smooth_sigma,
                    post_smooth_sigma=resolved_post_sigma,
                )

            if storage == "temp":
                self._finalize_temp_outputs(temp_root=work_root, out=out)
                work_dir = None
            else:
                work_dir = work_root

        self._write_run_manifest(
            out=out,
            storage=storage,
            model=self,
            work_dir=work_dir,
        )
        labels_dir = out / "labels" if (out / "labels").exists() else None
        return SamStarsResult(
            out=out,
            crowns=out / "crowns" / "crowns.gpkg",
            labels_dir=labels_dir,
            storage=storage,
            work_dir=work_dir,
        )


def train(
    *,
    tiles_dir: Path | None = None,
    img: Path | None = None,
    chm: Path | None = None,
    den: Path | None = None,
    crowns: Path | None = None,
    sam_base_ckpt: Path,
    out: Path,
    storage: str,
    rgb_bands: tuple[int, int, int] | None = None,
    nir_band: int | None = None,
    texture_band: int | None = None,
    device: str = "cpu",
    chip_size: int | None = None,
    chip_stride: int | None = None,
    sam_window_size_m: float | None = None,
    stardist_patch_size: int = 256,
    sam_epochs: int = 15,
    sam_min_iters: int = 0,
    sam_lr: float = 5e-5,
    fg_extra: int = DEFAULT_FG_EXTRA,
    stardist_epochs: int = 50,
    stardist_min_iters: int = 0,
    val_fraction: float = 0.2,
    crop_smooth_sigma: float | None = None,
    smooth_sigma: float | None = None,
    post_smooth_sigma: float | None = None,
    thin_seeds: bool = True,
    thin_dist: float = 0.8,
    seed_d_min: float = 4.5,
    seed_min_dist_px: int = 4,
    seed_gauss_sigma: int = 2,
    seed_min_samples: int = 1,
    seed_merge_radius: float = 3.0,
    seed_min_canopy_height: float = 2.0,
    stardist_input_key: str = "index",
    stardist_use_aux: bool = True,
    stardist_normalize_inputs: bool | None = None,
    stardist_augment: bool = True,
    stardist_window_size_m: float | None = None,
    stardist_window_overlap_m: float = 0.0,
    stardist_dedup_iou_thr: float = 0.0,
    stardist_cpu: bool = False,
    fast: bool = True,
    pipeline_mode: str = 'legacy',
    random_seed: int = 0,
    validation_tiles_dir: Path | None = None,
    stardist_steps_per_epoch: int = 100,
    embedding_cache_size: int = 64,
) -> SamStars:
    """Train a model bundle from a labeled scene or labeled tile directories.

    Args:
        tiles_dir: Directory of consistently raw or prepared labeled tiles.
            Supply this instead of the four scene inputs.
        img: RGB or multiband source imagery with a projected CRS and square pixels.
        chm: Canopy height raster in meters; aligned to img during preparation.
        den: First-return count or density raster; aligned to img during preparation.
        crowns: Crown GeoPackage with one polygon feature per tree.
        sam_base_ckpt: Base ViT-L checkpoint used to initialize proposal training.
        out: Destination model bundle directory. Use a new path for each experiment.
        storage: persistent retains intermediates under out/_training; temp
            retains only the finished bundle.
        rgb_bands: One-based source indices in red, green, blue order. Defaults
            to (1, 2, 3), without inferring a sensor from the band count.
        nir_band: Optional source NIR index for the NDVI cost term. Omit for RGB.
        texture_band: Optional source texture index; otherwise use mean RGB intensity.
        device: PyTorch device for proposal training: cpu, cuda, or mps.
        chip_size: SAM window width and height in native pixels (default 512).
            Supply this or sam_window_size_m, not both.
        chip_stride: SAM window spacing in native pixels; defaults to 75% of
            the window size, giving 25% overlap during training and prediction.
        sam_window_size_m: SAM ground-window width and height in metres, rounded
            to the nearest native pixel. Each window is resized to the fixed
            1024-pixel encoder input. Saved and reused for prediction.
        stardist_patch_size: Square refinement training patch in native pixels,
            a positive multiple of 16 (default 256). Every training/validation
            image must fit this patch. This is independent of the SAM window.
        sam_epochs: Number of proposal-decoder training epochs.
        sam_min_iters: Minimum proposal training iterations; may increase epochs.
        sam_lr: Proposal-decoder learning rate.
        fg_extra: Maximum additional foreground points per seed, a nonnegative
            integer. Zero keeps only the seed as foreground; the default is 80.
            Used for decoder training and the proposals that train refinement,
            and stored in the bundle for later segmentation.
        stardist_epochs: Number of refinement training epochs.
        stardist_min_iters: Minimum refinement training iterations.
        val_fraction: Fraction of refinement records assigned to validation.
            This is a record split, not a spatial holdout.
        crop_smooth_sigma: Proposal smoothing scale within each crop, in pixels.
        smooth_sigma: Proposal mosaic smoothing scale, in pixels.
        post_smooth_sigma: Optional smoothing after canopy-index fusion, in pixels.
        thin_seeds: Thin nearby seed points after detection.
        thin_dist: Seed thinning distance in projected CRS units.
        seed_d_min: Minimum density peak value; legacy mode also uses it for CHM.
        seed_min_dist_px: Seed peak-search radius in pixels.
        seed_gauss_sigma: Seed detection smoothing scale in pixels.
        seed_min_samples: Minimum samples in a seed cluster.
        seed_merge_radius: Candidate grouping distance in projected CRS units.
        seed_min_canopy_height: Canopy cutoff in metres; simple mode shares it
            between CHM peak selection, seed filtering, and positive/negative prompts.
        stardist_input_key: Refinement input record key, usually index.
        stardist_use_aux: Include normalized CHM as an auxiliary refinement channel.
        stardist_normalize_inputs: Normalize refinement inputs per channel.
        stardist_augment: Apply training augmentation to refinement samples.
        stardist_window_size_m: Default refinement window size for later segmentation,
            in map units, normally meters; None processes each proposal raster whole.
        stardist_window_overlap_m: Overlap between refinement windows in map units.
        stardist_dedup_iou_thr: Default final polygon deduplication threshold;
            zero disables it, positive values below one suppress matching overlaps.
        stardist_cpu: Run refinement training in a subprocess with CUDA hidden.
        fast: Batch prompt decoding within each fixed SAM window (default True). Does not
            change the window size or image scale for newly trained bundles.

    Returns:
        A SamStars instance backed by the saved bundle. Band settings and
        segmentation defaults are recorded in manifest.json. Prepared tiles
        retain their existing channel order and cost rasters.
    """
    return SamStars.train(
        tiles_dir=tiles_dir,
        img=img,
        chm=chm,
        den=den,
        crowns=crowns,
        sam_base_ckpt=sam_base_ckpt,
        out=out,
        storage=storage,
        rgb_bands=rgb_bands,
        nir_band=nir_band,
        texture_band=texture_band,
        device=device,
        chip_size=chip_size,
        chip_stride=chip_stride,
        sam_window_size_m=sam_window_size_m,
        stardist_patch_size=stardist_patch_size,
        sam_epochs=sam_epochs,
        sam_min_iters=sam_min_iters,
        sam_lr=sam_lr,
        fg_extra=fg_extra,
        stardist_epochs=stardist_epochs,
        stardist_min_iters=stardist_min_iters,
        val_fraction=val_fraction,
        crop_smooth_sigma=crop_smooth_sigma,
        smooth_sigma=smooth_sigma,
        post_smooth_sigma=post_smooth_sigma,
        thin_seeds=thin_seeds,
        thin_dist=thin_dist,
        seed_d_min=seed_d_min,
        seed_min_dist_px=seed_min_dist_px,
        seed_gauss_sigma=seed_gauss_sigma,
        seed_min_samples=seed_min_samples,
        seed_merge_radius=seed_merge_radius,
        seed_min_canopy_height=seed_min_canopy_height,
        stardist_input_key=stardist_input_key,
        stardist_use_aux=stardist_use_aux,
        stardist_normalize_inputs=stardist_normalize_inputs,
        stardist_augment=stardist_augment,
        stardist_window_size_m=stardist_window_size_m,
        stardist_window_overlap_m=stardist_window_overlap_m,
        stardist_dedup_iou_thr=stardist_dedup_iou_thr,
        stardist_cpu=stardist_cpu,
        fast=fast,
        pipeline_mode=pipeline_mode,
        random_seed=random_seed,
        validation_tiles_dir=validation_tiles_dir,
        stardist_steps_per_epoch=stardist_steps_per_epoch,
        embedding_cache_size=embedding_cache_size,
    )


def segment(
    *,
    model: Path | SamStars,
    out: Path,
    storage: str,
    tiles_dir: Path | None = None,
    img: Path | None = None,
    chm: Path | None = None,
    den: Path | None = None,
    rgb_bands: tuple[int, int, int] | None = None,
    nir_band: int | None = None,
    texture_band: int | None = None,
    device: str = "cpu",
    stardist_input_key: str | None = None,
    stardist_use_aux: bool | None = None,
    stardist_prob_thr: float | None = None,
    stardist_nms_thr: float | None = None,
    stardist_window_size_m: float | None = None,
    stardist_window_overlap_m: float | None = None,
    stardist_dedup_iou_thr: float | None = None,
    stardist_cpu: bool = False,
    stardist_resume: bool = False,
    stardist_resume_from_labels: bool = False,
    stardist_skip_sources: str | None = None,
    stardist_skip_sources_file: Path | None = None,
    run_sam: bool = True,
    run_stardist: bool = True,
    skip_existing_sam: bool = True,
    log_refinement: bool = False,
    fast: bool | None = None,
    fg_extra: int | None = None,
    crop_smooth_sigma: float | None = None,
    smooth_sigma: float | None = None,
    post_smooth_sigma: float | None = None,
) -> SamStarsResult:
    """Segment source rasters or prepared tiles using a trained model bundle.

    Args:
        model: Model bundle path or a loaded SamStars instance.
        out: Segmentation output directory, separate from the model bundle.
        storage: persistent keeps intermediates and checkpoints; temp retains
            final crown outputs after deleting temporary intermediates.
        tiles_dir: Prepared tile directories. Supply this instead of img/chm/den.
        img: Source imagery. Channel content must match the model's training data.
        chm: Canopy heights in meters, aligned to the imagery during preparation.
        den: First-return counts or density, consistent with training units.
        rgb_bands: One-based red, green, blue source indices. None reuses the bundle
            setting, or (1, 2, 3) when no setting was stored.
        nir_band: Optional NIR source index. None reuses the bundle setting.
        texture_band: Optional texture source index. None reuses the bundle setting.
        device: PyTorch device for proposals: cpu, cuda, or mps.
        stardist_input_key: Override the refinement input key stored in the bundle.
        stardist_use_aux: Override use of the normalized CHM refinement channel.
        stardist_prob_thr: Refinement probability threshold; set with stardist_nms_thr.
        stardist_nms_thr: Refinement suppression threshold; set with stardist_prob_thr.
        stardist_window_size_m: Internal refinement window size in projected map units.
        stardist_window_overlap_m: Window overlap in the same map units.
        stardist_dedup_iou_thr: Final cross-source polygon deduplication threshold.
            Zero disables it; positive values below one retain the higher-score
            polygon when pairwise intersection-over-union reaches the threshold.
        stardist_cpu: Run refinement in a subprocess with CUDA hidden.
        stardist_resume: Reuse complete refinement checkpoints in a persistent run.
        stardist_resume_from_labels: Rebuild checkpoints from labels; also requires
            stardist_resume. Recovered polygons have no confidence score.
        stardist_skip_sources: Comma-separated proposal source names to skip.
        stardist_skip_sources_file: Text file with one source name per line.
        run_sam: Run proposal generation. False reuses saved proposal records.
        run_stardist: Run refinement. False stops after proposal generation.
        skip_existing_sam: Skip complete proposal outputs already present under out.
        log_refinement: Show detailed refinement logs.
        fast: Override the bundle's batched proposal-decoding setting. None
            reuses the saved setting, or True when the bundle has no setting.
        fg_extra: Override the maximum additional foreground points per seed.
            None reuses the bundle setting, or 80 for older bundles. Zero keeps
            only the seed as foreground. Negative points and boxes are unchanged.
        crop_smooth_sigma: Override per-crop proposal smoothing, in pixels.
        smooth_sigma: Override proposal mosaic smoothing, in pixels.
        post_smooth_sigma: Override smoothing after canopy-index fusion, in pixels.

    Returns:
        Output paths in a SamStarsResult. Final crowns are written to
        out/crowns/crowns.gpkg when refinement runs. Overlapping tiles can leave
        overlapping polygons; deduplication does not guarantee disjoint crowns.

    Resume only with the same inputs, model, and settings. Use a new output
    directory when changing them. Band overrides apply to source preparation;
    prepared tile channels are used as supplied.
    """
    runtime = SamStars.load(model)
    return runtime.segment(
        out=out,
        storage=storage,
        tiles_dir=tiles_dir,
        img=img,
        chm=chm,
        den=den,
        rgb_bands=rgb_bands,
        nir_band=nir_band,
        texture_band=texture_band,
        device=device,
        stardist_input_key=stardist_input_key,
        stardist_use_aux=stardist_use_aux,
        stardist_prob_thr=stardist_prob_thr,
        stardist_nms_thr=stardist_nms_thr,
        stardist_window_size_m=stardist_window_size_m,
        stardist_window_overlap_m=stardist_window_overlap_m,
        stardist_dedup_iou_thr=stardist_dedup_iou_thr,
        stardist_cpu=stardist_cpu,
        stardist_resume=stardist_resume,
        stardist_resume_from_labels=stardist_resume_from_labels,
        stardist_skip_sources=stardist_skip_sources,
        stardist_skip_sources_file=stardist_skip_sources_file,
        run_sam=run_sam,
        run_stardist=run_stardist,
        skip_existing_sam=skip_existing_sam,
        log_refinement=log_refinement,
        fast=fast,
        fg_extra=fg_extra,
        crop_smooth_sigma=crop_smooth_sigma,
        smooth_sigma=smooth_sigma,
        post_smooth_sigma=post_smooth_sigma,
    )
