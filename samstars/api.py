"""Top-level runtime API for samstars."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

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


@dataclass(slots=True)
class SamStarsResult:
    """Stable user-facing result object for segmentation runs."""

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
    """Runtime wrapper around a flat samstars model bundle."""

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

        if not self.decoder_path.exists():
            raise FileNotFoundError(f"Missing decoder checkpoint: {self.decoder_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Missing StarDist config: {self.config_path}")
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Missing StarDist weights: {self.weights_path}")

        self.defaults = self.manifest.get("defaults", {})

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
    ) -> Path:
        """Write the flat model-bundle manifest."""
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
        """Create a temporary nested StarDist directory expected by the low-level refinement path."""
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
        img: Path,
        chm: Path,
        den: Path,
        crowns: Path,
        sam_base_ckpt: Path,
        out: Path,
        storage: str,
        rgb_bands: tuple[int, int, int] | None = None,
        device: str = "cpu",
        chip_size: int = 512,
        chip_stride: int = 384,
        sam_epochs: int = 15,
        sam_min_iters: int = 0,
        sam_lr: float = 5e-5,
        stardist_epochs: int = 50,
        stardist_min_iters: int = 0,
        val_fraction: float = 0.2,
        crop_smooth_sigma: float = 2.0,
        smooth_sigma: float = 4.0,
        post_smooth_sigma: float = 0.0,
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
        stardist_normalize_inputs: bool = True,
        stardist_augment: bool = True,
        stardist_window_size_m: float | None = None,
        stardist_window_overlap_m: float = 0.0,
        stardist_dedup_iou_thr: float = 0.0,
        fast: bool = False,
    ) -> SamStars:
        """Train a flat samstars model bundle from one labeled area."""
        from samstars.data import build_scale_stats_from_proposal_records, build_single_tile
        from samstars.segmentation.engine import run_segmentation
        from samstars.training.chips import build_training_chips
        from samstars.training.records import build_stardist_training_records
        from samstars.training.sam_training import train_decoder
        from samstars.training.stardist_training import train_stardist_model

        storage = _normalize_storage(storage)
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for stale in (
            out / "decoder.pth",
            out / "config.json",
            out / "weights.h5",
            out / "thresholds.json",
            out / _BUNDLE_MANIFEST,
        ):
            if stale.exists():
                stale.unlink()

        with cls._work_root(out, storage, prefix="samstars-train-") as work_root:
            training_root = work_root / "_training" if storage == "persistent" else work_root
            prep_root = training_root / "prep" / "scene"
            training_chips_dir = training_root / "training_chips"
            training_run_dir = training_root / "training_run"
            records_dir = training_root / "stardist_training_records"
            scale_stats = training_root / "training_scale_stats.json"
            stardist_train_root = training_root / "_stardist_model"

            decoder_path = out / "decoder.pth"

            build_single_tile(
                img=Path(img),
                chm=Path(chm),
                den=Path(den),
                crowns=Path(crowns),
                rgb_bands=rgb_bands,
                out=prep_root,
                thin_seeds=thin_seeds,
                thin_dist=thin_dist,
                seed_d_min=seed_d_min,
                seed_min_dist_px=seed_min_dist_px,
                seed_gauss_sigma=seed_gauss_sigma,
                seed_min_samples=seed_min_samples,
                seed_merge_radius=seed_merge_radius,
                seed_min_canopy_height=seed_min_canopy_height,
                skip_existing=False,
            )

            build_training_chips(
                tiles_dir=prep_root,
                out_dir=training_chips_dir,
                chip_size=chip_size,
                stride=chip_stride,
            )

            train_decoder(
                chip_dir=training_chips_dir,
                ckpt=Path(sam_base_ckpt),
                out_path=decoder_path,
                epochs=sam_epochs,
                min_iters=sam_min_iters,
                lr=sam_lr,
                device=device,
            )

            proposal_result = run_segmentation(
                tiles_dir=prep_root,
                out_dir=training_run_dir,
                sam_ckpt=decoder_path,
                device=device,
                run_sam=True,
                run_stardist=False,
                fast=fast,
                crop_smooth_sigma=crop_smooth_sigma,
                smooth_sigma=smooth_sigma,
                post_smooth_sigma=post_smooth_sigma,
            )

            train_records, val_records = build_stardist_training_records(
                proposal_records=proposal_result["proposal_records"],
                tiles_dir=prep_root,
                out_dir=records_dir,
                val_fraction=val_fraction,
            )

            build_scale_stats_from_proposal_records(
                proposal_records=proposal_result["proposal_records"],
                out=scale_stats,
                percentiles=(2.0, 98.0),
                chm_key="chm",
                nodata_values=(-9999.0,),
            )

            train_stardist_model(
                train_records=train_records,
                val_records=val_records,
                out_dir=stardist_train_root,
                model_name="stardist",
                epochs=stardist_epochs,
                min_iters=stardist_min_iters,
                input_key=stardist_input_key,
                use_aux=stardist_use_aux,
                scale_stats_path=scale_stats if stardist_use_aux else None,
                normalize_inputs=stardist_normalize_inputs,
                augment=stardist_augment,
            )

            stardist_dir = stardist_train_root / "stardist"
            shutil.copy2(stardist_dir / "config.json", out / "config.json")
            weight_path = _find_stardist_weight(stardist_dir)
            shutil.copy2(weight_path, out / "weights.h5")
            if (stardist_dir / "thresholds.json").exists():
                shutil.copy2(stardist_dir / "thresholds.json", out / "thresholds.json")
            elif (out / "thresholds.json").exists():
                (out / "thresholds.json").unlink()

        prob_thr, nms_thr = cls._load_stardist_thresholds_from_dir(out)
        cls._write_bundle_manifest(
            out=out,
            defaults={
                "rgb_bands": list(rgb_bands) if rgb_bands is not None else None,
                "stardist_input_key": stardist_input_key,
                "stardist_use_aux": stardist_use_aux,
                "stardist_prob_thr": prob_thr,
                "stardist_nms_thr": nms_thr,
                "stardist_window_size_m": stardist_window_size_m,
                "stardist_window_overlap_m": stardist_window_overlap_m,
                "stardist_dedup_iou_thr": stardist_dedup_iou_thr,
                "crop_smooth_sigma": crop_smooth_sigma,
                "smooth_sigma": smooth_sigma,
                "post_smooth_sigma": post_smooth_sigma,
                "fast": fast,
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
        device: str = "cpu",
        stardist_input_key: str | None = None,
        stardist_use_aux: bool | None = None,
        stardist_prob_thr: float | None = None,
        stardist_nms_thr: float | None = None,
        stardist_window_size_m: float | None = None,
        stardist_window_overlap_m: float | None = None,
        stardist_dedup_iou_thr: float | None = None,
        stardist_cpu: bool = False,
        run_sam: bool = True,
        run_stardist: bool = True,
        skip_existing_sam: bool = True,
        fast: bool | None = None,
        crop_smooth_sigma: float | None = None,
        smooth_sigma: float | None = None,
        post_smooth_sigma: float | None = None,
    ) -> SamStarsResult:
        """Run segmentation using this model bundle."""
        from samstars.segmentation.engine import run_segmentation

        storage = _normalize_storage(storage)
        out = Path(out)
        if out.resolve() == self.root.resolve():
            raise ValueError("out must be different from the model bundle directory.")
        out.mkdir(parents=True, exist_ok=True)

        resolved_rgb_bands = rgb_bands
        if resolved_rgb_bands is None and self.defaults.get("rgb_bands") is not None:
            resolved_rgb_bands = tuple(self.defaults["rgb_bands"])

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
        resolved_fast = bool(self.defaults.get("fast", False)) if fast is None else fast
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
                    run_sam=run_sam,
                    run_stardist=run_stardist,
                    skip_existing_sam=skip_existing_sam,
                    fast=resolved_fast,
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
    img: Path,
    chm: Path,
    den: Path,
    crowns: Path,
    sam_base_ckpt: Path,
    out: Path,
    storage: str,
    rgb_bands: tuple[int, int, int] | None = None,
    device: str = "cpu",
    chip_size: int = 512,
    chip_stride: int = 384,
    sam_epochs: int = 15,
    sam_min_iters: int = 0,
    sam_lr: float = 5e-5,
    stardist_epochs: int = 50,
    stardist_min_iters: int = 0,
    val_fraction: float = 0.2,
    crop_smooth_sigma: float = 2.0,
    smooth_sigma: float = 4.0,
    post_smooth_sigma: float = 0.0,
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
    stardist_normalize_inputs: bool = True,
    stardist_augment: bool = True,
    stardist_window_size_m: float | None = None,
    stardist_window_overlap_m: float = 0.0,
    stardist_dedup_iou_thr: float = 0.0,
    fast: bool = False,
) -> SamStars:
    """Train a flat samstars model bundle and return its runtime wrapper."""
    return SamStars.train(
        img=img,
        chm=chm,
        den=den,
        crowns=crowns,
        sam_base_ckpt=sam_base_ckpt,
        out=out,
        storage=storage,
        rgb_bands=rgb_bands,
        device=device,
        chip_size=chip_size,
        chip_stride=chip_stride,
        sam_epochs=sam_epochs,
        sam_min_iters=sam_min_iters,
        sam_lr=sam_lr,
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
        fast=fast,
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
    device: str = "cpu",
    stardist_input_key: str | None = None,
    stardist_use_aux: bool | None = None,
    stardist_prob_thr: float | None = None,
    stardist_nms_thr: float | None = None,
    stardist_window_size_m: float | None = None,
    stardist_window_overlap_m: float | None = None,
    stardist_dedup_iou_thr: float | None = None,
    stardist_cpu: bool = False,
    run_sam: bool = True,
    run_stardist: bool = True,
    skip_existing_sam: bool = True,
    fast: bool | None = None,
    crop_smooth_sigma: float | None = None,
    smooth_sigma: float | None = None,
    post_smooth_sigma: float | None = None,
) -> SamStarsResult:
    """Run segmentation from a model bundle path or SamStars instance."""
    runtime = SamStars.load(model)
    return runtime.segment(
        out=out,
        storage=storage,
        tiles_dir=tiles_dir,
        img=img,
        chm=chm,
        den=den,
        rgb_bands=rgb_bands,
        device=device,
        stardist_input_key=stardist_input_key,
        stardist_use_aux=stardist_use_aux,
        stardist_prob_thr=stardist_prob_thr,
        stardist_nms_thr=stardist_nms_thr,
        stardist_window_size_m=stardist_window_size_m,
        stardist_window_overlap_m=stardist_window_overlap_m,
        stardist_dedup_iou_thr=stardist_dedup_iou_thr,
        stardist_cpu=stardist_cpu,
        run_sam=run_sam,
        run_stardist=run_stardist,
        skip_existing_sam=skip_existing_sam,
        fast=fast,
        crop_smooth_sigma=crop_smooth_sigma,
        smooth_sigma=smooth_sigma,
        post_smooth_sigma=post_smooth_sigma,
    )
