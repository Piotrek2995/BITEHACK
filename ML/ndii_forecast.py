"""Train a simple NDII forecasting model and export predictions as GeoTIFF.

Assumes a folder of daily NDII rasters named like:
  ndii_YYYY-MM-DD.tif

Higher NDII values are interpreted as wetter snow (less appealing for skiing).
This script predicts future NDII rasters; you can optionally also export a
"ski quality" raster in [0, 1] where higher means better (drier) snow.

Model: global RandomForest regression trained on sampled pixels using lagged
NDII values (lookback window) and calendar features (day-of-year sin/cos).
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import rasterio
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class ModelConfig:
    lookback: int
    lead: int
    nodata: float
    clip_min: float
    clip_max: float


def _parse_date_from_name(path: Path) -> date:
    match = _DATE_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse date from filename: {path.name}")
    return datetime.strptime(match.group(1), "%Y-%m-%d").date()


def list_daily_files(
    data_dir: Path,
    *,
    exts: tuple[str, ...] = (".tif", ".tiff", ".nc"),
    prefix: str = "ndii_",
) -> list[Path]:
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(str(data_dir))

    candidates: list[Path] = []
    for ext in exts:
        candidates.extend(sorted(data_dir.glob(f"{prefix}*{ext}")))

    dated: list[tuple[date, Path]] = []
    for p in candidates:
        try:
            d = _parse_date_from_name(p)
        except ValueError:
            continue
        dated.append((d, p))

    dated.sort(key=lambda x: x[0])
    return [p for _, p in dated]


def _read_tif_array(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        arr = np.where(arr == np.float32(nodata), np.nan, arr)
    return arr, profile


def _read_nc_array(path: Path) -> np.ndarray:
    ds = xr.open_dataset(path)
    try:
        if "ndii" in ds.data_vars:
            da = ds["ndii"]
        else:
            # fallback: first data variable
            da = next(iter(ds.data_vars.values()))
        arr = da.values.astype(np.float32)
        return arr
    finally:
        ds.close()


def _site_suffix_from_dir(data_dir: Path) -> str | None:
    name = data_dir.name
    if name.startswith("ndii_daily_"):
        return name.removeprefix("ndii_daily_")
    if name.startswith("ndsi_daily_"):
        return name.removeprefix("ndsi_daily_")
    return None


def _resolve_reference_profile(data_dir: Path, files: Sequence[Path]) -> Mapping:
    """Resolve a GeoTIFF profile to use for output georeferencing.

    - Prefer an NDII GeoTIFF within data_dir.
    - If the dataset is NetCDF-only, try a sibling ndsi_daily_<site> GeoTIFF.
    """
    # 1) Prefer a local NDII GeoTIFF
    for p in files:
        if p.suffix.lower() in {".tif", ".tiff"}:
            _, profile = _read_tif_array(p)
            return profile

    # 2) Try sibling NDSI dataset with same site suffix
    suffix = _site_suffix_from_dir(data_dir)
    if suffix:
        candidate_dir = data_dir.parent / f"ndsi_daily_{suffix}"
        if candidate_dir.exists():
            candidates = sorted(candidate_dir.glob("ndsi_*.tif"))
            if candidates:
                _, profile = _read_tif_array(candidates[0])
                return profile

    raise FileNotFoundError(
        "No GeoTIFFs found to infer georeferencing. "
        "Provide NDII GeoTIFF inputs or create a matching ndsi_daily_<site> folder with GeoTIFFs."
    )


def _read_array(path: Path, *, reference_profile: Mapping) -> tuple[np.ndarray, Mapping]:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return _read_tif_array(path)
    if suffix == ".nc":
        arr = _read_nc_array(path)
        return arr, reference_profile
    raise ValueError(f"Unsupported file type: {path}")


def _doy_sin_cos(d: date) -> tuple[float, float]:
    doy = d.timetuple().tm_yday
    theta = 2.0 * np.pi * (doy / 365.25)
    return float(np.sin(theta)), float(np.cos(theta))


def _compute_slope(series: np.ndarray) -> np.ndarray:
    t = np.arange(series.shape[0], dtype=np.float32)
    t_mean = float(t.mean())
    denom = float(((t - t_mean) ** 2).sum())
    if denom == 0.0:
        return np.zeros(series.shape[1], dtype=np.float32)

    y_mean = series.mean(axis=0)
    numer = ((t[:, None] - t_mean) * (series - y_mean[None, :])).sum(axis=0)
    return (numer / denom).astype(np.float32)


def build_training_data(
    files: Sequence[Path],
    *,
    lookback: int,
    lead: int,
    sample_pixels_per_day: int,
    random_state: int,
    reference_profile: Mapping,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if lead < 1:
        raise ValueError("lead must be >= 1")
    if len(files) < lookback + lead + 1:
        raise ValueError(
            f"Not enough files: need at least {lookback + lead + 1}, got {len(files)}"
        )

    rng = np.random.default_rng(random_state)

    first_arr, first_profile = _read_array(files[0], reference_profile=reference_profile)
    height, width = first_arr.shape

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for idx in range(lookback - 1, len(files) - lead):
        input_paths = files[idx - lookback + 1 : idx + 1]
        target_path = files[idx + lead]

        stack_list: list[np.ndarray] = []
        for p in input_paths:
            a, _ = _read_array(p, reference_profile=first_profile)
            if a.shape != (height, width):
                raise ValueError(f"Shape mismatch at {p.name}: {a.shape} != {(height, width)}")
            stack_list.append(a)

        target, _ = _read_array(target_path, reference_profile=first_profile)
        if target.shape != (height, width):
            raise ValueError(
                f"Shape mismatch at {target_path.name}: {target.shape} != {(height, width)}"
            )

        stack = np.stack(stack_list, axis=0)  # (lookback, H, W)

        valid = np.isfinite(target)
        valid &= np.isfinite(stack).all(axis=0)
        valid_count = int(valid.sum())
        if valid_count == 0:
            continue

        take = min(sample_pixels_per_day, valid_count)
        flat_valid = np.flatnonzero(valid.ravel())
        chosen = rng.choice(flat_valid, size=take, replace=False)

        series = stack.reshape(lookback, -1)[:, chosen]  # (lookback, take)
        y = target.ravel()[chosen].astype(np.float32)

        mean = series.mean(axis=0).astype(np.float32)
        std = series.std(axis=0).astype(np.float32)
        slope = _compute_slope(series)

        target_date = _parse_date_from_name(target_path)
        doy_sin, doy_cos = _doy_sin_cos(target_date)
        doy_sin_v = np.full(take, doy_sin, dtype=np.float32)
        doy_cos_v = np.full(take, doy_cos, dtype=np.float32)

        X = np.concatenate(
            [
                series.T.astype(np.float32),
                mean[:, None],
                std[:, None],
                slope[:, None],
                doy_sin_v[:, None],
                doy_cos_v[:, None],
            ],
            axis=1,
        )

        X_parts.append(X)
        y_parts.append(y)

    if not X_parts:
        raise ValueError("No valid training samples found. Check nodata/NaNs and filenames.")

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    meta = {
        "height": height,
        "width": width,
        "reference_profile": first_profile,
        "n_samples": int(X_all.shape[0]),
        "n_features": int(X_all.shape[1]),
    }
    return X_all, y_all, meta


def train_model(
    data_dir: Path,
    *,
    lookback: int = 14,
    lead: int = 1,
    sample_pixels_per_day: int = 2000,
    random_state: int = 42,
    n_estimators: int = 500,
    max_depth: int | None = 26,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
) -> dict:
    files = list_daily_files(data_dir)

    reference_profile = _resolve_reference_profile(Path(data_dir), files)

    X, y, meta = build_training_data(
        files,
        lookback=lookback,
        lead=lead,
        sample_pixels_per_day=sample_pixels_per_day,
        random_state=random_state,
        reference_profile=reference_profile,
    )

    # time-aware-ish split: last 20% as test
    n = X.shape[0]
    split = int(n * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))

    ref_profile = meta["reference_profile"]
    nodata = float(ref_profile.get("nodata") if ref_profile.get("nodata") is not None else -9999.0)

    bundle = {
        "model": model,
        "config": asdict(
            ModelConfig(
                lookback=lookback,
                lead=lead,
                nodata=nodata,
                clip_min=clip_min,
                clip_max=clip_max,
            )
        ),
        "reference_profile": ref_profile,
        "metrics": {"rmse": rmse, "mae": mae, "n_test": int(len(y_test))},
        "training": {"data_dir": str(Path(data_dir).resolve()), **meta},
    }
    return bundle


def _build_features_for_all_pixels(
    stack: np.ndarray,  # (lookback, H, W)
    *,
    target_date: date,
) -> tuple[np.ndarray, np.ndarray]:
    lookback, height, width = stack.shape
    flat = stack.reshape(lookback, -1)

    valid = np.isfinite(flat).all(axis=0)
    series = flat[:, valid]

    mean = series.mean(axis=0).astype(np.float32)
    std = series.std(axis=0).astype(np.float32)
    slope = _compute_slope(series)

    doy_sin, doy_cos = _doy_sin_cos(target_date)
    n_valid = series.shape[1]

    X_valid = np.concatenate(
        [
            series.T.astype(np.float32),
            mean[:, None],
            std[:, None],
            slope[:, None],
            np.full((n_valid, 1), doy_sin, dtype=np.float32),
            np.full((n_valid, 1), doy_cos, dtype=np.float32),
        ],
        axis=1,
    )

    return X_valid, valid


def _write_geotiff(array: np.ndarray, profile: dict, *, nodata: float, path: Path) -> Path:
    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        nodata=np.float32(nodata),
        compress="deflate",
        predictor=3,
    )

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array.astype(np.float32), 1)

    return path


def ndii_to_ski_quality(ndii: np.ndarray, *, clip_min: float, clip_max: float, nodata: float) -> np.ndarray:
    """Convert NDII to a simple ski quality index in [0,1].

    Assumption: higher NDII => wetter snow => worse skiing.

    Quality = 1 - normalized(NDII), where NDII is clipped to [clip_min, clip_max].
    """
    out = ndii.copy().astype(np.float32)
    valid = np.isfinite(out) & (out != np.float32(nodata))
    out_valid = np.clip(out[valid], clip_min, clip_max)

    denom = (clip_max - clip_min)
    if denom <= 0:
        q = np.zeros_like(out_valid, dtype=np.float32)
    else:
        norm = (out_valid - clip_min) / denom
        q = 1.0 - norm

    out[:] = np.float32(nodata)
    out[valid] = q.astype(np.float32)
    return out


def predict_to_tif(
    model_bundle: dict,
    data_dir: Path,
    *,
    target_date: date | None = None,
    output_path: Path,
    quality_path: Path | None = None,
) -> Path:
    cfg = model_bundle["config"]
    lookback = int(cfg["lookback"])
    lead = int(cfg["lead"])
    nodata = float(cfg["nodata"])
    clip_min = float(cfg.get("clip_min", -1.0))
    clip_max = float(cfg.get("clip_max", 1.0))

    files = list_daily_files(data_dir)
    if len(files) < lookback:
        raise ValueError(f"Not enough files in {data_dir} for lookback={lookback}")

    last_date = _parse_date_from_name(files[-1])
    if target_date is None:
        target_date = last_date + timedelta(days=lead)

    recent = files[-lookback:]
    reference_profile = model_bundle.get("reference_profile")
    if not isinstance(reference_profile, Mapping):
        raise ValueError("Model bundle missing reference_profile")

    first_arr, profile = _read_array(recent[0], reference_profile=reference_profile)
    height, width = first_arr.shape

    stack_list = [first_arr]
    for p in recent[1:]:
        a, _ = _read_array(p, reference_profile=profile)
        if a.shape != (height, width):
            raise ValueError(f"Shape mismatch at {p.name}: {a.shape} != {(height, width)}")
        stack_list.append(a)

    stack = np.stack(stack_list, axis=0)

    X_valid, valid_mask = _build_features_for_all_pixels(stack, target_date=target_date)
    model = model_bundle["model"]
    y_valid = model.predict(X_valid).astype(np.float32)
    y_valid = np.clip(y_valid, clip_min, clip_max)

    out = np.full(height * width, np.float32(nodata), dtype=np.float32)
    out[valid_mask] = y_valid
    out = out.reshape(height, width)

    written = _write_geotiff(out, profile, nodata=nodata, path=output_path)

    if quality_path is not None:
        q = ndii_to_ski_quality(out, clip_min=clip_min, clip_max=clip_max, nodata=nodata)
        _write_geotiff(q, profile, nodata=nodata, path=quality_path)

    return written


def rollout_predictions(
    model_bundle: dict,
    data_dir: Path,
    *,
    start_date: date | None,
    end_date: date,
    out_dir: Path,
    write_quality: bool = False,
) -> list[Path]:
    cfg = model_bundle["config"]
    lookback = int(cfg["lookback"])
    lead = int(cfg["lead"])
    nodata = float(cfg["nodata"])
    clip_min = float(cfg.get("clip_min", -1.0))
    clip_max = float(cfg.get("clip_max", 1.0))

    files = list_daily_files(data_dir)
    if len(files) < lookback:
        raise ValueError(f"Not enough files in {data_dir} for lookback={lookback}")

    last_date = _parse_date_from_name(files[-1])
    if start_date is None:
        start_date = last_date + timedelta(days=lead)
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")

    recent = files[-lookback:]
    reference_profile = model_bundle.get("reference_profile")
    if not isinstance(reference_profile, Mapping):
        raise ValueError("Model bundle missing reference_profile")

    first_arr, profile = _read_array(recent[0], reference_profile=reference_profile)
    height, width = first_arr.shape

    history: list[np.ndarray] = [first_arr]
    for p in recent[1:]:
        a, _ = _read_array(p, reference_profile=profile)
        if a.shape != (height, width):
            raise ValueError(f"Shape mismatch at {p.name}: {a.shape} != {(height, width)}")
        history.append(a)

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = model_bundle["model"]
    written: list[Path] = []

    current = start_date
    while current <= end_date:
        stack = np.stack(history[-lookback:], axis=0)
        X_valid, valid_mask = _build_features_for_all_pixels(stack, target_date=current)
        y_valid = model.predict(X_valid).astype(np.float32)
        y_valid = np.clip(y_valid, clip_min, clip_max)

        out = np.full(height * width, np.float32(nodata), dtype=np.float32)
        out[valid_mask] = y_valid
        out = out.reshape(height, width)

        out_path = out_dir / f"pred_ndii_{current.isoformat()}.tif"
        _write_geotiff(out, profile, nodata=nodata, path=out_path)
        written.append(out_path)

        if write_quality:
            q = ndii_to_ski_quality(out, clip_min=clip_min, clip_max=clip_max, nodata=nodata)
            q_path = out_dir / f"pred_quality_{current.isoformat()}.tif"
            _write_geotiff(q, profile, nodata=nodata, path=q_path)

        history.append(np.where(out == np.float32(nodata), np.nan, out).astype(np.float32))
        current = current + timedelta(days=lead)

    return written


def _parse_date_arg(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train NDII model and export predicted GeoTIFF (supports .tif/.tiff and .nc inputs)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train model on a folder of daily ndii_YYYY-MM-DD.tif")
    p_train.add_argument("--data-dir", required=True, type=Path)
    p_train.add_argument("--model-out", required=True, type=Path)
    p_train.add_argument("--lookback", type=int, default=14)
    p_train.add_argument("--lead", type=int, default=1)
    p_train.add_argument("--sample-pixels-per-day", type=int, default=2000)
    p_train.add_argument("--random-state", type=int, default=42)
    p_train.add_argument("--n-estimators", type=int, default=500)
    p_train.add_argument("--max-depth", type=int, default=26)

    p_pred = sub.add_parser("predict", help="Predict a future date GeoTIFF from last lookback days")
    p_pred.add_argument("--data-dir", required=True, type=Path)
    p_pred.add_argument("--model", required=True, type=Path)
    p_pred.add_argument("--target-date", type=_parse_date_arg, default=None)
    p_pred.add_argument("--out", required=True, type=Path)
    p_pred.add_argument(
        "--quality-out",
        type=Path,
        default=None,
        help="Optional GeoTIFF output for ski quality (higher=better, derived from NDII)",
    )

    p_roll = sub.add_parser("rollout", help="Recursive rollout predictions over a date range")
    p_roll.add_argument("--data-dir", required=True, type=Path)
    p_roll.add_argument("--model", required=True, type=Path)
    p_roll.add_argument("--start-date", type=_parse_date_arg, default=None)
    p_roll.add_argument("--end-date", type=_parse_date_arg, required=True)
    p_roll.add_argument("--out-dir", required=True, type=Path)
    p_roll.add_argument(
        "--write-quality",
        action="store_true",
        help="Also write pred_quality_YYYY-MM-DD.tif (quality in [0,1])",
    )

    args = parser.parse_args(argv)

    if args.cmd == "train":
        bundle = train_model(
            args.data_dir,
            lookback=args.lookback,
            lead=args.lead,
            sample_pixels_per_day=args.sample_pixels_per_day,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )
        model_out = args.model_out.expanduser().resolve()
        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, model_out)
        m = bundle["metrics"]
        print(f"Saved model: {model_out}")
        print(f"Test RMSE: {m['rmse']:.4f} | MAE: {m['mae']:.4f} | n_test: {m['n_test']}")
        return 0

    if args.cmd == "predict":
        bundle = joblib.load(args.model)
        out_path = predict_to_tif(
            bundle,
            args.data_dir,
            target_date=args.target_date,
            output_path=args.out,
            quality_path=args.quality_out,
        )
        print(f"Wrote prediction: {out_path}")
        if args.quality_out is not None:
            print(f"Wrote quality: {Path(args.quality_out).expanduser().resolve()}")
        return 0

    if args.cmd == "rollout":
        bundle = joblib.load(args.model)
        written = rollout_predictions(
            bundle,
            args.data_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            out_dir=args.out_dir,
            write_quality=bool(args.write_quality),
        )
        print(f"Wrote {len(written)} NDII predictions to: {Path(args.out_dir).expanduser().resolve()}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
