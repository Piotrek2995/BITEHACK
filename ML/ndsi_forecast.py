"""Train a simple NDSI forecasting model and export predictions as GeoTIFF.

This script assumes you have a folder of daily NDSI rasters named like:
  ndsi_YYYY-MM-DD.tif

It trains a global regression model on sampled pixels using lagged NDSI values
(lookback window) and calendar features, then predicts a future date raster.

Designed to be lightweight and dependency-minimal (numpy/rasterio/scikit-learn).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

import joblib
import numpy as np
import rasterio
from rasterio.io import DatasetReader
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
    exts: tuple[str, ...] = (".tif", ".tiff"),
    prefix: str = "ndsi_",
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


def _doy_sin_cos(d: date) -> tuple[float, float]:
    doy = d.timetuple().tm_yday
    theta = 2.0 * np.pi * (doy / 365.25)
    return float(np.sin(theta)), float(np.cos(theta))


def _compute_slope(series: np.ndarray) -> np.ndarray:
    """Compute linear slope over axis=0 for shape (lookback, n)."""
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

    first_arr, first_profile = _read_tif_array(files[0])
    height, width = first_arr.shape

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for idx in range(lookback - 1, len(files) - lead):
        input_paths = files[idx - lookback + 1 : idx + 1]
        target_path = files[idx + lead]

        stack_list: list[np.ndarray] = []
        for p in input_paths:
            a, _ = _read_tif_array(p)
            if a.shape != (height, width):
                raise ValueError(f"Shape mismatch at {p.name}: {a.shape} != {(height, width)}")
            stack_list.append(a)

        target, _ = _read_tif_array(target_path)
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
    lookback: int = 7,
    lead: int = 7,
    sample_pixels_per_day: int = 2000,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int | None = 24,
) -> dict:
    files = list_daily_files(data_dir)

    X, y, meta = build_training_data(
        files,
        lookback=lookback,
        lead=lead,
        sample_pixels_per_day=sample_pixels_per_day,
        random_state=random_state,
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

    # default output nodata: match typical files, fallback -9999
    ref_profile = meta["reference_profile"]
    nodata = float(ref_profile.get("nodata") if ref_profile.get("nodata") is not None else -9999.0)

    bundle = {
        "model": model,
        "config": asdict(
            ModelConfig(
                lookback=lookback,
                lead=lead,
                nodata=nodata,
                clip_min=-1.0,
                clip_max=1.0,
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

    series = flat[:, valid]  # (lookback, n_valid)
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


def predict_to_tif(
    model_bundle: dict,
    data_dir: Path,
    *,
    target_date: date | None = None,
    output_path: Path,
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

    # read reference/profile from first recent file
    first_arr, profile = _read_tif_array(recent[0])
    height, width = first_arr.shape

    stack_list = [first_arr]
    for p in recent[1:]:
        a, _ = _read_tif_array(p)
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

    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        nodata=np.float32(nodata),
        compress="deflate",
        predictor=3,
    )

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(out, 1)

    return output_path


def _parse_date_arg(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train NDSI model and export predicted GeoTIFF.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train model on a folder of daily ndsi_YYYY-MM-DD.tif")
    p_train.add_argument("--data-dir", required=True, type=Path)
    p_train.add_argument("--model-out", required=True, type=Path)
    p_train.add_argument("--lookback", type=int, default=7)
    p_train.add_argument("--lead", type=int, default=7)
    p_train.add_argument("--sample-pixels-per-day", type=int, default=2000)
    p_train.add_argument("--random-state", type=int, default=42)
    p_train.add_argument("--n-estimators", type=int, default=300)
    p_train.add_argument("--max-depth", type=int, default=24)

    p_pred = sub.add_parser("predict", help="Predict a future date GeoTIFF from last lookback days")
    p_pred.add_argument("--data-dir", required=True, type=Path)
    p_pred.add_argument("--model", required=True, type=Path)
    p_pred.add_argument("--target-date", type=_parse_date_arg, default=None)
    p_pred.add_argument("--out", required=True, type=Path)

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
        args.model_out = args.model_out.expanduser().resolve()
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, args.model_out)
        m = bundle["metrics"]
        print(f"Saved model: {args.model_out}")
        print(f"Test RMSE: {m['rmse']:.4f} | MAE: {m['mae']:.4f} | n_test: {m['n_test']}")
        return 0

    if args.cmd == "predict":
        bundle = joblib.load(args.model)
        out_path = predict_to_tif(bundle, args.data_dir, target_date=args.target_date, output_path=args.out)
        print(f"Wrote prediction: {out_path}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
