"""Batch-train and rollout NDSI forecasts for all local GeoTIFF datasets.

Scans subfolders under this directory for files named ndsi_YYYY-MM-DD.tif,
trains a lead=1 RandomForest model per dataset, then writes rollout predictions
up to an end date.

Example:
  /home/shadeform/.venv/bin/python batch_forecast_all.py --end-date 2026-02-28
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import joblib

from ndsi_forecast import list_daily_files, rollout_predictions, train_model


@dataclass(frozen=True)
class DatasetJob:
    name: str
    data_dir: Path


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def discover_datasets(root: Path) -> list[DatasetJob]:
    jobs: list[DatasetJob] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.startswith("ndsi_daily_"):
            continue
        tif_files = list(p.glob("ndsi_*.tif"))
        if not tif_files:
            continue
        jobs.append(DatasetJob(name=p.name, data_dir=p))
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train+rollout forecasts for all NDSI GeoTIFF sets.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--end-date", type=_parse_date, required=True)
    parser.add_argument("--lookback", type=int, default=14)
    parser.add_argument("--lead", type=int, default=1)
    parser.add_argument("--sample-pixels-per-day", type=int, default=2000)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=26)
    parser.add_argument("--models-dir", type=Path, default=Path("./models"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("./predictions"))

    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve()
    models_dir = args.models_dir.expanduser().resolve()
    predictions_dir = args.predictions_dir.expanduser().resolve()

    jobs = discover_datasets(root)
    if not jobs:
        print(f"No ndsi_daily_* datasets with ndsi_*.tif found under: {root}")
        return 2

    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(jobs)} datasets under {root}:")
    for j in jobs:
        files = list_daily_files(j.data_dir)
        first = files[0].name
        last = files[-1].name
        print(f"- {j.name}: {len(files)} tifs ({first} .. {last})")

    print("\n=== TRAIN + ROLLOUT ===")
    for j in jobs:
        files = list_daily_files(j.data_dir)
        last_date = datetime.strptime(files[-1].stem.split("ndsi_")[-1], "%Y-%m-%d").date()
        if args.end_date <= last_date:
            print(f"\n[{j.name}] Skip: end-date {args.end_date} <= last observed {last_date}")
            continue

        model_path = models_dir / f"{j.name}_rf_lb{args.lookback}_lead{args.lead}.joblib"
        out_dir = predictions_dir / f"{j.name}_lb{args.lookback}_lead{args.lead}_to_{args.end_date.isoformat()}"

        print(f"\n[{j.name}] Training model -> {model_path}")
        bundle = train_model(
            j.data_dir,
            lookback=args.lookback,
            lead=args.lead,
            sample_pixels_per_day=args.sample_pixels_per_day,
            random_state=42,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )
        joblib.dump(bundle, model_path)
        m = bundle["metrics"]
        print(f"[{j.name}] Test RMSE {m['rmse']:.4f} | MAE {m['mae']:.4f}")

        print(f"[{j.name}] Rolling out predictions -> {out_dir}")
        written = rollout_predictions(
            bundle,
            j.data_dir,
            start_date=None,
            end_date=args.end_date,
            out_dir=out_dir,
        )
        print(f"[{j.name}] Wrote {len(written)} GeoTIFFs")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
