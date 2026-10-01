"""Command-line entry point.

    python -m parking_pricing.cli --plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import PricingConfig
from .pipeline import daily_summary, run_pipeline, write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parking-pricing",
        description="Run the three dynamic parking-pricing models over the dataset.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Input CSV (default: data/parking_dataset.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for prices.csv and daily_summary.csv (default: outputs).",
    )
    parser.add_argument(
        "--base-price",
        type=float,
        default=PricingConfig.base_price,
        help="Starting price for every lot (default: %(default)s).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=PricingConfig.competitor_radius_m,
        help="Competitor radius in metres (default: %(default)s).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Also render the comparison figures into the output directory.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print the model-comparison report and write it to the output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = PricingConfig(
        base_price=args.base_price,
        competitor_radius_m=args.radius,
    )

    prices = run_pipeline(args.data, cfg)
    paths = write_outputs(prices, args.out_dir)

    daily = daily_summary(prices)
    print(f"Priced {len(prices):,} lot-timestamps across {prices['lot_id'].nunique()} lots")
    print(f"  {prices['Timestamp'].min()}  ->  {prices['Timestamp'].max()}")
    print(f"  price band: {cfg.price_floor:.2f} - {cfg.price_ceiling:.2f}")
    for model in ("price_model1", "price_model2", "price_model3"):
        series = prices[model]
        print(f"  {model:<13} mean {series.mean():6.2f}  min {series.min():6.2f}  max {series.max():6.2f}")
    for name, path in paths.items():
        print(f"wrote {name}: {path}")

    if args.report:
        from .evaluate import compare_models, format_report

        report = format_report(prices, cfg)
        print()
        print(report)

        report_path = args.out_dir / "model_comparison.txt"
        report_path.write_text(report, encoding="utf-8")
        compare_models(prices, cfg).to_csv(args.out_dir / "model_comparison.csv")
        print(f"\nwrote report: {report_path}")

    if args.plots:
        from .plots import render_all

        for path in render_all(prices, daily, args.out_dir):
            print(f"wrote plot: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
