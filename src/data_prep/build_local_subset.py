"""Stratified subset of the training data for local GPU training.

The full 22,841-example set proved impractical on this laptop's GPU
(sustained load causes real thermal throttling -- see docs/build-log.md).
This subset is sized to keep a single training run short enough to
finish safely rather than requiring hours of continuous heavy load.
"""

import argparse
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=60)
    p.add_argument("--output", default=str(ROOT / "data" / "train_local_subset.parquet"))
    args = p.parse_args()

    df = pd.read_parquet(ROOT / "data" / "train.parquet")
    subset = (
        df.groupby("intent", group_keys=False)
        .sample(n=args.per_class, random_state=42)
        .reset_index(drop=True)
    )

    subset.to_parquet(args.output, index=False)
    print(f"Classes: {subset['intent'].nunique()}")
    print(f"Examples per class: {args.per_class}")
    print(f"Total: {len(subset)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
