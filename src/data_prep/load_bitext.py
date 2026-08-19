"""Load the Bitext customer support dataset, explore it, and write a train/test split."""

import pathlib

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "data"


def load() -> pd.DataFrame:
    ds = load_dataset(DATASET_NAME, split="train")
    return ds.to_pandas()


def explore(df: pd.DataFrame) -> None:
    print(f"Total examples: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    print("\nIntent category counts:")
    counts = df["intent"].value_counts()
    print(counts)

    print(f"\nNumber of intent categories: {df['intent'].nunique()}")
    print(f"Smallest category: {counts.idxmin()} ({counts.min()} examples)")
    print(f"Largest category: {counts.idxmax()} ({counts.max()} examples)")

    print("\nSample rows:")
    print(df[["instruction", "intent"]].sample(5, random_state=42))


def split_and_save(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # The Bitext dataset's {{Placeholder}} tokens are literal, unfilled
    # strings, not substituted values -- so many rows share byte-identical
    # `instruction` text (2,237 of 26,872, confirmed always mapping to the
    # same intent). train_test_split only guarantees disjoint rows, not
    # disjoint text, so duplicate text leaks across the split regardless.
    # Deduplicating first (24,635 unique rows remain, class balance still
    # healthy -- smallest class 493 examples) closes that gap.
    before = len(df)
    df = df.drop_duplicates(subset="instruction", keep="first").reset_index(drop=True)
    print(f"\nDeduplicated by instruction text: {before} -> {len(df)} rows")

    train_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["intent"]
    )

    train_df.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    test_df.to_parquet(OUTPUT_DIR / "test.parquet", index=False)

    print(f"\nTrain set: {len(train_df)} examples -> {OUTPUT_DIR / 'train.parquet'}")
    print(f"Test set: {len(test_df)} examples -> {OUTPUT_DIR / 'test.parquet'}")


if __name__ == "__main__":
    df = load()
    explore(df)
    split_and_save(df)
