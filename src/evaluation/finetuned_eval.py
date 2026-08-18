"""Fine-tuned model eval: same 270-example stratified sample and same
prompt/parsing logic as baseline_eval.py, but with the QLoRA adapter
(trained on Colab, see docs/build-log.md) applied on top of the base
model, so the before/after comparison is apples-to-apples.

Runs on local GPU. This is pure inference (no training), so the thermal
throttling that ruled out local training doesn't apply here -- a few
hundred short generations is nowhere near the sustained load that
triggered it.
"""

import json
import pathlib

import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SAMPLES_PER_CLASS = 10
MAX_NEW_TOKENS = 16

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ADAPTER_DIR = ROOT / "models" / "qlora-adapter"
RESULTS_DIR = ROOT / "results" / "finetuned"


def load_eval_sample() -> pd.DataFrame:
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    sample = test_df.groupby("intent", group_keys=False).sample(
        n=SAMPLES_PER_CLASS, random_state=42
    )
    return sample.reset_index(drop=True)


def build_prompt(tokenizer, labels: list[str], message: str) -> str:
    system = (
        "You are a customer support ticket classifier. Given a customer "
        "message, respond with exactly one intent label from this list, and "
        "nothing else:\n" + ", ".join(labels)
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": message},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_prediction(raw: str, labels: list[str]) -> str:
    cleaned = raw.strip().lower().strip(".\"'")
    for label in labels:
        if cleaned == label.lower():
            return label
    for label in labels:
        if label.lower() in cleaned:
            return label
    return "UNPARSEABLE"


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME} with adapter from {ADAPTER_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    sample = load_eval_sample()
    labels = sorted(sample["intent"].unique().tolist())
    print(f"Evaluating on {len(sample)} examples across {len(labels)} classes")

    predictions = []
    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        prompt = build_prompt(tokenizer, labels, row["instruction"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1] :]
        raw_text = tokenizer.decode(generated, skip_special_tokens=True)
        predictions.append(parse_prediction(raw_text, labels))

    sample = sample.copy()
    sample["predicted_intent"] = predictions

    accuracy = (sample["predicted_intent"] == sample["intent"]).mean()
    report = classification_report(
        sample["intent"], sample["predicted_intent"], labels=labels, zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(sample["intent"], sample["predicted_intent"], labels=labels)
    unparseable_rate = (sample["predicted_intent"] == "UNPARSEABLE").mean()

    print(f"\nAccuracy: {accuracy:.3f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.3f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.3f}")
    print(f"Unparseable output rate: {unparseable_rate:.3f}")
    print("\nBaseline for comparison: accuracy 0.556, macro F1 0.483, weighted F1 0.483")

    metrics = {
        "model": f"{MODEL_NAME} + QLoRA adapter (1 epoch, 9990 examples, Colab T4)",
        "eval_type": "fine-tuned",
        "n_samples": len(sample),
        "samples_per_class": SAMPLES_PER_CLASS,
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "unparseable_rate": unparseable_rate,
        "per_class_report": report,
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    sample.to_csv(RESULTS_DIR / "predictions.csv", index=False)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Fine-tuned model confusion matrix ({MODEL_NAME} + QLoRA)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)

    print(f"\nSaved metrics, predictions, and confusion matrix to {RESULTS_DIR}")


if __name__ == "__main__":
    run()
