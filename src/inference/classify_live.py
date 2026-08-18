"""Interactive real-time classification: type a support message, see the
fine-tuned model's predicted intent immediately.

Complements the 270-example quantitative eval with a qualitative sanity
check on genuinely novel, non-templated input -- the Bitext test set is
templated/synthetic, so it can't tell us how the model handles messages
that don't match its template patterns (typos, rambling, slang, multiple
issues in one message, etc.). Shows the model's raw output rather than
snapping it to the nearest known label, so a garbled or off-format
response on an unusual message is visible rather than hidden.
"""

import pathlib

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 16

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ADAPTER_DIR = ROOT / "models" / "qlora-adapter"


def main() -> None:
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    labels = sorted(train_df["intent"].unique().tolist())

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

    system = (
        "You are a customer support ticket classifier. Given a customer "
        "message, respond with exactly one intent label from this list, and "
        "nothing else:\n" + ", ".join(labels)
    )

    print(f"\nModel loaded. {len(labels)} known intent categories.")
    print("Type a support message and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            message = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.lower() in ("quit", "exit"):
            break
        if not message:
            continue

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1] :]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()

        match = "✓ known label" if raw in labels else "⚠ not an exact known label"
        print(f"  Predicted: {raw}  ({match})\n")


if __name__ == "__main__":
    main()
