"""QLoRA fine-tuning for ticket intent classification.

Runs as an Azure ML job on GPU compute (bitsandbytes 4-bit quantization
requires CUDA). Azure ML sets MLFLOW_TRACKING_URI automatically for jobs
in this workspace, so report_to="mlflow" below logs params/metrics/the
adapter to the workspace's MLflow tracking with no extra wiring.
"""

import argparse
import pathlib

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train-data", default=str(ROOT / "data" / "train.parquet"))
    p.add_argument("--output-dir", default=str(ROOT / "models" / "qlora-adapter"))
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-seq-length", type=int, default=512)
    return p.parse_args()


def build_example(labels: list[str], row: pd.Series, tokenizer) -> str:
    system = (
        "You are a customer support ticket classifier. Given a customer "
        "message, respond with exactly one intent label from this list, and "
        "nothing else:\n" + ", ".join(labels)
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": row["instruction"]},
        {"role": "assistant", "content": row["intent"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_df = pd.read_parquet(args.train_data)
    labels = sorted(train_df["intent"].unique().tolist())
    train_df = train_df.copy()
    train_df["text"] = train_df.apply(lambda r: build_example(labels, r, tokenizer), axis=1)
    train_dataset = Dataset.from_pandas(train_df[["text"]], preserve_index=False)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        logging_steps=20,
        save_strategy="epoch",
        bf16=True,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="mlflow",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
