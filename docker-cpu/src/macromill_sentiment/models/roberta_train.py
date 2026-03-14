"""
RoBERTa fine-tuning for sentiment analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)
from tqdm.auto import tqdm


class SentimentDataset(Dataset):
    """PyTorch Dataset for IMDB sentiment data."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train_roberta(
    data_path: Path,
    artifact_dir: Path,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    max_samples: int | None = None,
    seed: int = 42,
    test_size: float = 0.2,
    local_files_only: bool = False,
    strip_html: bool = True,
    lowercase: bool = True,
):
    """Train RoBERTa for sentiment classification.
    
    Data split:
    - First: hold out test_size (default 20%) as test set (never seen during training)
    - Second: split remaining into train/val (90%/10%)
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from macromill_sentiment.data.preprocess import clean_text

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df = df.dropna()

    # Convert labels: positive=1, negative=0
    df["label"] = (df["sentiment"] == "positive").astype(int)

    if max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    texts = df["review"].tolist()
    labels = df["label"].tolist()

    # Apply preprocessing
    preprocess_cfg = type("PreprocessConfig", (), {"strip_html": strip_html, "lowercase": lowercase})()
    texts = [clean_text(t, preprocess_cfg) for t in texts]

    print(f"Total samples: {len(texts)}")

    # First split: hold out test set (never seen during training)
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    # Second split: train/val on remaining data (90%/10%)
    val_size = 0.1  # 10% of remaining = 8% of total
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels,
        test_size=val_size,
        random_state=seed,
        stratify=train_val_labels,
    )

    print(f"Split: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print("Loading RoBERTa model and tokenizer...")
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, local_files_only=local_files_only)
    model.to(device)

    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    # Training loop
    print(f"Starting training for {epochs} epoch(s)...")
    progress_bar = tqdm(total=num_training_steps, desc="Training")

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch,
            )

            loss = outputs.loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_batch = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels_batch).sum().item()
                total += labels_batch.size(0)

        val_accuracy = correct / total
        print(f"Epoch {epoch + 1}/{epochs} - Val Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy

    progress_bar.close()

    # Save model
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "pytorch_model.bin"
    tokenizer_path = artifact_dir

    print(f"Saving model to {artifact_dir}")
    model.save_pretrained(str(tokenizer_path))
    tokenizer.save_pretrained(str(tokenizer_path))

    # Save metadata with all split information
    meta = {
        "model_name": "roberta_base",
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "n_test": len(test_dataset),
        "test_size": test_size,
        "val_size": val_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "seed": seed,
        "max_samples": max_samples,
        "data_path": str(data_path),
        "best_val_accuracy": best_val_acc,
        "preprocess": {"strip_html": strip_html, "lowercase": lowercase},
    }

    meta_path = artifact_dir / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Training complete! Model saved to {artifact_dir}")
    return artifact_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--artifact-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    train_roberta(
        data_path=Path(args.data_path),
        artifact_dir=Path(args.artifact_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        max_samples=args.max_samples,
        seed=args.seed,
        test_size=args.test_size,
    )
