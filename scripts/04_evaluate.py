import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from dataloader.dataloader_ehrknowgen import PatientDataset, RecordListDataset
from dataloader.collate import collate_fn

from engine.trainer import (
    make_vocab,
    build_sequence_vocab,
)

from engine.evaluator import evaluate, evaluate_knowgen
from models.model_factory import create_model


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # -------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------
    data_path = (
        Path(args.processed_dir)
        / "ehrknowgen_dataset.pkl"
    )

    # Load every processed record here (not just --max-samples worth) so
    # filtering by the checkpoint's held-out val_hadm_ids below can't miss
    # records that --max-samples would otherwise have truncated away.
    dataset = PatientDataset(
        data_path,
        max_samples=0,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset is empty."
        )

    records = dataset.records

    # -------------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------------
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    print(f"Loaded checkpoint: {args.checkpoint}")

    # -------------------------------------------------------------
    # Restrict evaluation to the held-out validation split saved at
    # training time, so this doesn't just re-score the training data.
    # --max-samples only applies in --full mode, where there's no fixed
    # split to respect.
    # -------------------------------------------------------------
    if "val_hadm_ids" in checkpoint and not args.full:
        val_ids = set(checkpoint["val_hadm_ids"])

        records = [
            r for r in records
            if r["hadm_id"] in val_ids
        ]

        print(
            f"Evaluating on held-out validation split: "
            f"{len(records)} records"
        )

    elif not args.full:
        if args.max_samples > 0:
            records = records[:args.max_samples]

        print(
            "Checkpoint has no stored validation split "
            "(trained before the train/val split was added) - "
            "evaluating on all loaded records. This is NOT a "
            "true holdout evaluation."
        )

    else:
        if args.max_samples > 0:
            records = records[:args.max_samples]

        print(
            f"--full set: evaluating on all {len(records)} loaded "
            "records, including any used for training."
        )

    if len(records) == 0:
        raise RuntimeError(
            "No records left to evaluate. Check --processed-dir / "
            "--max-samples, or pass --full."
        )

    eval_dataset = RecordListDataset(records)

    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # -------------------------------------------------------------
    # Build fallback vocabularies
    # (used only if checkpoint doesn't contain them)
    # -------------------------------------------------------------
    if "graph" in checkpoint:
        model = create_model(
            model_type="knowgen",
            graph=checkpoint["graph"],
            text_model_name=checkpoint.get("text_model_name", "t5-small"),
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        metrics = evaluate_knowgen(model, dataloader, device)

        print("\n================ Evaluation ================\n")
        for key, value in metrics.items():
            print(f"{key:20s}: {value:.4f}")
        return

    label_vocab = make_vocab(records)
    sequence_vocab = build_sequence_vocab(records)

    # -------------------------------------------------------------
    # Use vocabularies stored in checkpoint
    # -------------------------------------------------------------
    if "label_vocab" in checkpoint:
        label_vocab = checkpoint["label_vocab"]

    if "sequence_vocab" in checkpoint:
        sequence_vocab = checkpoint["sequence_vocab"]

    vocab_size = max(sequence_vocab.values()) + 1
    num_labels = len(label_vocab)

    print(f"Vocabulary Size : {vocab_size}")
    print(f"Number of Labels: {num_labels}")

    # -------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------
    model = create_model(
        model_type="lightweight",
        vocab_size=vocab_size,
        num_labels=num_labels,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    # -------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------
    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        label_vocab=label_vocab,
        sequence_vocab=sequence_vocab,
        device=device,
    )

    print("\n================ Evaluation ================\n")

    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate Lightweight EHR-KnowGen"
    )

    parser.add_argument(
        "--processed-dir",
        default="data/mimiciii/processed",
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/ehrknowgen_demo.pt",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate on all loaded records instead of just the "
             "held-out validation split stored in the checkpoint.",
    )

    args = parser.parse_args()

    main(args)