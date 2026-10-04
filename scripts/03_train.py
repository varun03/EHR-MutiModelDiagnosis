import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from dataloader.dataloader_ehrknowgen import PatientDataset, RecordListDataset
from dataloader.collate import collate_fn
from engine.trainer import (
    set_seed,
    make_vocab,
    build_sequence_vocab,
    split_records,
    train_lightweight,
    train_t5,
    train_knowgen,
)
from models.model_factory import create_model
from preprocessing.knowledge_graph import build_knowledge_graph, label_prior


def main(args):
    set_seed(args.seed)

    data_path = (
        Path(args.processed_dir)
        / "ehrknowgen_dataset.pkl"
    )

    dataset = PatientDataset(
        data_path,
        max_samples=args.max_samples
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset is empty. Run preprocessing first."
        )

    records = dataset.records

    train_records, val_records = split_records(
        records,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    val_hadm_ids = [
        r["hadm_id"]
        for r in val_records
    ]

    label_vocab = make_vocab(train_records)
    sequence_vocab = build_sequence_vocab(train_records)

    print(
        f"Records: {len(records)} "
        f"(train={len(train_records)}, val={len(val_records)})"
    )
    print(
        f"Labels: {len(label_vocab)}"
    )
    print(
        f"Sequence vocabulary: {len(sequence_vocab)}"
    )

    train_dataset = RecordListDataset(train_records)

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    Path(args.output_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    if args.model == "lightweight":

        model = create_model(
            model_type="lightweight",
            vocab_size=max(sequence_vocab.values()) + 1,
            num_labels=len(label_vocab),
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )

        train_lightweight(
            model=model,
            dataloader=dataloader,
            label_vocab=label_vocab,
            sequence_vocab=sequence_vocab,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )

        checkpoint_path = Path(args.output_dir) / "ehrknowgen_demo.pt"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "label_vocab": label_vocab,
                "sequence_vocab": sequence_vocab,
                "val_hadm_ids": val_hadm_ids,
            },
            checkpoint_path
        )

    elif args.model == "t5":

        model = create_model(
            model_type="t5",
            text_model_name=args.text_model_name,
        )

        train_t5(
            model=model,
            dataloader=dataloader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )

        checkpoint_path = Path(args.output_dir) / "ehrknowgen_demo_t5.pt"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "text_model_name": args.text_model_name,
                "val_hadm_ids": val_hadm_ids,
            },
            checkpoint_path
        )

    elif args.model == "knowgen":

        graph = build_knowledge_graph(
            args.mapping_dir,
            args.raw_dir,
            train_records,
        )

        print(
            f"Knowledge graph: {len(graph['nodes'])} nodes "
            f"({graph['num_coarse']} CCS + "
            f"{len(graph['nodes']) - graph['num_coarse']} ICD)"
        )

        model = create_model(
            model_type="knowgen",
            graph=graph,
            class_prior=label_prior(train_records),
            text_model_name=args.text_model_name,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )

        train_knowgen(
            model=model,
            dataloader=dataloader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )

        checkpoint_path = Path(args.output_dir) / "ehrknowgen_demo_knowgen.pt"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "text_model_name": args.text_model_name,
                "graph": graph,
                "val_hadm_ids": val_hadm_ids,
            },
            checkpoint_path
        )

    else:
        raise ValueError(
            f"Unknown model type: {args.model}"
        )

    print(
        f"Saved model to {checkpoint_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processed-dir",
        default="data/mimiciii/processed"
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints"
    )
    parser.add_argument(
        "--model",
        default="lightweight"
    )
    parser.add_argument(
        "--mapping-dir",
        default="data/mappings"
    )
    parser.add_argument(
        "--raw-dir",
        default="data/mimiciii/raw"
    )
    parser.add_argument(
        "--text-model-name",
        default="t5-small"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2
    )

    args = parser.parse_args()
    main(args)
