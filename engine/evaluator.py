import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    label_vocab,
    sequence_vocab,
    device,
):
    """
    Evaluate the lightweight EHR-KnowGen model.
    """

    from engine.trainer import (
        encode_css,
        encode_labels,
        encode_sequence,
    )

    model.eval()

    all_true = []
    all_pred = []

    for batch in dataloader:

        records = batch["records"]

        sequences = []
        masks = []

        # ------------------------------------------------------------
        # Encode EHR sequence
        # ------------------------------------------------------------
        for record in records:

            text = " ".join(
                [
                    record.get("text", ""),
                    " ".join(record.get("labs", [])),
                    " ".join(record.get("events", [])),
                ]
            )

            ids, mask = encode_sequence(
                text,
                sequence_vocab,
            )

            sequences.append(ids)
            masks.append(mask)

        input_ids = torch.stack(sequences).to(device)
        attention_mask = torch.stack(masks).to(device)

        # ------------------------------------------------------------
        # Encode CCS knowledge
        # ------------------------------------------------------------
        knowledge_ids = encode_css(
            batch["css_inputs"],
            label_vocab,
        ).to(device)

        knowledge_mask = (knowledge_ids != 0)

        # ------------------------------------------------------------
        # Encode targets
        # ------------------------------------------------------------
        targets = encode_labels(
            [
                record.get("target", [])
                for record in records
            ],
            label_vocab,
        ).to(device)

        # ------------------------------------------------------------
        # Forward
        # ------------------------------------------------------------
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            knowledge_ids=knowledge_ids,
            knowledge_mask=knowledge_mask,
        )

        logits = output["logits"]

        predictions = (
            torch.sigmoid(logits) > 0.5
        ).cpu().numpy()

        all_pred.append(predictions)
        all_true.append(targets.cpu().numpy())

    # ------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------
    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    metrics = {
        "micro_precision": precision_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "micro_recall": recall_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "micro_f1": f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }

    return metrics


@torch.no_grad()
def evaluate_knowgen(model, dataloader, device, max_length=128):
    """Generate diagnosis text and score the predicted code sets."""

    model.eval()

    tp = fp = fn = 0
    examples = []

    for batch in dataloader:
        generated = model.generate(
            batch["modalities"],
            max_length=max_length,
        )

        for text, record in zip(generated, batch["records"]):
            pred = {t.strip() for t in text.split(";") if t.strip()}
            true = {str(t) for t in record.get("target", [])}

            tp += len(pred & true)
            fp += len(pred - true)
            fn += len(true - pred)

            if len(examples) < 3:
                examples.append((sorted(true), sorted(pred)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    for true, pred in examples:
        print(f"target={true}\npred  ={pred}\n")

    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
    }
