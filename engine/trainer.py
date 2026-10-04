import random
import numpy as np
import torch
from sklearn.metrics import f1_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_records(records, val_ratio=0.2, seed=42):
    """Deterministically split records into train/validation lists.

    Uses a local Random instance (seeded independently of set_seed) so the
    split is reproducible across the train and evaluate scripts regardless
    of what other random state has been consumed.
    """

    n = len(records)

    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    val_size = int(round(n * val_ratio))

    if n > 1:
        val_size = min(max(val_size, 1), n - 1)
    else:
        val_size = 0

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_records = [records[i] for i in train_indices]
    val_records = [records[i] for i in val_indices]

    return train_records, val_records


def make_vocab(records):
    labels = sorted({
        str(label)
        for record in records
        for label in record.get("target", [])
        if str(label)
    })

    # 0 is reserved for padding.
    token_to_id = {
        "<PAD>": 0
    }

    for label in labels:
        token_to_id[label] = len(token_to_id)

    return token_to_id


def encode_labels(label_lists, token_to_id):
    batch = len(label_lists)
    vocab_size = len(token_to_id)

    target = torch.zeros(
        batch,
        vocab_size,
        dtype=torch.float32
    )

    for i, labels in enumerate(label_lists):
        for label in labels:
            if label in token_to_id:
                target[
                    i,
                    token_to_id[label]
                ] = 1.0

    return target


def encode_css(css_lists, token_to_id, max_len=32):
    output = torch.zeros(
        len(css_lists),
        max_len,
        dtype=torch.long
    )

    for i, codes in enumerate(css_lists):
        ids = [
            token_to_id.get(str(code), 0)
            for code in codes[:max_len]
        ]
        if ids:
            output[i, :len(ids)] = torch.tensor(
                ids,
                dtype=torch.long
            )

    return output


def build_sequence_vocab(records):
    # Lightweight text tokenizer. For the demo baseline, we use whitespace
    # tokens from labs/events/clinical text.
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for record in records:
        text = " ".join([
            str(record.get("text", "")),
            " ".join(record.get("labs", [])),
            " ".join(record.get("events", [])),
        ])

        for token in text.lower().split():
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab


def encode_sequence(text, vocab, max_len=128):
    ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in text.lower().split()[:max_len]
    ]

    output = torch.zeros(
        max_len,
        dtype=torch.long
    )

    if ids:
        output[:len(ids)] = torch.tensor(
            ids,
            dtype=torch.long
        )

    mask = output != 0

    return output, mask


def train_lightweight(
    model,
    dataloader,
    label_vocab,
    sequence_vocab,
    device,
    epochs=3,
    learning_rate=5e-4,
):
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        all_true = []
        all_pred = []

        for batch in dataloader:
            records = batch["records"]

            sequences = []
            masks = []

            for record in records:
                text = " ".join([
                    record.get("text", ""),
                    " ".join(record.get("labs", [])),
                    " ".join(record.get("events", [])),
                ])

                ids, mask = encode_sequence(
                    text,
                    sequence_vocab
                )

                sequences.append(ids)
                masks.append(mask)

            input_ids = torch.stack(sequences).to(device)
            attention_mask = torch.stack(masks).to(device)

            css_ids = encode_css(
                batch["css_inputs"],
                label_vocab
            ).to(device)

            targets = encode_labels(
                [
                    r.get("target", [])
                    for r in records
                ],
                label_vocab
            ).to(device)

            optimizer.zero_grad()

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                css_ids=css_ids,
            )

            loss = criterion(
                output["logits"],
                targets
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += loss.item()

            pred = (
                torch.sigmoid(
                    output["logits"]
                ) > 0.5
            ).detach().cpu().numpy()

            all_pred.append(pred)
            all_true.append(
                targets.detach().cpu().numpy()
            )

        y_true = np.concatenate(
            all_true,
            axis=0
        )
        y_pred = np.concatenate(
            all_pred,
            axis=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss={total_loss / max(1, len(dataloader)):.4f} | "
            f"Micro-F1={f1:.4f}"
        )


def train_t5(
    model,
    dataloader,
    device,
    epochs=3,
    learning_rate=5e-4,
    max_length=128,
):
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        for batch in dataloader:
            input_texts = batch["texts"]
            target_texts = batch["label_texts"]
            css_texts = [
                " ; ".join(str(c) for c in codes)
                for codes in batch["css_inputs"]
            ]

            optimizer.zero_grad()

            outputs = model(
                input_texts=input_texts,
                target_texts=target_texts,
                css_texts=css_texts,
                max_length=max_length,
            )

            loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss={total_loss / max(1, len(dataloader)):.4f}"
        )



def train_knowgen(
    model,
    dataloader,
    device,
    epochs=3,
    learning_rate=5e-4,
    max_length=192,
    epoch_callback=None,
):
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    for epoch in range(epochs):
        model.train()

        total = total_f = total_c = 0.0

        for batch in dataloader:
            optimizer.zero_grad()

            out = model(
                batch["modalities"],
                target_texts=batch["label_texts"],
                target_lists=[
                    r.get("target", [])
                    for r in batch["records"]
                ],
                max_length=max_length,
            )

            out["loss"].backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total += out["loss"].item()
            total_f += out["loss_f"].item()
            total_c += out["loss_c"].item()

        n = max(1, len(dataloader))
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss={total / n:.4f} | "
            f"L_f={total_f / n:.4f} | "
            f"L_c={total_c / n:.4f}"
        )

        if epoch_callback is not None:
            epoch_callback(epoch + 1, total / n, total_f / n, total_c / n)
