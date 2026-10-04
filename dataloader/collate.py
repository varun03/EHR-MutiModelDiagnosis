import torch


def build_text(record):
    parts = []

    if record.get("text"):
        parts.append(
            "clinical_text: " + record["text"]
        )

    if record.get("labs"):
        parts.append(
            "labs: " + " ; ".join(record["labs"])
        )

    if record.get("events"):
        parts.append(
            "events: " + " ; ".join(record["events"])
        )

    return " | ".join(parts)


def collate_fn(batch):
    texts = [build_text(x) for x in batch]

    labels = [
        x.get("target", [])
        for x in batch
    ]

    label_texts = [
        " ; ".join(sorted(str(v) for v in x))
        for x in labels
    ]

    css_inputs = [
        x.get("diagnosis_ccs", [])
        for x in batch
    ]

    modalities = {
        "notes": [x.get("text", "") or "" for x in batch],
        "events": [" ; ".join(x.get("events", [])) for x in batch],
        "labs": [" ; ".join(x.get("labs", [])) for x in batch],
        "prescriptions": [
            " ; ".join(x.get("prescriptions", [])) for x in batch
        ],
    }

    return {
        "modalities": modalities,
        "subject_id": torch.tensor(
            [x["subject_id"] for x in batch],
            dtype=torch.long
        ),
        "hadm_id": torch.tensor(
            [x["hadm_id"] for x in batch],
            dtype=torch.long
        ),
        "texts": texts,
        "label_texts": label_texts,
        "css_inputs": css_inputs,
        "records": batch,
    }
