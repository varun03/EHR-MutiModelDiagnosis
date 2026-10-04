import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class PatientDataset(Dataset):
    """Admission-level EHR dataset for the EHR-KnowGen reproduction.

    Returns raw multimodal fields. Tokenization and padding are handled by
    the collate function so variable-length records are supported.
    """

    def __init__(self, data_path, max_samples=0):
        data_path = Path(data_path)
        with open(data_path, "rb") as f:
            self.records = pickle.load(f)

        if max_samples > 0:
            self.records = self.records[:max_samples]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


class RecordListDataset(Dataset):
    """Wraps an in-memory list of records, e.g. a train/validation split."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def make_dataloader(
    data_path,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    max_samples=0,
    collate_fn=None,
):
    dataset = PatientDataset(
        data_path,
        max_samples=max_samples
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
