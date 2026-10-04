import json
from collections import Counter
from pathlib import Path

import pandas as pd


def label_prior(records):
    """Fraction of admissions in which each CCS category occurs."""

    counts = Counter(
        str(t) for r in records for t in set(r.get("target", []))
    )
    n = max(1, len(records))
    return {c: v / n for c, v in counts.items()}


def build_knowledge_graph(
    mapping_dir,
    raw_dir,
    train_records,
    max_fine=300,
):
    """Two-level external knowledge graph shared by every patient.

    coarse nodes: CCS disease categories (from icd_ccs_diag.json)
    fine nodes:   ICD-9 diagnosis codes, named by their MIMIC long title,
                  each linked to its parent CCS category.

    The fine nodes are the ICD codes that are most frequent in the training
    split, so the graph is a fixed, global vocabulary and does not depend on
    any individual patient's labels at inference time.
    """

    ccs_to_icd = json.load(
        open(Path(mapping_dir) / "icd_ccs_diag.json")
    )

    titles = pd.read_csv(
        Path(raw_dir) / "D_ICD_DIAGNOSES.csv",
        dtype=str,
    )
    titles.columns = [c.lower() for c in titles.columns]
    icd_title = dict(zip(titles["icd9_code"], titles["long_title"]))

    icd_parent = {}
    for ccs, codes in ccs_to_icd.items():
        for code in codes:
            icd_parent.setdefault(str(code), ccs)

    counts = Counter(
        str(c)
        for r in train_records
        for c in r.get("diagnosis_icd", [])
    )

    fine_codes = [
        c for c, _ in counts.most_common()
        if c in icd_title and c in icd_parent
    ][:max_fine]

    coarse = sorted(ccs_to_icd)
    coarse_idx = {c: i for i, c in enumerate(coarse)}

    nodes = list(coarse) + [icd_title[c] for c in fine_codes]
    levels = [0] * len(coarse) + [1] * len(fine_codes)

    # (child_node, parent_node) edges
    edges = [
        (len(coarse) + i, coarse_idx[icd_parent[c]])
        for i, c in enumerate(fine_codes)
    ]

    return {
        "nodes": nodes,
        "levels": levels,
        "edges": edges,
        "num_coarse": len(coarse),
        "coarse_names": coarse,
    }
