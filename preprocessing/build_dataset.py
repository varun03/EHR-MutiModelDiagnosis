import argparse
import pickle
from pathlib import Path

import pandas as pd


def flatten(values):
    result = []
    for x in values or []:
        if isinstance(x, list):
            result.extend(str(v) for v in x if str(v))
        elif str(x):
            result.append(str(x))
    return list(dict.fromkeys(result))


def load_lab_names(raw_dir):
    path = Path(raw_dir) / "D_LABITEMS.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.lower() for c in df.columns]
    return dict(zip(df["itemid"], df["label"]))


def lab_to_text(labs, lab_names=None, max_labs=12):
    """One short entry per lab test: name, latest value, abnormal marker.

    Abnormal tests (most frequently abnormal first) come before normal
    ones, so the few entries that fit in the model's input window are the
    clinically informative ones.
    """

    lab_names = lab_names or {}
    latest = {}
    abnormal = {}

    for row in labs:
        if not isinstance(row, dict) or pd.isna(row.get("VALUE")):
            continue
        item = str(int(row["ITEMID"])) if pd.notna(row.get("ITEMID")) else ""
        if not item:
            continue
        time = str(row.get("CHARTTIME", ""))
        if item not in latest or time >= latest[item][0]:
            latest[item] = (time, row)
        if str(row.get("FLAG")) == "abnormal":
            abnormal[item] = abnormal.get(item, 0) + 1

    ranked = sorted(
        latest,
        key=lambda i: (-abnormal.get(i, 0), i),
    )

    output = []
    for item in ranked[:max_labs]:
        row = latest[item][1]
        name = lab_names.get(item, f"lab {item}")
        unit = row.get("VALUEUOM", "")
        unit = "" if pd.isna(unit) else str(unit)
        flag = " abnormal" if item in abnormal else ""
        output.append(f"{name} {row['VALUE']} {unit}{flag}".strip())
    return output


def build_record(
    row,
    max_labs=12,
    max_text_chars=2000,
    max_drugs=24,
    drop_procedures=False,
    lab_names=None,
):
    diagnosis_icd = flatten(row.get("diagnosis_icd", []))
    diagnosis_ccs = flatten(row.get("diagnosis_ccs", []))
    procedure_icd = flatten(row.get("procedure_icd", []))
    procedure_ccs = flatten(row.get("procedure_ccs", []))

    labs = lab_to_text(
        row.get("labs", []),
        lab_names=lab_names,
        max_labs=max_labs
    )

    # Drug names only (no dose/route): short, deduplicated, case-folded.
    prescriptions = list(dict.fromkeys(
        str(x).lower() for x in flatten(row.get("prescriptions", []))
    ))[:max_drugs]

    text = str(row.get("text", "") or "")
    text = text[:max_text_chars]

    # For this first reproduction, the target is the admission's CCS
    # diagnosis set. A later version can use target_disease.json /
    # label_ccs_target.json from the original repository.
    target = diagnosis_ccs

    # Diagnosis codes are deliberately NOT part of the events: they are the
    # label, so including them would leak the answer into the input.
    #
    # Procedure codes are a treatment signal that is partly a consequence of
    # the diagnosis (and is coded at discharge together with it). Pass
    # drop_procedures=True to remove them for a cleaner diagnosis task.
    if drop_procedures:
        events = []
    else:
        # Category names only: the raw ICD procedure numbers are opaque to
        # a text model and just spend input tokens.
        events = [str(x) for x in procedure_ccs]

    return {
        "subject_id": int(row["SUBJECT_ID"]),
        "hadm_id": int(row["HADM_ID"]),
        "text": text,
        "labs": labs,
        "prescriptions": prescriptions,
        "events": events,
        "diagnosis_icd": diagnosis_icd,
        "diagnosis_ccs": diagnosis_ccs,
        "procedure_icd": procedure_icd,
        "procedure_ccs": procedure_ccs,
        "target": target,
    }


def main(args):
    input_path = Path(args.processed_dir) / "admissions.pkl"
    output_path = Path(args.processed_dir) / "ehrknowgen_dataset.pkl"

    with open(input_path, "rb") as f:
        df = pickle.load(f)

    if args.max_samples > 0:
        df = df.iloc[:args.max_samples].copy()

    lab_names = load_lab_names(args.raw_dir)

    records = [
        build_record(
            row,
            max_labs=args.max_labs,
            max_text_chars=args.max_text_chars,
            max_drugs=args.max_drugs,
            drop_procedures=args.drop_procedures,
            lab_names=lab_names,
        )
        for _, row in df.iterrows()
    ]

    with open(output_path, "wb") as f:
        pickle.dump(records, f)

    print(f"Saved {len(records)} records to {output_path}")
    if args.drop_procedures:
        print("Procedure codes were dropped from the model input.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir", default="data/mimiciii/processed"
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--raw-dir", default="data/mimiciii/raw")
    parser.add_argument("--max-labs", type=int, default=12)
    parser.add_argument("--max-text-chars", type=int, default=2000)
    parser.add_argument("--max-drugs", type=int, default=24)
    parser.add_argument("--drop-procedures", action="store_true")
    main(parser.parse_args())
