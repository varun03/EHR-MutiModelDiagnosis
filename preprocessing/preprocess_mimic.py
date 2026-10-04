import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from preprocessing.icd_ccs_mapper import ICDCCSMapper


def read_csv(raw_dir, filename, required=True):
    path = Path(raw_dir) / filename

    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Missing required file: {path}\n"
                f"Put the MIMIC-III CSV files in {raw_dir}"
            )

        return None

    print(f"Loading {path}")

    df = pd.read_csv(
        path,
        low_memory=False
    )

    # Normalize column names
    df.columns = [
        str(col).strip().upper()
        for col in df.columns
    ]

    print(
        f"  Columns: {list(df.columns)}"
    )

    return df


def clean_code(x):
    if pd.isna(x):
        return ""

    return str(x).strip()


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main(args):

    raw = Path(args.raw_dir)
    out = Path(args.processed_dir)
    mapping_dir = Path(args.mapping_dir)

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------
    # 1. Load MIMIC files
    # -------------------------------------------------

    admissions = read_csv(
        raw,
        "ADMISSIONS.csv"
    )

    patients = read_csv(
        raw,
        "PATIENTS.csv"
    )

    diagnoses = read_csv(
        raw,
        "DIAGNOSES_ICD.csv"
    )

    procedures = read_csv(
        raw,
        "PROCEDURES_ICD.csv"
    )

    labs = read_csv(
        raw,
        "LABEVENTS.csv"
    )

    notes = read_csv(
        raw,
        "NOTEEVENTS.csv",
        required=False
    )

    prescriptions = read_csv(
        raw,
        "PRESCRIPTIONS.csv",
        required=False
    )

    # -------------------------------------------------
    # 2. Validate required columns
    # -------------------------------------------------

    required_columns = {

        "ADMISSIONS": [
            "SUBJECT_ID",
            "HADM_ID"
        ],

        "PATIENTS": [
            "SUBJECT_ID"
        ],

        "DIAGNOSES_ICD": [
            "SUBJECT_ID",
            "HADM_ID",
            "ICD9_CODE"
        ],

        "PROCEDURES_ICD": [
            "SUBJECT_ID",
            "HADM_ID",
            "ICD9_CODE"
        ],

        "LABEVENTS": [
            "SUBJECT_ID",
            "HADM_ID",
            "ITEMID"
        ]
    }

    loaded_data = {
        "ADMISSIONS": admissions,
        "PATIENTS": patients,
        "DIAGNOSES_ICD": diagnoses,
        "PROCEDURES_ICD": procedures,
        "LABEVENTS": labs
    }

    for name, columns in required_columns.items():

        df = loaded_data[name]

        missing = [
            col
            for col in columns
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f"\n{name}.csv is missing columns: "
                f"{missing}\n"
                f"Available columns: "
                f"{list(df.columns)}"
            )

    # -------------------------------------------------
    # 3. Select ADMISSIONS columns
    # -------------------------------------------------

    admissions_cols = [
        c
        for c in [
            "SUBJECT_ID",
            "HADM_ID",
            "ADMITTIME",
            "DISCHTIME",
            "ADMISSION_TYPE",
            "ADMISSION_LOCATION",
            "DISCHARGE_LOCATION"
        ]
        if c in admissions.columns
    ]

    admissions = admissions[
        admissions_cols
    ].copy()

    # -------------------------------------------------
    # 4. Select PATIENT columns
    # -------------------------------------------------

    patients_cols = [
        c
        for c in [
            "SUBJECT_ID",
            "GENDER",
            "DOB",
            "DOD",
            "EXPIRE_FLAG"
        ]
        if c in patients.columns
    ]

    patients = patients[
        patients_cols
    ].copy()

    # -------------------------------------------------
    # 5. Process diagnoses
    # -------------------------------------------------

    diagnosis_cols = [
        c
        for c in [
            "SUBJECT_ID",
            "HADM_ID",
            "ICD9_CODE",
            "SEQ_NUM"
        ]
        if c in diagnoses.columns
    ]

    diagnoses = diagnoses[
        diagnosis_cols
    ].copy()

    diagnoses["ICD9_CODE"] = (
        diagnoses["ICD9_CODE"]
        .map(clean_code)
    )

    diagnoses = diagnoses[
        diagnoses["ICD9_CODE"] != ""
    ]

    print(
        f"Diagnosis rows: {len(diagnoses)}"
    )

    # -------------------------------------------------
    # 6. Process procedures
    # -------------------------------------------------

    procedure_cols = [
        c
        for c in [
            "SUBJECT_ID",
            "HADM_ID",
            "ICD9_CODE",
            "SEQ_NUM"
        ]
        if c in procedures.columns
    ]

    procedures = procedures[
        procedure_cols
    ].copy()

    procedures["ICD9_CODE"] = (
        procedures["ICD9_CODE"]
        .map(clean_code)
    )

    procedures = procedures[
        procedures["ICD9_CODE"] != ""
    ]

    print(
        f"Procedure rows: {len(procedures)}"
    )

    # -------------------------------------------------
    # 7. Process laboratory data
    # -------------------------------------------------

    lab_cols = [
        c
        for c in [
            "SUBJECT_ID",
            "HADM_ID",
            "ITEMID",
            "CHARTTIME",
            "VALUE",
            "VALUENUM",
            "VALUEUOM",
            "FLAG"
        ]
        if c in labs.columns
    ]

    labs = labs[
        lab_cols
    ].copy()

    print(
        f"Lab rows: {len(labs)}"
    )

    # -------------------------------------------------
    # 8. Process clinical notes
    # -------------------------------------------------

    if notes is not None:

        note_cols = [
            c
            for c in [
                "SUBJECT_ID",
                "HADM_ID",
                "CHARTDATE",
                "CHARTTIME",
                "CATEGORY",
                "DESCRIPTION",
                "TEXT"
            ]
            if c in notes.columns
        ]

        notes = notes[
            note_cols
        ].copy()

        print(
            f"Note rows: {len(notes)}"
        )

    else:

        print(
            "NOTEEVENTS.csv not available"
        )

    # -------------------------------------------------
    # 9. Load ICD -> CCS mappings
    # -------------------------------------------------

    # The original repository's icd_ccs_*.json files are keyed by CCS
    # category name -> [icd_codes]; the icd_code -> category direction that
    # ICDCCSMapper needs lives in ccs_icd_*.json instead.
    diag_mapper = ICDCCSMapper(
        mapping_dir / "ccs_icd_diag.json"
    )

    proc_mapper = ICDCCSMapper(
        mapping_dir / "ccs_icd_proce.json"
    )

    # -------------------------------------------------
    # 10. Map diagnosis ICD -> CCS
    # -------------------------------------------------

    diagnoses["CCS_CODE"] = (
        diagnoses["ICD9_CODE"]
        .apply(
            lambda x:
            diag_mapper.map_code(x)
        )
    )

    # -------------------------------------------------
    # 11. Map procedure ICD -> CCS
    # -------------------------------------------------

    procedures["CCS_CODE"] = (
        procedures["ICD9_CODE"]
        .apply(
            lambda x:
            proc_mapper.map_code(x)
        )
    )

    # -------------------------------------------------
    # 12. Group diagnoses by admission
    # -------------------------------------------------

    diag_group = (

        diagnoses

        .groupby(
            [
                "SUBJECT_ID",
                "HADM_ID"
            ]
        )

        .agg(

            diagnosis_icd=(
                "ICD9_CODE",
                list
            ),

            diagnosis_ccs=(
                "CCS_CODE",
                list
            )

        )

        .reset_index()

    )

    def flatten_ccs(values):

        result = []

        for x in values:

            if isinstance(x, list):

                result.extend(
                    str(v)
                    for v in x
                    if str(v)
                )

            else:

                if str(x):
                    result.append(
                        str(x)
                    )

        return list(
            dict.fromkeys(result)
        )

    diag_group[
        "diagnosis_ccs"
    ] = (

        diag_group[
            "diagnosis_ccs"
        ]

        .apply(
            flatten_ccs
        )

    )

    # -------------------------------------------------
    # 13. Group procedures
    # -------------------------------------------------

    proc_group = (

        procedures

        .groupby(
            [
                "SUBJECT_ID",
                "HADM_ID"
            ]
        )

        .agg(

            procedure_icd=(
                "ICD9_CODE",
                list
            ),

            procedure_ccs=(
                "CCS_CODE",
                list
            )

        )

        .reset_index()

    )

    proc_group[
        "procedure_ccs"
    ] = (

        proc_group[
            "procedure_ccs"
        ]

        .apply(
            flatten_ccs
        )

    )

    # -------------------------------------------------
    # 14. Group labs
    # -------------------------------------------------

    if "HADM_ID" in labs.columns:

        lab_group = (

            labs

            .dropna(
                subset=[
                    "HADM_ID"
                ]
            )

            .groupby(
                [
                    "SUBJECT_ID",
                    "HADM_ID"
                ]
            )

            .apply(
                lambda g:
                g.to_dict(
                    "records"
                )
            )

            .rename(
                "labs"
            )

            .reset_index()

        )

    else:

        lab_group = pd.DataFrame(
            columns=[
                "SUBJECT_ID",
                "HADM_ID",
                "labs"
            ]
        )

    # -------------------------------------------------
    # 15. Group clinical notes
    # -------------------------------------------------

    if (

        notes is not None

        and "HADM_ID"
        in notes.columns

        and "TEXT"
        in notes.columns

    ):

        note_group = (

            notes

            .dropna(
                subset=[
                    "HADM_ID"
                ]
            )

            .groupby(
                [
                    "SUBJECT_ID",
                    "HADM_ID"
                ]
            )["TEXT"]

            .apply(
                lambda x:

                "\n".join(

                    str(v)

                    for v
                    in x.dropna()
                    .tolist()

                )
            )

            .rename(
                "text"
            )

            .reset_index()

        )

    else:

        note_group = pd.DataFrame(

            columns=[
                "SUBJECT_ID",
                "HADM_ID",
                "text"
            ]

        )

    # -------------------------------------------------
    # 15b. Group prescriptions (treatment modality)
    # -------------------------------------------------

    if (
        prescriptions is not None
        and "HADM_ID" in prescriptions.columns
        and "DRUG" in prescriptions.columns
    ):

        prescriptions = prescriptions.dropna(
            subset=["HADM_ID", "DRUG"]
        ).copy()

        prescriptions["HADM_ID"] = prescriptions["HADM_ID"].astype(int)

        def drug_to_text(row):
            # Drug name only: dose/route lengthen the text without adding
            # much diagnostic signal.
            return str(row["DRUG"]).strip()

        prescriptions["drug_text"] = prescriptions.apply(
            drug_to_text, axis=1
        )

        presc_group = (
            prescriptions
            .groupby(["SUBJECT_ID", "HADM_ID"])["drug_text"]
            .apply(lambda x: list(dict.fromkeys(x)))
            .rename("prescriptions")
            .reset_index()
        )

    else:

        print("PRESCRIPTIONS.csv not found - prescriptions modality empty.")

        presc_group = pd.DataFrame(
            columns=["SUBJECT_ID", "HADM_ID", "prescriptions"]
        )

    # -------------------------------------------------
    # 16. Merge all modalities
    # -------------------------------------------------

    merged = (

        admissions

        .merge(
            patients,
            on="SUBJECT_ID",
            how="left"
        )

        .merge(
            diag_group,
            on=[
                "SUBJECT_ID",
                "HADM_ID"
            ],
            how="left"
        )

        .merge(
            proc_group,
            on=[
                "SUBJECT_ID",
                "HADM_ID"
            ],
            how="left"
        )

        .merge(
            lab_group,
            on=[
                "SUBJECT_ID",
                "HADM_ID"
            ],
            how="left"
        )

        .merge(
            presc_group,
            on=[
                "SUBJECT_ID",
                "HADM_ID"
            ],
            how="left"
        )

        .merge(
            note_group,
            on=[
                "SUBJECT_ID",
                "HADM_ID"
            ],
            how="left"
        )

    )

    # -------------------------------------------------
    # 17. Fill missing modalities
    # -------------------------------------------------

    for col in [

        "diagnosis_icd",
        "diagnosis_ccs",
        "procedure_icd",
        "procedure_ccs",
        "labs",
        "prescriptions"

    ]:

        merged[col] = (

            merged[col]

            .apply(

                lambda x:

                x

                if isinstance(
                    x,
                    list
                )

                else []

            )

        )

    merged["text"] = (
        merged["text"]
        .fillna("")
    )

    # -------------------------------------------------
    # 18. Keep admissions with diagnosis
    # -------------------------------------------------

    merged = (

        merged[

            merged[
                "diagnosis_icd"
            ]

            .apply(
                len
            )

            > 0

        ]

        .reset_index(
            drop=True
        )

    )

    # -------------------------------------------------
    # 19. Save processed files
    # -------------------------------------------------

    save_pickle(

        merged,

        out /
        "admissions.pkl"

    )

    save_pickle(

        diag_group,

        out /
        "diagnoses.pkl"

    )

    save_pickle(

        proc_group,

        out /
        "procedures.pkl"

    )

    save_pickle(

        lab_group,

        out /
        "labs.pkl"

    )

    save_pickle(

        note_group,

        out /
        "notes.pkl"

    )

    # -------------------------------------------------
    # 20. Save summary
    # -------------------------------------------------

    summary = {

        "num_admissions":
            int(
                len(
                    merged
                )
            ),

        "num_diagnosis_rows":
            int(
                len(
                    diagnoses
                )
            ),

        "num_procedure_rows":
            int(
                len(
                    procedures
                )
            ),

        "num_lab_rows":
            int(
                len(
                    labs
                )
            ),

        "notes_available":
            notes is not None,

        "num_notes_rows":
            int(
                len(
                    notes
                )
            )

            if notes is not None

            else 0

    }

    with open(

        out /
        "preprocessing_summary.json",

        "w"

    ) as f:

        json.dump(

            summary,

            f,

            indent=2

        )

    print(
        "\n================================"
    )

    print(
        "Preprocessing complete"
    )

    print(
        "================================"
    )

    print(
        json.dumps(
            summary,
            indent=2
        )
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        default="data/mimiciii/raw"
    )

    parser.add_argument(
        "--processed-dir",
        default="data/mimiciii/processed"
    )

    parser.add_argument(
        "--mapping-dir",
        default="data/mappings"
    )

    main(
        parser.parse_args()
    )
