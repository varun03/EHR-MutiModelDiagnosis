# EHR-KnowGen Demo Reproduction

A CPU-friendly reproduction pipeline inspired by the uploaded EHR-KnowGen repository.

## Important
This project is designed to work first with the MIMIC-III Demo. The demo does not provide the full clinical-note corpus required for the original multimodal text branch, so the text modality is optional. The pipeline therefore supports:

- Diagnosis ICD-9 codes
- Procedure ICD-9 codes
- ICD-9 -> CCS mappings
- Laboratory events
- Prescriptions (treatment modality)
- Optional clinical notes
- Target labels derived from CCS diagnosis categories
- A lightweight multimodal PyTorch model with modality prompts
- Optional Hugging Face T5-small generation model
- `knowgen`: a T5-based model that follows the EHR-KnowGen architecture (see below)

The default demo training uses a lightweight model so it can run on CPU without requiring a large pretrained model.

## Directory

data/mimiciii/raw/        Put MIMIC-III Demo CSV files here
data/mimiciii/processed/  Generated pickle/JSON files
data/mappings/            ICD -> CCS mapping files
preprocessing/             Raw data processing, knowledge graph
dataloader/                PyTorch Dataset/DataLoader
models/                    Model definitions
engine/                    Training/evaluation
scripts/                   Command-line entry points

## Expected raw files

At minimum:
- ADMISSIONS.csv
- PATIENTS.csv
- DIAGNOSES_ICD.csv
- PROCEDURES_ICD.csv
- LABEVENTS.csv
- D_ICD_DIAGNOSES.csv
- D_ICD_PROCEDURES.csv
- D_LABITEMS.csv

Optional:
- NOTEEVENTS.csv
- PRESCRIPTIONS.csv (treatment modality; empty if missing)
- D_ITEMS.csv

## Mapping files

Copy these from the original EHR-KnowGen repository if available:
- icd_ccs_diag.json
- ccs_icd_diag.json
- icd_ccs_proce.json
- ccs_icd_proce.json

The preprocessing script also has a fallback mode: if mappings are absent, it keeps ICD codes as knowledge tokens. This is useful for testing the demo pipeline, but for a faithful reproduction you should use the original mapping files.

## Installation

Python 3.10+ is recommended.

    pip install -r requirements.txt

## Run

1. Put raw CSV files under data/mimiciii/raw/
2. Put mapping JSON files under data/mappings/ if available.
3. Run:

    python scripts/01_preprocess.py

4. Build the model-ready dataset:

    python scripts/02_build_dataset.py

5. Run a CPU smoke test:

    python scripts/03_train.py --epochs 1 --batch-size 1 --max-samples 20

6. Evaluate:

    python scripts/04_evaluate.py

`03_train.py` holds out a deterministic validation split (`--val-ratio`, default 0.2) and saves the held-out admission IDs into the checkpoint. `04_evaluate.py` uses those IDs to evaluate only on unseen admissions by default. Pass `--full` to instead evaluate on every loaded record, including ones used for training (this is also the fallback for older checkpoints saved before this split existed).

## Optional T5 mode

The project includes an optional T5-small generation model. It is disabled by default because it is much heavier on CPU.

    python scripts/03_train.py --model t5 --epochs 1 --batch-size 1 --max-samples 10

This requires the transformers package and may download T5-small from Hugging Face.

## EHR-KnowGen model (`--model knowgen`)

    python scripts/03_train.py --model knowgen --epochs 10 --max-samples 100
    python scripts/04_evaluate.py --checkpoint checkpoints/ehrknowgen_demo_knowgen.pt

Architecture (`models/knowgen_t5.py`):

- Four modalities (notes, events, labs, prescriptions) are each encoded by a shared T5 encoder.
- `SoftPromptFusion`: learnable soft prompts per modality cross-attend to that modality and map it into one unified feature space.
- External knowledge is a global two-level graph (CCS categories + ICD-9 codes, `preprocessing/knowledge_graph.py`), identical for every patient.
- `KnowledgeAttention` / `KnowledgeCalibration` form the disease-related information extractor (auxiliary loss L_c).
- The T5 decoder generates the diagnoses (loss L_f). Total loss = L_f + 0.5 * L_c.
- `model.explain(batch)` returns the knowledge-graph nodes the model attended to, as evidence.

Use `--text-model-name` to switch the base model (for example `google/flan-t5-base`).

### Treatment signals and label leakage

- Diagnosis codes are **not** part of the model input (they are the label).
- Procedure codes are in the `events` modality by default. They are partly a consequence of the diagnosis and are coded at discharge together with it. For a cleaner diagnosis task, rebuild the dataset without them:

      python scripts/02_build_dataset.py --drop-procedures

- Prescriptions are a treatment modality. Drug orders can still hint at the diagnosis (for example insulin, metoprolol or antibiotics). This is the kind of evidence the method is meant to use, but state it when reporting results.

## Data use

MIMIC-III is credentialed data. Obtain it from PhysioNet and follow its data-use requirements. Do not commit the raw dataset to Git.
