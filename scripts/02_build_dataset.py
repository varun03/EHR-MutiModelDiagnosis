import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.build_dataset import main
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        default="data/mimiciii/processed"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0
    )
    parser.add_argument(
        "--raw-dir",
        default="data/mimiciii/raw"
    )
    parser.add_argument(
        "--max-labs",
        type=int,
        default=12
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=2000
    )
    parser.add_argument(
        "--max-drugs",
        type=int,
        default=24
    )
    parser.add_argument(
        "--drop-procedures",
        action="store_true",
        help="Remove procedure codes from the model input (cleaner "
             "diagnosis task: procedures are partly a consequence of "
             "the diagnosis)."
    )
    args = parser.parse_args()
    main(args)
