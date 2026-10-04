import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.preprocess_mimic import main
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/mimiciii/raw")
    parser.add_argument(
        "--processed-dir",
        default="data/mimiciii/processed"
    )
    parser.add_argument(
        "--mapping-dir",
        default="data/mappings"
    )
    args = parser.parse_args()
    main(args)
