"""Train the KnowGen model and check that it really learns.

Uses k-fold cross-validation (every admission is held out once), because a
single 20% split of the MIMIC demo is only ~26 patients and far too noisy.

Per fold, on the held-out admissions only, it reports:
  * a frequency baseline (always predict the most common diagnoses)
  * disease detection from the knowledge-calibration head
  * the generated diagnosis text

The model only counts as "learning" if it beats the baseline on average.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from dataloader.collate import collate_fn
from dataloader.dataloader_ehrknowgen import PatientDataset, RecordListDataset
from engine.trainer import set_seed, train_knowgen
from models.model_factory import create_model
from preprocessing.knowledge_graph import build_knowledge_graph, label_prior


def set_scores(pred_sets, true_sets):
    tp = sum(len(p & t) for p, t in zip(pred_sets, true_sets))
    fp = sum(len(p - t) for p, t in zip(pred_sets, true_sets))
    fn = sum(len(t - p) for p, t in zip(pred_sets, true_sets))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    jaccard = sum(
        len(p & t) / max(1, len(p | t))
        for p, t in zip(pred_sets, true_sets)
    ) / max(1, len(true_sets))
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "jaccard": round(jaccard, 4),
    }


@torch.no_grad()
def detect(model, loader, top_k):
    """Top-k diseases per patient from the calibration head."""

    model.eval()
    topk, true = [], []
    for batch in loader:
        probs, _ = model.predict_codes(batch["modalities"])
        for row in probs:
            idx = row.topk(top_k).indices.tolist()
            topk.append({model.code_vocab[i] for i in idx})
        true.extend({str(t) for t in r["target"]} for r in batch["records"])
    return topk, true


@torch.no_grad()
def generate(model, loader):
    model.eval()
    pred, true = [], []
    for batch in loader:
        for text, r in zip(
            model.generate(batch["modalities"]), batch["records"]
        ):
            pred.append({t.strip() for t in text.split(";") if t.strip()})
            true.append({str(t) for t in r["target"]})
    return pred, true


def make_folds(records, folds, seed):
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    chunks = [idx[i::folds] for i in range(folds)]
    for k in range(folds):
        val = set(chunks[k])
        yield (
            [records[i] for i in idx if i not in val],
            [records[i] for i in chunks[k]],
        )


def run_fold(fold, train_records, val_records, args, device):
    def loader(recs, shuffle=False):
        return DataLoader(
            RecordListDataset(recs),
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

    train_loader = loader(train_records, shuffle=True)
    train_eval = loader(train_records)
    val_loader = loader(val_records)

    freq = Counter(str(t) for r in train_records for t in r["target"])
    top_k = max(1, round(
        sum(len(r["target"]) for r in train_records) / len(train_records)
    ))
    baseline_set = {c for c, _ in freq.most_common(top_k)}
    val_true = [{str(t) for t in r["target"]} for r in val_records]
    baseline = set_scores([baseline_set] * len(val_true), val_true)

    print(
        f"\n===== fold {fold}: train={len(train_records)} "
        f"val={len(val_records)} | baseline F1={baseline['f1']} ====="
    )

    graph = build_knowledge_graph(
        args.mapping_dir, args.raw_dir, train_records
    )
    model = create_model(
        model_type="knowgen",
        graph=graph,
        class_prior=label_prior(train_records),
        text_model_name=args.text_model_name,
    ).to(device)

    history = []

    def on_epoch(epoch, loss, loss_f, loss_c):
        entry = {
            "epoch": epoch,
            "L_f": round(loss_f, 4),
            "L_c": round(loss_c, 4),
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            tr_pred, tr_true = detect(model, train_eval, top_k)
            va_pred, va_true = detect(model, val_loader, top_k)
            entry["train_detect_f1"] = set_scores(tr_pred, tr_true)["f1"]
            entry["val_detect_f1"] = set_scores(va_pred, va_true)["f1"]
            print(
                f"   detection F1 (top-{top_k}) epoch {epoch}: "
                f"train={entry['train_detect_f1']}  "
                f"val={entry['val_detect_f1']}  "
                f"(baseline={baseline['f1']})",
                flush=True,
            )
        history.append(entry)

    train_knowgen(
        model,
        train_loader,
        device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        epoch_callback=on_epoch,
    )

    topk, true = detect(model, val_loader, top_k)
    result = {
        "fold": fold,
        "baseline": baseline,
        "detect": set_scores(topk, true),
        "history": history,
    }

    if not args.skip_generation:
        gen_pred, gen_true = generate(model, val_loader)
        result["generation"] = set_scores(gen_pred, gen_true)
        result["generation_examples"] = [
            {"true": sorted(t)[:6], "generated": sorted(p)[:6]}
            for p, t in list(zip(gen_pred, gen_true))[:2]
        ]

    print(f"   fold {fold} result: baseline={baseline['f1']} "
          f"detect={result['detect']['f1']} "
          f"generation={result.get('generation', {}).get('f1', '-')}")
    return result, model, graph


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
    return mean, var ** 0.5


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = PatientDataset(
        Path(args.processed_dir) / "ehrknowgen_dataset.pkl",
        max_samples=args.max_samples,
    ).records
    print(f"{len(records)} admissions, {args.folds}-fold cross-validation")

    results = []
    last = None
    for fold, (tr, va) in enumerate(make_folds(records, args.folds, args.seed)):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        set_seed(args.seed + fold)
        res, model, graph = run_fold(fold, tr, va, args, device)
        results.append(res)
        last = (model, graph, va)

    summary = {}
    for key in ("baseline", "detect", "generation"):
        if all(key in r for r in results):
            f1s = [r[key]["f1"] for r in results]
            m, s = mean_std(f1s)
            summary[key] = {"mean_f1": round(m, 4), "std": round(s, 4),
                            "per_fold": f1s}

    gain = [
        d - b for d, b in zip(
            (r["detect"]["f1"] for r in results),
            (r["baseline"]["f1"] for r in results),
        )
    ]
    gm, gs = mean_std(gain)
    summary["detect_gain_over_baseline"] = {
        "mean": round(gm, 4), "std": round(gs, 4),
        "folds_beating_baseline": sum(g > 0 for g in gain),
        "folds": len(gain),
    }

    print("\n=========== CROSS-VALIDATED HELD-OUT RESULTS ===========")
    for k, v in summary.items():
        print(f"{k:28s} {v}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "learning_report.json").write_text(
        json.dumps({"summary": summary, "folds": results}, indent=2)
    )

    model, graph, va = last
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "text_model_name": args.text_model_name,
            "graph": graph,
            "val_hadm_ids": [r["hadm_id"] for r in va],
        },
        out / "ehrknowgen_demo_knowgen.pt",
    )
    print(f"Saved report + checkpoint to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", default="data/mimiciii/processed")
    p.add_argument("--mapping-dir", default="data/mappings")
    p.add_argument("--raw-dir", default="data/mimiciii/raw")
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--text-model-name", default="t5-small")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--only-fold", type=int, default=None)
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
