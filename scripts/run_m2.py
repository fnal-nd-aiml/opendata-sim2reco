#!/usr/bin/env python
"""Train and evaluate the M2 surrogate.
Usage: scripts/run_m2.py OUT_DIR --stems data/slim/A data/slim/B ... [--epochs 20] [--bs 1024] [--max-train N] [--eval-only]"""
import sys, pathlib, argparse, json, glob
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("--stems", nargs="*", default=None); ap.add_argument("--slim-dir", default="data/slim_1A")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--bs", type=int, default=1024); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-train", type=int, default=None); ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--m1", default="reports/m1_1A/metrics.json"); ap.add_argument("--exclude-inttype", type=int, nargs="*", default=None, help="GENIE intType codes held out of train/val (8 = 2p2h)"); ap.add_argument("--only-inttype", type=int, nargs="*", default=None, help="evaluate only test events of these intType codes")
    ap.add_argument("--d-model", type=int, default=128); ap.add_argument("--n-layers", type=int, default=4); ap.add_argument("--n-files", type=int, default=None)
    ap.add_argument("--flow-hidden", type=int, default=768); ap.add_argument("--flow-layers", type=int, default=5); ap.add_argument("--steps", type=int, default=100)
    a = ap.parse_args()
    from sim2reco.train.m2 import train, evaluate, load_model, make_loaders, write_data_table
    from sim2reco.data.compact import load_compact, Tier1Transform
    from sim2reco.data.dataset import split_by_subrun
    stems = a.stems or sorted(p[:-len(".truth.parquet")] for p in glob.glob(f"{a.slim_dir}/*.truth.parquet"))
    if a.n_files: stems = stems[:a.n_files]
    def restrict(d, idx, ld, tf, ptf=None):
        if not a.only_inttype: return idx, ld
        from torch.utils.data import DataLoader
        from sim2reco.data.compact import CompactDataset, collate
        keep = idx["test"][np.isin(d["intType"][idx["test"]], a.only_inttype)]
        idx = dict(idx, test=keep); ld = dict(ld, test=DataLoader(CompactDataset(d, keep, tf, 0, prong_tf=ptf), batch_size=a.bs, collate_fn=collate, num_workers=4))
        print(f"evaluating on {len(keep):,} test events of intType {a.only_inttype}"); return idx, ld
    print(f"{len(stems)} files")
    if a.eval_only:
        d = load_compact(stems); split = split_by_subrun(d["subrun"], seed=0)
        model, tf = load_model(pathlib.Path(a.out_dir) / "model.pt")
        idx, ds, ld = make_loaders(d, split, tf, a.bs, 0)
    else:
        d, split, tf, idx, ld, out = train(stems, a.out_dir, a.epochs, a.bs, a.lr, 0, "cuda", a.d_model, a.n_layers, a.flow_hidden, a.flow_layers, max_train_events=a.max_train, exclude_inttype=a.exclude_inttype)
        model, tf = load_model(pathlib.Path(a.out_dir) / "model.pt")
    idx, ld = restrict(d, idx, ld, tf)
    m1 = json.load(open(a.m1)) if pathlib.Path(a.m1).exists() else None
    write_data_table(stems, d, split, a.out_dir, None if a.eval_only else a.epochs, sum(p.numel() for p in model.parameters()))
    M = evaluate(model, tf, d, idx["test"], ld["test"], a.out_dir, n_steps=a.steps, m1_metrics=m1)
    print(json.dumps({"tier0": {k: round(v["logloss"], 4) for k, v in M["tier0"].items()}, "mult_logloss": round(M["multiplicity"]["logloss"], 4),
                      "auc_marginal": round(M["classifier_auc_marginal"], 4), "auc_conditional": round(M["classifier_auc_conditional"], 4),
                      "w1": {k: round(v["w1"], 4) for k, v in M["tier1_model_space"].items()}}, indent=1))
