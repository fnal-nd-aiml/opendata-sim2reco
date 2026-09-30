#!/usr/bin/env python
"""Conform a pruned surrogate ntuple to the full branch schema of a reference AnaTuple.

Usage: scripts/conform_ntuple.py SURROGATE.root REFERENCE.root OUT.root [--pot POT] [--strict] [--n-ref 20000]

Every branch of REFERENCE's `MasterAnaDev` tree that the surrogate does not produce is added, filled with that
branch's default as observed in REFERENCE (the most common value over the first --n-ref events; jagged branches
become empty lists with a zero `_sz` counter). Branches the surrogate produces are copied as they are; types are
checked against the reference. A `Meta` tree is written with the entry counts and the given POT (-1 if none).
With --strict, surrogate-only branches (truth passthrough, surrogate_p_reco_exists) are dropped so the branch
list matches the reference exactly.

Caveat: the added branches carry no information. They exist so that analysis code that binds every branch of
the tuple runs unchanged; any physics read from them is the reference file's sentinel, not a prediction.
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import awkward as ak
import numpy as np
import uproot

from sim2reco.io.writer import write_ntuple


def branch_defaults(ref_tree, names, n_ref):
    """Most common value per branch (per element for fixed arrays) over the first n_ref events."""
    out, irregular = {}, []
    for chunk in ref_tree.iterate(names, entry_stop=n_ref, step_size=n_ref, library="ak"):
        for k in names:
            a = chunk[k]
            t = ref_tree[k].typename
            if t.endswith("[]"):
                out[k] = None  # jagged -> empty
                continue
            try:
                v = ak.to_numpy(a)
            except Exception:  # declared fixed-size but irregular in the file: treat as jagged
                out[k] = None; irregular.append(k); continue
            if v.ndim == 1:
                vals, cnt = np.unique(v, return_counts=True); out[k] = vals[cnt.argmax()].item()
            else:
                flat = v.reshape(len(v), -1); mode = [np.unique(flat[:, j], return_counts=True) for j in range(flat.shape[1])]
                out[k] = np.array([m[0][m[1].argmax()] for m in mode]).reshape(v.shape[1:]).tolist()
    if irregular:
        print(f"note: {len(irregular)} branches declared fixed-size are irregular in the reference and are written empty: {irregular[:6]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("surrogate"); ap.add_argument("reference"); ap.add_argument("out")
    ap.add_argument("--pot", type=float, default=-1.0); ap.add_argument("--strict", action="store_true"); ap.add_argument("--n-ref", type=int, default=20000)
    ap.add_argument("--truth-entries", type=float, default=None, help="Total_Truth_Entries for Meta (default: unknown, -1)")
    a = ap.parse_args()
    S = uproot.open(a.surrogate)["MasterAnaDev"]; R = uproot.open(a.reference)["MasterAnaDev"]
    n = S.num_entries
    s_names = [k for k in S.keys() if not k.endswith("_sz")]
    r_names = [k for k in R.keys() if not k.endswith("_sz")]
    shared = [k for k in s_names if k in R.keys()]
    only_s = [k for k in s_names if k not in R.keys()]
    missing = [k for k in r_names if k not in S.keys()]
    bad = [(k, S[k].typename, R[k].typename) for k in shared if S[k].typename != R[k].typename]
    if bad:
        sys.exit(f"type mismatch on shared branches: {bad[:5]}")
    cache = pathlib.Path(a.reference).with_suffix(".defaults.json")
    if cache.exists():
        defaults = json.loads(cache.read_text())
    else:
        print(f"deriving defaults for {len(missing)} branches from the first {a.n_ref} reference events ...", flush=True)
        defaults = branch_defaults(R, missing, a.n_ref); cache.write_text(json.dumps(defaults))
    out = {}
    arrays = S.arrays(s_names if not a.strict else shared, library="ak")
    for k in (s_names if not a.strict else shared):
        v = arrays[k]
        jagged = S[k].typename.endswith("[]")
        out[k] = v if jagged else ak.to_numpy(v)
    dtype_of = {"double": np.float64, "float": np.float32, "int32_t": np.int32, "bool": np.bool_, "int64_t": np.int64}

    def empty_jagged(dt):
        return ak.Array(ak.contents.ListOffsetArray(ak.index.Index64(np.zeros(n + 1, np.int64)), ak.contents.NumpyArray(np.zeros(0, dt))))

    # reference counter names: reuse them where a counter belongs to exactly one jagged branch
    ref_counter = {k: R[k].count_branch.name for k in R.keys() if R[k].count_branch is not None}
    from collections import Counter
    usage = Counter(ref_counter.values())
    counter_names = {}; zero_counters = set()
    for k in missing:
        c = ref_counter.get(k)
        if c is None: continue
        if usage[c] == 1: counter_names[k] = c
        else: zero_counters.add(c)  # shared counter: keep it as a scalar set to 0, give the branch a private _sz
    nested = []
    for k in missing:
        if k in counter_names.values() or k in ref_counter.values() and k not in zero_counters and usage.get(k, 0) == 1:
            continue  # written implicitly as the counter of its jagged branch
        t = R[k].typename; base = t.split("[")[0]; dt = dtype_of.get(base, np.float64)
        if k in zero_counters:
            out[k] = np.zeros(n, dtype=dt); continue
        if t.startswith("std::vector<std::vector"):
            nested.append(k); out[k] = empty_jagged(np.float64); continue
        if t.endswith("[]") or defaults.get(k) is None:
            out[k] = empty_jagged(dt)
        elif "[" in t:
            out[k] = np.broadcast_to(np.array(defaults[k], dtype=dt), (n,) + np.array(defaults[k]).shape).copy()
        else:
            out[k] = np.full(n, defaults[k], dtype=dt)
    if nested:
        print(f"note: {len(nested)} nested-vector branches cannot be written by uproot and are written as flat empty lists: {nested}")
    write_ntuple(a.out, out, counter_names=counter_names)
    with uproot.update(a.out) as f:
        f.mktree("Meta", {"POT_Used": np.float64, "POT_Total": np.float64, "Total_Reco_Entries": np.float64, "Total_Truth_Entries": np.float64})
        f["Meta"].extend({"POT_Used": np.array([a.pot]), "POT_Total": np.array([a.pot]), "Total_Reco_Entries": np.array([float(n)]),
                          "Total_Truth_Entries": np.array([a.truth_entries if a.truth_entries is not None else -1.0])})
    print(f"{a.out}: {n} events; {len(shared)} surrogate branches, {len(missing)} reference branches filled with defaults, "
          f"{0 if a.strict else len(only_s)} surrogate-only branches kept; Meta written (POT {a.pot})")


if __name__ == "__main__":
    main()
