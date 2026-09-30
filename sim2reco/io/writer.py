"""Write a pruned MasterAnaDev-style ntuple with uproot, matching the open-data tuple's types and counters."""
from __future__ import annotations

import awkward as ak
import numpy as np
import uproot


def _counter_name(name: str) -> str:
    return name + "_sz"


def write_ntuple(path: str, branches: dict, tree: str = "MasterAnaDev", counter_names: dict | None = None) -> None:
    """branches: {name: numpy array (scalar / fixed-size) or awkward jagged array}.

    Jagged branches get a MINERvA-style `<name>_sz` int32 counter unless `counter_names` maps the branch to
    another counter name (each counter name may be used by one branch only). Types follow the input dtypes, so
    pass int32 for integer branches and bool for booleans where the tuple uses them.
    """
    cn = counter_names or {}
    data = {}
    for k, v in branches.items():
        if isinstance(v, ak.Array):
            data[k] = v
        else:
            data[k] = np.asarray(v)
    with uproot.recreate(path) as f:
        f.mktree(tree, {k: _type_of(v) for k, v in data.items()}, counter_name=lambda name: cn.get(name, _counter_name(name)))
        f[tree].extend(data)


def _type_of(v):
    if isinstance(v, ak.Array):
        return v.type
    v = np.asarray(v)
    if v.ndim == 1:
        return v.dtype
    return np.dtype((v.dtype, v.shape[1:]))
