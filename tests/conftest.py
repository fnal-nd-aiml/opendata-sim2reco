"""Fixtures load the example ntuple if present, otherwise the slimmed Parquet (+ typenames JSON) of the same file."""
import json
import pathlib
import awkward as ak
import pytest
import uproot

from sim2reco.io.branches import RECO_TREE_BRANCHES, TRUTH_TREE_BRANCHES

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
# any slimmed MC file with its typenames JSON serves as the fixture (playlist 1A by default)
_cands = sorted(ROOT_DIR.glob("data/slim*/MasterAnaDev_mc_AnaTuple_run*_Playlist.reco.typenames.json"))
STEM = _cands[0].name.replace(".reco.typenames.json", "") if _cands else "MasterAnaDev_mc_AnaTuple_run00110000_Playlist"
ROOT_FILE = ROOT_DIR / f"{STEM}.root"
SLIM = (_cands[0].parent if _cands else ROOT_DIR / "data" / "slim_1A") / STEM
N_RECO = 20000
N_TRUTH = 40000


def _source():
    if ROOT_FILE.exists():
        return "root"
    if (SLIM.parent / f"{STEM}.reco.parquet").exists():
        return "parquet"
    pytest.skip("neither the example ntuple nor its slimmed Parquet is present")


@pytest.fixture(scope="session")
def reco():
    if _source() == "root":
        return uproot.open(ROOT_FILE)["MasterAnaDev"].arrays(RECO_TREE_BRANCHES, entry_stop=N_RECO)
    return ak.from_parquet(SLIM.parent / f"{STEM}.reco.parquet")[:N_RECO]


@pytest.fixture(scope="session")
def truth():
    if _source() == "root":
        return uproot.open(ROOT_FILE)["Truth"].arrays(TRUTH_TREE_BRANCHES, entry_stop=N_TRUTH)
    return ak.from_parquet(SLIM.parent / f"{STEM}.truth.parquet")[:N_TRUTH]


@pytest.fixture(scope="session")
def reco_typenames():
    if _source() == "root":
        t = uproot.open(ROOT_FILE)["MasterAnaDev"]
        names = RECO_TREE_BRANCHES + [b + "_sz" for b in RECO_TREE_BRANCHES if b + "_sz" in t.keys()]
        return {k: t[k].typename for k in names}
    return json.loads((SLIM.parent / f"{STEM}.reco.typenames.json").read_text())
