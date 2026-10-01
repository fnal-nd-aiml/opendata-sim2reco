"""Physical constants and MINERvA conventions used across the package. Units: MeV, mm, radians."""
import numpy as np

M_MU = 105.6583755
M_PI = 139.57039
M_PI0 = 134.9768
M_P = 938.27208816
M_N = 939.56542052
M_E = 0.51099895
M_K = 493.677
M_K0 = 497.611
M_LAMBDA = 1115.683
M_SIGMA_P = 1189.37
M_SIGMA_0 = 1192.642
M_SIGMA_M = 1197.449

# NuMI beam points below the detector z axis (rotation about x) by MINERvA's numi_beam_angle_rad = -0.05887
# (3.373 deg). All MasterAnaDev theta/phi branches are in that beam frame; Px/Py/Pz are detector frame.
# Verified by scanning the angle against the tuple (docs/tuple_notes.md).
BEAM_ANGLE_RAD = -0.05887

PDG_MASS = {
    13: M_MU, -13: M_MU, 11: M_E, -11: M_E, 22: 0.0,
    211: M_PI, -211: M_PI, 111: M_PI0,
    2212: M_P, 2112: M_N,
    321: M_K, -321: M_K, 311: M_K0, -311: M_K0, 130: M_K0, 310: M_K0,
    3122: M_LAMBDA, 3222: M_SIGMA_P, 3212: M_SIGMA_0, 3112: M_SIGMA_M,
}

# Species classes for the model input (docs/PROJECT.md 13.1 step 4). 0 is reserved for padding.
CLASS_NAMES = ["pad", "mu-", "e", "gamma", "p", "pi+", "pi-", "pi0", "K+-", "K0", "hyperon", "other", "n"]
NEUTRON_CLASS = 12  # only used when neutrons are admitted as input tokens (variant study, 2026-10-01)
PDG_CLASS = {
    13: 1, 11: 2, -11: 2, 22: 3, 2212: 4, 211: 5, -211: 6, 111: 7,
    321: 8, -321: 8, 311: 9, -311: 9, 130: 9, 310: 9,
    3122: 10, 3222: 10, 3212: 10, 3112: 10, 3322: 10, 3312: 10, 3334: 10,
    2112: 12,
}
N_CLASSES = len(CLASS_NAMES)

# Tuple fill conventions for the pruned output ntuple.
FILL_PRONG_MOM = -1.0     # pion_P/E/Px/Py/Pz for empty slot or failed fit
FILL_PRONG_T = -9999.0    # pion_T for empty slot or failed fit
FILL_PRONG_ANG = -9.0     # pion_theta/phi for empty slot
FILL_PRONG_INT = -1       # hadron_isExiting etc. for empty slot
FILL_PROTON = -9999.0     # proton_* scalars when no primary proton
N_PRONG_SLOTS = 10
