# janus/datamodules/class_map.py
"""
TotalSegmentator to Janus organ mapping.
Maps 117 TotalSeg labels -> 19 organ channels (14 organs + 5 computed spaces)
"""

import numpy as np

TOTALSEG_CLASS_MAP = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
    5: "liver", 6: "stomach", 7: "pancreas", 8: "adrenal_gland_right",
    9: "adrenal_gland_left", 10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right", 13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right",
    15: "esophagus", 16: "trachea", 17: "thyroid_gland", 18: "small_bowel",
    19: "duodenum", 20: "colon", 21: "urinary_bladder", 22: "prostate",
    23: "kidney_cyst_left", 24: "kidney_cyst_right",
    25: "sacrum", 26: "vertebrae_S1", 27: "vertebrae_L5", 28: "vertebrae_L4",
    29: "vertebrae_L3", 30: "vertebrae_L2", 31: "vertebrae_L1", 32: "vertebrae_T12",
    33: "vertebrae_T11", 34: "vertebrae_T10", 35: "vertebrae_T9", 36: "vertebrae_T8",
    37: "vertebrae_T7", 38: "vertebrae_T6", 39: "vertebrae_T5", 40: "vertebrae_T4",
    41: "vertebrae_T3", 42: "vertebrae_T2", 43: "vertebrae_T1", 44: "vertebrae_C7",
    45: "vertebrae_C6", 46: "vertebrae_C5", 47: "vertebrae_C4", 48: "vertebrae_C3",
    49: "vertebrae_C2", 50: "vertebrae_C1", 51: "heart", 52: "aorta",
    53: "pulmonary_vein", 54: "brachiocephalic_trunk", 55: "subclavian_artery_right",
    56: "subclavian_artery_left", 57: "common_carotid_artery_right",
    58: "common_carotid_artery_left", 59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right", 61: "atrial_appendage_left",
    62: "superior_vena_cava", 63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein", 65: "iliac_artery_left",
    66: "iliac_artery_right", 67: "iliac_vena_left", 68: "iliac_vena_right",
    69: "humerus_left", 70: "humerus_right", 71: "scapula_left", 72: "scapula_right",
    73: "clavicula_left", 74: "clavicula_right", 75: "femur_left", 76: "femur_right",
    77: "hip_left", 78: "hip_right", 79: "spinal_cord",
    80: "gluteus_maximus_left", 81: "gluteus_maximus_right",
    82: "gluteus_medius_left", 83: "gluteus_medius_right",
    84: "gluteus_minimus_left", 85: "gluteus_minimus_right",
    86: "autochthon_left", 87: "autochthon_right",
    88: "iliopsoas_left", 89: "iliopsoas_right", 90: "brain", 91: "skull",
    92: "rib_left_1", 93: "rib_left_2", 94: "rib_left_3", 95: "rib_left_4",
    96: "rib_left_5", 97: "rib_left_6", 98: "rib_left_7", 99: "rib_left_8",
    100: "rib_left_9", 101: "rib_left_10", 102: "rib_left_11", 103: "rib_left_12",
    104: "rib_right_1", 105: "rib_right_2", 106: "rib_right_3", 107: "rib_right_4",
    108: "rib_right_5", 109: "rib_right_6", 110: "rib_right_7", 111: "rib_right_8",
    112: "rib_right_9", 113: "rib_right_10", 114: "rib_right_11", 115: "rib_right_12",
    116: "sternum", 117: "costal_cartilages",
}

LABEL_TO_ID_TOTALSEG = {v: k for k, v in TOTALSEG_CLASS_MAP.items()}

SPINE_LABELS = list(range(25, 51))
RIB_LABELS = list(range(92, 116))
LUNG_LABELS = list(range(10, 15))

ALL_BONE_LABELS = (
    SPINE_LABELS + RIB_LABELS + [116, 117] + list(range(69, 79)) + [91]
)

# Organ merge: 117 TotalSeg labels -> 20 channels
# Format: channel_idx: (organ_name, [totalseg_label_ids])
ORGAN_CHANNELS = {
    0: ("liver", [5]),
    1: ("gallbladder", [4]),
    2: ("pancreas", [7]),
    3: ("spleen", [1]),
    4: ("kidneys", [2, 3]),
    5: ("kidney_cysts", []),  # Loaded from separate file
    6: ("prostate", [22]),
    7: ("stomach_esophagus", [6, 15]),
    8: ("small_bowel", [18, 19]),
    9: ("colon", [20]),
    10: ("lungs", [10, 11, 12, 13, 14]),
    11: ("heart", [51, 61]),
    12: ("aorta", [52, 54, 55, 56, 57, 58, 65, 66]),
    13: ("veins", [53, 59, 60, 62, 63, 64, 67, 68]),
    14: ("bones", ALL_BONE_LABELS),
    15: ("pleural_space", []),  # Computed by dilating lungs
    16: ("periportal_space", []),  # Computed by dilating liver
    17: ("perivascular_space", []),  # Computed by dilating vessels
    18: ("pericardial_space", []),  # Computed by dilating heart
    19: ("subcutaneous_space", []),  # Computed as body - organs
}

# Organ names in channel order
ORGAN_NAMES = [ORGAN_CHANNELS[i][0] for i in range(20)]

# Channel indices
LIVER_CH = 0
GALLBLADDER_CH = 1
PANCREAS_CH = 2
SPLEEN_CH = 3
KIDNEYS_CH = 4
KIDNEY_CYSTS_CH = 5
PROSTATE_CH = 6
STOMACH_ESOPHAGUS_CH = 7
SMALL_BOWEL_CH = 8
COLON_CH = 9
LUNGS_CH = 10
HEART_CH = 11
AORTA_CH = 12
VEINS_CH = 13
BONES_CH = 14
PLEURAL_SPACE_CH = 15
PERIPORTAL_SPACE_CH = 16
PERIVASCULAR_SPACE_CH = 17
PERICARDIAL_SPACE_CH = 18
SUBCUTANEOUS_SPACE_CH = 19


def get_organ_id_map(scheme="totalseg", merge_name="radioprior_v1", organs=None):
    """Map TotalSeg label IDs -> organ names. Returns dict {label_id: organ_name}."""
    organ_id_map = {}
    organ_set = set(organs) if organs else None

    for _, (organ_name, label_ids) in ORGAN_CHANNELS.items():
        if organ_set is None or organ_name in organ_set:
            for label_id in label_ids:
                organ_id_map[label_id] = organ_name

    return organ_id_map


# Janus aliases
JANUS_V1_CHANNEL_LIST = ORGAN_NAMES
JANUS_V1_ORGANS = ORGAN_NAMES
KIDNEYS_CHANNEL_IDX = KIDNEYS_CH
LUNGS_CHANNEL_IDX = LUNGS_CH
LIVER_CHANNEL_IDX = LIVER_CH
HEART_CHANNEL_IDX = HEART_CH
AORTA_CHANNEL_IDX = AORTA_CH
VEINS_CHANNEL_IDX = VEINS_CH
PLEURAL_SPACE_CHANNEL_IDX = PLEURAL_SPACE_CH
PERIPORTAL_SPACE_CHANNEL_IDX = PERIPORTAL_SPACE_CH
PERIVASCULAR_SPACE_CHANNEL_IDX = PERIVASCULAR_SPACE_CH
PERICARDIAL_SPACE_CHANNEL_IDX = PERICARDIAL_SPACE_CH
SUBCUTANEOUS_SPACE_CHANNEL_IDX = SUBCUTANEOUS_SPACE_CH


def merge_kidney_cysts_to_masks(masks, kidney_cyst_segmentation):
    """Add kidney cysts to kidney_cysts channel (channel 5)."""
    cyst_mask = (kidney_cyst_segmentation == 1) | (kidney_cyst_segmentation == 2)
    masks[KIDNEY_CYSTS_CH] = cyst_mask.astype(np.uint8)
    return masks


def compute_pleural_space_mask(masks, spacing_mm, dilation_mm=4.0):
    """Dilate lungs by dilation_mm, subtract original = pleural space shell."""
    from scipy.ndimage import binary_dilation

    lung_mask = masks[LUNGS_CH]
    if lung_mask.sum() == 0:
        return masks

    dilation_voxels = [max(1, int(np.round(dilation_mm / spacing_mm[i]))) for i in range(3)]
    struct = np.zeros((2 * dilation_voxels[2] + 1, 2 * dilation_voxels[1] + 1, 2 * dilation_voxels[0] + 1), dtype=bool)
    center = np.array(dilation_voxels[::-1])

    for z in range(struct.shape[0]):
        for y in range(struct.shape[1]):
            for x in range(struct.shape[2]):
                pos = np.array([z, y, x])
                if np.sum(((pos - center) / dilation_voxels[::-1]) ** 2) <= 1.0:
                    struct[z, y, x] = True

    dilated = binary_dilation(lung_mask > 0, structure=struct).astype(np.uint8)
    masks[PLEURAL_SPACE_CH] = dilated - (lung_mask > 0).astype(np.uint8)
    return masks


def create_dilated_mask(mask, spacing_mm, dilation_mm):
    """Dilate mask by dilation_mm, return shell only (dilated - original)."""
    from scipy.ndimage import binary_dilation

    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    dilation_voxels = [max(1, int(np.round(dilation_mm / spacing_mm[i]))) for i in range(3)]
    struct = np.zeros((2 * dilation_voxels[2] + 1, 2 * dilation_voxels[1] + 1, 2 * dilation_voxels[0] + 1), dtype=bool)
    center = np.array(dilation_voxels[::-1])

    for z in range(struct.shape[0]):
        for y in range(struct.shape[1]):
            for x in range(struct.shape[2]):
                pos = np.array([z, y, x])
                if np.sum(((pos - center) / dilation_voxels[::-1]) ** 2) <= 1.0:
                    struct[z, y, x] = True

    dilated = binary_dilation(mask > 0, structure=struct).astype(np.uint8)
    return dilated - (mask > 0).astype(np.uint8)


def compute_all_dilated_spaces(masks, spacing_mm, body_mask=None):
    """Compute dilated space masks for channels 16-19."""
    masks[PERIPORTAL_SPACE_CH] = create_dilated_mask(masks[LIVER_CH], spacing_mm, dilation_mm=4.0)

    vascular = ((masks[AORTA_CH] > 0) | (masks[VEINS_CH] > 0)).astype(np.uint8)
    masks[PERIVASCULAR_SPACE_CH] = create_dilated_mask(vascular, spacing_mm, dilation_mm=3.0)

    masks[PERICARDIAL_SPACE_CH] = create_dilated_mask(masks[HEART_CH], spacing_mm, dilation_mm=3.0)

    if body_mask is not None:
        all_organs = np.zeros_like(body_mask, dtype=np.uint8)
        for i in range(15):  # First 15 channels are actual organs (0-14)
            all_organs |= (masks[i] > 0).astype(np.uint8)
        masks[SUBCUTANEOUS_SPACE_CH] = ((body_mask > 0) & (all_organs == 0)).astype(np.uint8)
    else:
        masks[SUBCUTANEOUS_SPACE_CH] = np.zeros_like(masks[0], dtype=np.uint8)

    return masks
