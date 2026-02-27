# Disease → Organ Attention Mapping

Defines which segmentation masks are used as attention priors for each disease.
Source: `janus/configs/disease_config.py` → `DEFAULT_ATTENTION_CONFIG`

---

## Single-organ (focused attention on one structure)

| Disease | Organ Mask | Dilation |
|---|---|---|
| hepatomegaly | liver | 3 mm |
| splenomegaly | spleen | 3 mm |
| prostatomegaly | prostate | 3 mm |
| coronary_calcification | heart | 3 mm |
| abdominal_aortic_aneurysm | aorta | 3 mm |
| atherosclerosis | aorta | 3 mm |
| gallstones | gallbladder | 3 mm |
| pancreatic_atrophy | pancreas | 3 mm |
| hydronephrosis | kidneys | 3 mm |
| renal_hypodensities | kidneys | 3 mm |
| atelectasis | lungs | 3 mm |
| osteopenia | bones | 0 mm |
| fracture | bones | 0 mm |

---

## Union (attention pooled over multiple structures)

| Disease | Organ Masks | Dilation |
|---|---|---|
| cardiomegaly | heart ∪ lungs | 4 mm |
| aortic_valve_calcification | heart ∪ aorta | 3 mm |
| biliary_ductal_dilation | liver ∪ gallbladder ∪ pancreas | 3 mm |
| surgically_absent_gallbladder | gallbladder ∪ liver | 3 mm |
| renal_cyst | kidneys ∪ kidney_cysts | 3 mm |
| pleural_effusion | lungs ∪ pleural_space | 3 mm |
| bowel_obstruction | small_bowel ∪ colon | 3 mm |
| submucosal_edema | small_bowel ∪ colon | 3 mm |
| hiatal_hernia | stomach_esophagus ∪ lungs | 3 mm |

---

## Comparative (two organs pooled separately, then concatenated)

| Disease | Organ A | Organ B | Dilation |
|---|---|---|---|
| hepatic_steatosis | liver | spleen | 3 mm |

The model separately pools visual features inside each organ, then concatenates before classification. Captures the liver-to-spleen attenuation ratio signal.

---

## ROI (localised sub-organ region)

| Disease | Base Mask | ROI Key | Dilation |
|---|---|---|---|
| appendicitis | colon | appendix_roi | 3 mm |

---

## Global (no organ mask — full volume attention)

| Disease | Reason |
|---|---|
| free_air | Diffuse; no stable anatomical anchor |
| thrombosis | Ovarian/portal veins not fully segmented |
| ascites | Peritoneal fluid; distributed across multiple cavities |
| anasarca | Diffuse soft-tissue oedema; no stable anchor |
| metastatic_disease | Unpredictable multi-organ distribution |
| lymphadenopathy | Nodal chains not segmented |

---

## Segmentation channels reference

| Channel | Organ |
|---|---|
| 0 | liver |
| 1 | gallbladder |
| 2 | pancreas |
| 3 | spleen |
| 4 | kidneys |
| 5 | kidney_cysts |
| 6 | prostate |
| 7 | stomach_esophagus |
| 8 | small_bowel |
| 9 | colon |
| 10 | lungs |
| 11 | heart |
| 12 | aorta |
| 13 | veins |
| 14 | bones |
| 15 | pleural_space |
| 16 | periportal_space |
| 17 | perivascular_space |
| 18 | pericardial_space |
| 19 | subcutaneous_space |
