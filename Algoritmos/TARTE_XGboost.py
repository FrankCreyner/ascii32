import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random
import time
import re
import warnings
from itertools import cycle

warnings.filterwarnings("ignore")
import torch
import tarte_ai

def patched_extract_per_layer(self, x, edge_attr, mask, model_config, num_layers_transformer):
    model_config["num_layers_transformer"] = num_layers_transformer
    model_config["dim_projection"] = self.dim_embedding
    model = tarte_ai.TARTE_Pretrain_NN(**model_config)

    max_dim_ = np.max(self.pretrained_model_configs_["dim_projection"])
    if self.dim_embedding == max_dim_:
        model.tarte_linear = torch.nn.ModuleDict({"tab_enc": torch.nn.Identity()})
    else:
        remove_idx = [2]
        remove_idx += [i for i in range(4, 8)]
        for idx in remove_idx:
            model.tarte_linear[f"proj_{self.dim_embedding}"][idx] = torch.nn.Identity()

    model = model.to(self.device_)
    model.load_state_dict(self.pretrain_model_dict_, strict=False)
    model.layer_norm = torch.nn.LayerNorm(model_config["dim_transformer"]).to(self.device_)

    with torch.no_grad():
        model.eval()
        X_ = model(x, edge_attr, mask)[0]
        X_ = X_.detach().to("cpu").numpy()

    return X_

tarte_ai.TARTE_TableEncoder._extract_per_layer = patched_extract_per_layer

from sklearn.metrics import balanced_accuracy_score, roc_curve, auc
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
)

# ==========================================
# 1. CONFIGURATION AND REPRODUCIBILITY
# ==========================================
SEED = 42

print(f"\n{'='*75}")
print(f"[*] USING FIXED REPRODUCIBILITY SEED: {SEED}")
print(f"{'='*75}\n")

def set_seeds(seed):
    """Sets all seeds to guarantee reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
set_seeds(SEED)

# ==========================================
# 2. DEFINITION OF MAPPINGS AND TRANSLATIONS
# ==========================================
tarte_mapping = {
    'PTID': 'Patient Identifier',
    'EXAMDATE': 'Examination Date',
    'AGE': 'Patient Age',
    'PTGENDER': 'Patient Gender',
    'PTEDUCAT': 'Patient Education Level in Years',
    'PTETHCAT': 'Patient Ethnic Category',
    'PTRACCAT': 'Patient Racial Category',
    'APOE4': 'Apolipoprotein E epsilon 4 allele count',
    'FDG': 'Fluorodeoxyglucose PET scan glucose metabolism',
    'AV45': 'Florbetapir PET scan amyloid beta load',
    'ADAS13': 'Alzheimer Disease Assessment Scale 13 items',
    'MMSE': 'Mini Mental State Examination score',
    'FAQ': 'Functional Activities Questionnaire',
    'MOCA': 'Montreal Cognitive Assessment',
    'RAVLT_immediate': 'Rey Auditory Verbal Learning Test immediate recall',
    'RAVLT_learning': 'Rey Auditory Verbal Learning Test learning score',
    'RAVLT_forgetting': 'Rey Auditory Verbal Learning Test forgetting score',
    'RAVLT_perc_forgetting': 'Rey Auditory Verbal Learning Test percentage forgetting',
    'EcogPtMem': 'Everyday Cognition Patient self report Memory',
    'EcogPtLang': 'Everyday Cognition Patient self report Language',
    'EcogPtVisspat': 'Everyday Cognition Patient self report Visuospatial abilities',
    'EcogPtPlan': 'Everyday Cognition Patient self report Planning',
    'EcogPtOrgan': 'Everyday Cognition Patient self report Organization',
    'EcogPtDivatt': 'Everyday Cognition Patient self report Divided attention',
    'EcogSPMem': 'Everyday Cognition Study Partner report Memory',
    'EcogSPLang': 'Everyday Cognition Study Partner report Language',
    'EcogSPVisspat': 'Everyday Cognition Study Partner report Visuospatial abilities',
    'EcogSPPlan': 'Everyday Cognition Study Partner report Planning',
    'EcogSPOrgan': 'Everyday Cognition Study Partner report Organization',
    'EcogSPDivatt': 'Everyday Cognition Study Partner report Divided attention',
    'Ventricles': 'Ventricles volume',
    'Hippocampus': 'Hippocampus volume',
    'WholeBrain': 'Whole Brain volume',
    'Entorhinal': 'Entorhinal cortex volume',
    'Fusiform': 'Fusiform gyrus volume',
    'MidTemp': 'Middle Temporal gyrus volume',
    'ICV': 'Intracranial Volume',
    'DX': 'Clinical Diagnosis'
}

tarte_mapping_multimodal = {
    'HCI_BAIPETNMRC_09_12_16': 'Hypometabolic Convergence Index from FDG PET',
    'SROI_BAIPETNMRC_09_12_16': 'Statistically predefined Region of Interest glucose metabolism from FDG PET',
    'MCSUVRWM3_BAIPETNMRC_09_12_16': 'Motion Corrected Standardized Uptake Value Ratio with White Matter reference from FDG PET',
    'CEREBELLUMGREYMATTER_UCBERKELEYAV45_10_17_16': 'Cerebellum Grey Matter Florbetapir AV45 PET amyloid uptake',
    'ERODED_SUBCORTICALWM_UCBERKELEYAV45_10_17_16': 'Eroded Subcortical White Matter Florbetapir AV45 PET amyloid uptake',
    'SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF_UCBERKELEYAV45_10_17_16': 'Summary Florbetapir AV45 PET amyloid uptake normalized by whole cerebellum',
    'SUMMARYSUVR_COMPOSITE_REFNORM_0.79CUTOFF_UCBERKELEYAV45_10_17_16': 'Summary Florbetapir AV45 PET amyloid uptake normalized by composite reference',
    'VENTRICLE_3RD_UCBERKELEYAV45_10_17_16': 'Third Ventricle Florbetapir AV45 PET amyloid uptake',
    'VENTRICLE_4TH_UCBERKELEYAV45_10_17_16': 'Fourth Ventricle Florbetapir AV45 PET amyloid uptake',
    'CC_ANTERIOR_UCBERKELEYAV45_10_17_16': 'Anterior Corpus Callosum Florbetapir AV45 PET amyloid uptake',
    'CC_CENTRAL_UCBERKELEYAV45_10_17_16': 'Central Corpus Callosum Florbetapir AV45 PET amyloid uptake',
    'CTX_LH_CUNEUS_UCBERKELEYAV45_10_17_16': 'Left Hemisphere Cuneus Cortex Florbetapir AV45 PET amyloid uptake',
    'CTX_LH_ENTORHINAL_UCBERKELEYAV45_10_17_16': 'Left Hemisphere Entorhinal Cortex Florbetapir AV45 PET amyloid uptake',
    'LEFT_CAUDATE_UCBERKELEYAV45_10_17_16': 'Left Caudate Florbetapir AV45 PET amyloid uptake',
    'LEFT_CHOROID_PLEXUS_UCBERKELEYAV45_10_17_16': 'Left Choroid Plexus Florbetapir AV45 PET amyloid uptake',
    'LEFT_HIPPOCAMPUS_UCBERKELEYAV45_10_17_16': 'Left Hippocampus Florbetapir AV45 PET amyloid uptake',
    'LEFT_PALLIDUM_UCBERKELEYAV45_10_17_16': 'Left Pallidum Florbetapir AV45 PET amyloid uptake',
    'LEFT_THALAMUS_PROPER_UCBERKELEYAV45_10_17_16': 'Left Thalamus Proper Florbetapir AV45 PET amyloid uptake',
    'NON_WM_HYPOINTENSITIES_UCBERKELEYAV45_10_17_16': 'Non-White Matter Hypointensities Florbetapir AV45 PET amyloid uptake',
    'OPTIC_CHIASM_UCBERKELEYAV45_10_17_16': 'Optic Chiasm Florbetapir AV45 PET amyloid uptake',
    'CTX_LH_FRONTALPOLE_SIZE_UCBERKELEYAV45_10_17_16': 'Left Hemisphere Frontal Pole Cortex volume size',
    'CTX_LH_MEDIALORBITOFRONTAL_SIZE_UCBERKELEYAV45_10_17_16': 'Left Hemisphere Medial Orbitofrontal Cortex volume size',
    'CTX_LH_TEMPORALPOLE_SIZE_UCBERKELEYAV45_10_17_16': 'Left Hemisphere Temporal Pole Cortex volume size',
    'CTX_LH_UNKNOWN_SIZE_UCBERKELEYAV45_10_17_16': 'Left Hemisphere Unknown Cortex Region volume size',
    'CTX_RH_FRONTALPOLE_SIZE_UCBERKELEYAV45_10_17_16': 'Right Hemisphere Frontal Pole Cortex volume size',
    'CTX_RH_TEMPORALPOLE_SIZE_UCBERKELEYAV45_10_17_16': 'Right Hemisphere Temporal Pole Cortex volume size',
    'CTX_RH_UNKNOWN_SIZE_UCBERKELEYAV45_10_17_16': 'Right Hemisphere Unknown Cortex Region volume size',
    'LEFT_ACCUMBENS_AREA_SIZE_UCBERKELEYAV45_10_17_16': 'Left Accumbens Area volume size',
    'LEFT_PALLIDUM_SIZE_UCBERKELEYAV45_10_17_16': 'Left Pallidum volume size',
    'LEFT_THALAMUS_PROPER_SIZE_UCBERKELEYAV45_10_17_16': 'Left Thalamus Proper volume size',
    'OPTIC_CHIASM_SIZE_UCBERKELEYAV45_10_17_16': 'Optic Chiasm volume size',
    'RIGHT_ACCUMBENS_AREA_SIZE_UCBERKELEYAV45_10_17_16': 'Right Accumbens Area volume size',
    'RIGHT_PALLIDUM_SIZE_UCBERKELEYAV45_10_17_16': 'Right Pallidum volume size',
    'ABETA_UPENNBIOMK9_04_19_17': 'Cerebrospinal fluid Amyloid Beta 1-42 biomarker',
    'TAU_UPENNBIOMK9_04_19_17': 'Cerebrospinal fluid Total Tau biomarker',
    'PTAU_UPENNBIOMK9_04_19_17': 'Cerebrospinal fluid Phosphorylated Tau biomarker',
    'Bankssts_SA_MEAN': 'Banks of the Superior Temporal Sulcus Surface Area Mean',
    'Bankssts_SA_ASYM': 'Banks of the Superior Temporal Sulcus Surface Area Asymmetry',
    'CaudalMiddleFrontal_SA_MEAN': 'Caudal Middle Frontal gyrus Surface Area Mean',
    'CaudalMiddleFrontal_SA_ASYM': 'Caudal Middle Frontal gyrus Surface Area Asymmetry'
}

tarte_mapping_surface_area = {
    'Cuneus_SA_MEAN': 'Cuneus Cortex Surface Area Mean',
    'Cuneus_SA_ASYM': 'Cuneus Cortex Surface Area Asymmetry',
    'Entorhinal_SA_MEAN': 'Entorhinal Cortex Surface Area Mean',
    'Entorhinal_SA_ASYM': 'Entorhinal Cortex Surface Area Asymmetry',
    'FrontalPole_SA_MEAN': 'Frontal Pole Cortex Surface Area Mean',
    'FrontalPole_SA_ASYM': 'Frontal Pole Cortex Surface Area Asymmetry',
    'Fusiform_SA_MEAN': 'Fusiform Gyrus Surface Area Mean',
    'Fusiform_SA_ASYM': 'Fusiform Gyrus Surface Area Asymmetry',
    'InferiorParietal_SA_MEAN': 'Inferior Parietal Cortex Surface Area Mean',
    'InferiorParietal_SA_ASYM': 'Inferior Parietal Cortex Surface Area Asymmetry',
    'InferiorTemporal_SA_MEAN': 'Inferior Temporal Gyrus Surface Area Mean',
    'InferiorTemporal_SA_ASYM': 'Inferior Temporal Gyrus Surface Area Asymmetry',
    'MiddleTemporal_SA_MEAN': 'Middle Temporal Gyrus Surface Area Mean',
    'MiddleTemporal_SA_ASYM': 'Middle Temporal Gyrus Surface Area Asymmetry',
    'IsthmusCingulate_SA_MEAN': 'Isthmus of Cingulate Gyrus Surface Area Mean',
    'IsthmusCingulate_SA_ASYM': 'Isthmus of Cingulate Gyrus Surface Area Asymmetry',
    'PosteriorCingulate_SA_MEAN': 'Posterior Cingulate Cortex Surface Area Mean',
    'PosteriorCingulate_SA_ASYM': 'Posterior Cingulate Cortex Surface Area Asymmetry',
    'LateralOccipital_SA_MEAN': 'Lateral Occipital Cortex Surface Area Mean',
    'LateralOccipital_SA_ASYM': 'Lateral Occipital Cortex Surface Area Asymmetry',
    'LateralOrbitofrontal_SA_MEAN': 'Lateral Orbitofrontal Cortex Surface Area Mean',
    'LateralOrbitofrontal_SA_ASYM': 'Lateral Orbitofrontal Cortex Surface Area Asymmetry',
    'Lingual_SA_MEAN': 'Lingual Gyrus Surface Area Mean',
    'Lingual_SA_ASYM': 'Lingual Gyrus Surface Area Asymmetry',
    'Paracentral_SA_MEAN': 'Paracentral Lobule Surface Area Mean',
    'Paracentral_SA_ASYM': 'Paracentral Lobule Surface Area Asymmetry',
    'Parahippocampal_SA_MEAN': 'Parahippocampal Gyrus Surface Area Mean',
    'Parahippocampal_SA_ASYM': 'Parahippocampal Gyrus Surface Area Asymmetry',
    'Pericalcarine_SA_MEAN': 'Pericalcarine Cortex Surface Area Mean',
    'Pericalcarine_SA_ASYM': 'Pericalcarine Cortex Surface Area Asymmetry',
    'ParsOpercularis_SA_MEAN': 'Pars Opercularis Surface Area Mean',
    'ParsOpercularis_SA_ASYM': 'Pars Opercularis Surface Area Asymmetry',
    'ParsOrbitalis_SA_MEAN': 'Pars Orbitalis Surface Area Mean',
    'ParsOrbitalis_SA_ASYM': 'Pars Orbitalis Surface Area Asymmetry',
    'ParsTriangularis_SA_MEAN': 'Pars Triangularis Surface Area Mean',
    'ParsTriangularis_SA_ASYM': 'Pars Triangularis Surface Area Asymmetry',
    'Postcentral_SA_MEAN': 'Postcentral Gyrus Surface Area Mean',
    'Postcentral_SA_ASYM': 'Postcentral Gyrus Surface Area Asymmetry',
    'Precentral_SA_MEAN': 'Precentral Gyrus Surface Area Mean',
    'Precentral_SA_ASYM': 'Precentral Gyrus Surface Area Asymmetry'
}

tarte_mapping_cortical = {
    'Precuneus_SA_MEAN': 'Precuneus Cortex Surface Area Mean',
    'Precuneus_SA_ASYM': 'Precuneus Cortex Surface Area Asymmetry',
    'RostralMiddleFrontal_SA_MEAN': 'Rostral Middle Frontal Gyrus Surface Area Mean',
    'RostralMiddleFrontal_SA_ASYM': 'Rostral Middle Frontal Gyrus Surface Area Asymmetry',
    'SuperiorParietal_SA_MEAN': 'Superior Parietal Cortex Surface Area Mean',
    'SuperiorParietal_SA_ASYM': 'Superior Parietal Cortex Surface Area Asymmetry',
    'SuperiorTemporal_SA_MEAN': 'Superior Temporal Gyrus Surface Area Mean',
    'SuperiorTemporal_SA_ASYM': 'Superior Temporal Gyrus Surface Area Asymmetry',
    'Supramarginal_SA_MEAN': 'Supramarginal Gyrus Surface Area Mean',
    'Supramarginal_SA_ASYM': 'Supramarginal Gyrus Surface Area Asymmetry',
    'TemporalPole_SA_MEAN': 'Temporal Pole Cortex Surface Area Mean',
    'TemporalPole_SA_ASYM': 'Temporal Pole Cortex Surface Area Asymmetry',
    'TransverseTemporal_SA_MEAN': 'Transverse Temporal Gyrus Surface Area Mean',
    'TransverseTemporal_SA_ASYM': 'Transverse Temporal Gyrus Surface Area Asymmetry',
    'Bankssts2_TA_MEAN': 'Banks of the Superior Temporal Sulcus Cortical Thickness Average Mean',
    'Bankssts2_TA_ASYM': 'Banks of the Superior Temporal Sulcus Cortical Thickness Average Asymmetry',
    'CaudalAnteriorCingulate2_TA_MEAN': 'Caudal Anterior Cingulate Cortex Cortical Thickness Average Mean',
    'CaudalAnteriorCingulate2_TA_ASYM': 'Caudal Anterior Cingulate Cortex Cortical Thickness Average Asymmetry',
    'Cuneus2_TA_MEAN': 'Cuneus Cortex Cortical Thickness Average Mean',
    'Cuneus2_TA_ASYM': 'Cuneus Cortex Cortical Thickness Average Asymmetry',
    'Entorhinal2_TA_MEAN': 'Entorhinal Cortex Cortical Thickness Average Mean',
    'Entorhinal2_TA_ASYM': 'Entorhinal Cortex Cortical Thickness Average Asymmetry',
    'FrontalPole2_TA_MEAN': 'Frontal Pole Cortex Cortical Thickness Average Mean',
    'FrontalPole2_TA_ASYM': 'Frontal Pole Cortex Cortical Thickness Average Asymmetry',
    'Fusiform2_TA_MEAN': 'Fusiform Gyrus Cortical Thickness Average Mean',
    'Fusiform2_TA_ASYM': 'Fusiform Gyrus Cortical Thickness Average Asymmetry',
    'InferiorParietal2_TA_MEAN': 'Inferior Parietal Cortex Cortical Thickness Average Mean',
    'InferiorParietal2_TA_ASYM': 'Inferior Parietal Cortex Cortical Thickness Average Asymmetry',
    'Bankssts3_TS_MEAN': 'Banks of the Superior Temporal Sulcus Cortical Thickness Standard Deviation Mean',
    'Bankssts3_TS_ASYM': 'Banks of the Superior Temporal Sulcus Cortical Thickness Standard Deviation Asymmetry',
    'CaudalAnteriorCingulate3_TS_MEAN': 'Caudal Anterior Cingulate Cortex Cortical Thickness Standard Deviation Mean',
    'CaudalAnteriorCingulate3_TS_ASYM': 'Caudal Anterior Cingulate Cortex Cortical Thickness Standard Deviation Asymmetry',
    'Cuneus3_TS_MEAN': 'Cuneus Cortex Cortical Thickness Standard Deviation Mean',
    'Cuneus3_TS_ASYM': 'Cuneus Cortex Cortical Thickness Standard Deviation Asymmetry',
    'Entorhinal3_TS_MEAN': 'Entorhinal Cortex Cortical Thickness Standard Deviation Mean',
    'Entorhinal3_TS_ASYM': 'Entorhinal Cortex Cortical Thickness Standard Deviation Asymmetry',
    'FrontalPole3_TS_MEAN': 'Frontal Pole Cortex Cortical Thickness Standard Deviation Mean',
    'FrontalPole3_TS_ASYM': 'Frontal Pole Cortex Cortical Thickness Standard Deviation Asymmetry',
    'Fusiform3_TS_MEAN': 'Fusiform Gyrus Cortical Thickness Standard Deviation Mean',
    'Fusiform3_TS_ASYM': 'Fusiform Gyrus Cortical Thickness Standard Deviation Asymmetry'
}

tarte_mapping_cortical_2 = {
    'InferiorParietal3_TS_MEAN': 'Inferior Parietal Cortex Cortical Thickness Standard Deviation Mean',
    'InferiorParietal3_TS_ASYM': 'Inferior Parietal Cortex Cortical Thickness Standard Deviation Asymmetry',
    'InferiorTemporal2_TA_MEAN': 'Inferior Temporal Gyrus Cortical Thickness Average Mean',
    'InferiorTemporal2_TA_ASYM': 'Inferior Temporal Gyrus Cortical Thickness Average Asymmetry',
    'InferiorTemporal3_TS_MEAN': 'Inferior Temporal Gyrus Cortical Thickness Standard Deviation Mean',
    'InferiorTemporal3_TS_ASYM': 'Inferior Temporal Gyrus Cortical Thickness Standard Deviation Asymmetry',
    'IsthmusCingulate2_TS_MEAN': 'Isthmus of Cingulate Gyrus Cortical Thickness Standard Deviation Mean (V2)',
    'Coffee_Cingulate2_TS_ASYM': 'Isthmus of Cingulate Gyrus Cortical Thickness Standard Deviation Asymmetry (V2)',
    'IsthmusCingulate3_TS_MEAN': 'Isthmus of Cingulate Gyrus Cortical Thickness Standard Deviation Mean (V3)',
    'IsthmusCingulate3_TS_ASYM': 'Isthmus of Cingulate Gyrus Cortical Thickness Standard Deviation Asymmetry (V3)',
    'LateralOccipital2_TA_MEAN': 'Lateral Occipital Cortex Cortical Thickness Average Mean',
    'LateralOccipital2_TA_ASYM': 'Lateral Occipital Cortex Cortical Thickness Average Asymmetry',
    'LateralOccipital3_TS_MEAN': 'Lateral Occipital Cortex Cortical Thickness Standard Deviation Mean',
    'LateralOccipital3_TS_ASYM': 'Lateral Occipital Cortex Cortical Thickness Standard Deviation Asymmetry',
    'LateralOrbitofrontal2_TA_MEAN': 'Lateral Orbitofrontal Cortex Cortical Thickness Average Mean',
    'LateralOrbitofrontal2_TA_ASYM': 'Lateral Orbitofrontal Cortex Cortical Thickness Average Asymmetry',
    'LateralOrbitofrontal3_TS_MEAN': 'Lateral Orbitofrontal Cortex Cortical Thickness Standard Deviation Mean',
    'LateralOrbitofrontal3_TS_ASYM': 'Lateral Orbitofrontal Cortex Cortical Thickness Standard Deviation Asymmetry',
    'Lingual2_TA_MEAN': 'Lingual Gyrus Cortical Thickness Average Mean',
    'Lingual2_TA_ASYM': 'Lingual Gyrus Cortical Thickness Average Asymmetry',
    'Lingual3_TS_MEAN': 'Lingual Gyrus Cortical Thickness Standard Deviation Mean',
    'Lingual3_TS_ASYM': 'Lingual Gyrus Cortical Thickness Standard Deviation Asymmetry',
    'MedialOrbitofrontal2_TA_MEAN': 'Medial Orbitofrontal Cortex Cortical Thickness Average Mean',
    'MedialOrbitofrontal2_TA_ASYM': 'Medial Orbitofrontal Cortex Cortical Thickness Average Asymmetry',
    'MedialOrbitofrontal3_TS_MEAN': 'Medial Orbitofrontal Cortex Cortical Thickness Standard Deviation Mean',
    'MedialOrbitofrontal3_TS_ASYM': 'Medial Orbitofrontal Cortex Cortical Thickness Standard Deviation Asymmetry',
    'MiddleTemporal2_TA_MEAN': 'Middle Temporal Gyrus Cortical Thickness Average Mean',
    'MiddleTemporal2_TA_ASYM': 'Middle Temporal Gyrus Cortical Thickness Average Asymmetry',
    'MiddleTemporal3_TS_MEAN': 'Middle Temporal Gyrus Cortical Thickness Standard Deviation Mean',
    'MiddleTemporal3_TS_ASYM': 'Middle Temporal Gyrus Cortical Thickness Standard Deviation Asymmetry',
    'Paracentral2_TA_MEAN': 'Paracentral Lobule Cortical Thickness Average Mean',
    'Paracentral2_TA_ASYM': 'Paracentral Lobule Cortical Thickness Average Asymmetry',
    'Paracentral3_TS_MEAN': 'Paracentral Lobule Cortical Thickness Standard Deviation Mean',
    'Paracentral3_TS_ASYM': 'Paracentral Lobule Cortical Thickness Standard Deviation Asymmetry',
    'Parahippocampal2_TA_MEAN': 'Parahippocampal Gyrus Cortical Thickness Average Mean',
    'Parahippocampal2_TA_ASYM': 'Parahippocampal Gyrus Cortical Thickness Average Asymmetry',
    'Parahippocampal3_TS_MEAN': 'Parahippocampal Gyrus Cortical Thickness Standard Deviation Mean',
    'Parahippocampal3_TS_ASYM': 'Parahippocampal Gyrus Cortical Thickness Standard Deviation Asymmetry',
    'ParsOpercularis2_TA_MEAN': 'Pars Opercularis Cortical Thickness Average Mean',
    'ParsOpercularis2_TA_ASYM': 'Pars Opercularis Cortical Thickness Average Asymmetry'
}

# Wait, check if there was a typo in line 253 of XGboost script: 'Coffee_Cingulate2_TS_ASYM' or 'IsthmusCingulate2_TS_ASYM'?
# In the original TARTE_XGboost (2).py:
# 253:     'IsthmusCingulate2_TS_ASYM': 'Isthmus of Cingulate Gyrus Cortical Thickness Standard Deviation Asymmetry (V2)',
# Ah, the original had 'IsthmusCingulate2_TS_ASYM'. Let's ensure it's mapped exactly as 'IsthmusCingulate2_TS_ASYM'.

tarte_mapping_cortical_3 = {
    'ParsOpercularis3_TS_ASYM': 'Pars Opercularis Cortical Thickness Standard Deviation Asymmetry',
    'ParsOrbitalis2_TA_MEAN': 'Pars Orbitalis Cortical Thickness Average Mean',
    'ParsOrbitalis2_TA_ASYM': 'Pars Orbitalis Cortical Thickness Average Asymmetry',
    'ParsOrbitalis3_TS_MEAN': 'Pars Orbitalis Cortical Thickness Standard Deviation Mean',
    'ParsOrbitalis3_TS_ASYM': 'Pars Orbitalis Cortical Thickness Standard Deviation Asymmetry',
    'ParsTriangularis2_TA_MEAN': 'Pars Triangularis Cortical Thickness Average Mean',
    'ParsTriangularis2_TA_ASYM': 'Pars Triangularis Cortical Thickness Average Asymmetry',
    'ParsTriangularis3_TS_MEAN': 'Pars Triangularis Cortical Thickness Standard Deviation Mean',
    'ParsTriangularis3_TS_ASYM': 'Pars Triangularis Cortical Thickness Standard Deviation Asymmetry',
    'Pericalcarine2_TA_MEAN': 'Pericalcarine Cortex Cortical Thickness Average Mean',
    'Pericalcarine2_TA_ASYM': 'Pericalcarine Cortex Cortical Thickness Average Asymmetry',
    'Pericalcarine3_TS_MEAN': 'Pericalcarine Cortex Cortical Thickness Standard Deviation Mean',
    'Pericalcarine3_TS_ASYM': 'Pericalcarine Cortex Cortical Thickness Standard Deviation Asymmetry',
    'Postcentral2_TA_MEAN': 'Postcentral Gyrus Cortical Thickness Average Mean',
    'Postcentral2_TA_ASYM': 'Postcentral Gyrus Cortical Thickness Average Asymmetry',
    'Postcentral3_TS_MEAN': 'Postcentral Gyrus Cortical Thickness Standard Deviation Mean',
    'Postcentral3_TS_ASYM': 'Postcentral Gyrus Cortical Thickness Standard Deviation Asymmetry',
    'PosteriorCingulate2_TA_MEAN': 'Posterior Cingulate Cortex Cortical Thickness Average Mean',
    'PosteriorCingulate2_TA_ASYM': 'Posterior Cingulate Cortex Cortical Thickness Average Asymmetry',
    'PosteriorCingulate3_TS_MEAN': 'Posterior Cingulate Cortex Cortical Thickness Standard Deviation Mean',
    'PosteriorCingulate3_TS_ASYM': 'Posterior Cingulate Cortex Cortical Thickness Standard Deviation Asymmetry',
    'Precentral3_TS_MEAN': 'Precentral Gyrus Cortical Thickness Standard Deviation Mean',
    'Precentral3_TS_ASYM': 'Precentral Gyrus Cortical Thickness Standard Deviation Asymmetry',
    'Precuneus2_TA_MEAN': 'Precuneus Cortex Cortical Thickness Average Mean',
    'Precuneus2_TA_ASYM': 'Precuneus Cortex Cortical Thickness Average Asymmetry',
    'Precuneus3_TS_MEAN': 'Precuneus Cortex Cortical Thickness Standard Deviation Mean',
    'Precuneus3_TS_ASYM': 'Precuneus Cortex Cortical Thickness Standard Deviation Asymmetry',
    'RostralAnteriorCingulate2_TA_MEAN': 'Rostral Anterior Cingulate Cortex Cortical Thickness Average Mean',
    'RostralAnteriorCingulate2_TA_ASYM': 'Rostral Anterior Cingulate Cortex Cortical Thickness Average Asymmetry',
    'RostralAnteriorCingulate3_TS_MEAN': 'Rostral Anterior Cingulate Cortex Cortical Thickness Standard Deviation Mean',
    'RostralAnteriorCingulate3_TS_ASYM': 'Rostral Anterior Cingulate Cortex Cortical Thickness Standard Deviation Asymmetry',
    'RostralMiddleFrontal2_TA_MEAN': 'Rostral Middle Frontal Gyrus Cortical Thickness Average Mean',
    'RostralMiddleFrontal2_TA_ASYM': 'Rostral Middle Frontal Gyrus Cortical Thickness Average Asymmetry',
    'RostralMiddleFrontal3_TS_MEAN': 'Rostral Middle Frontal Gyrus Cortical Thickness Standard Deviation Mean',
    'RostralMiddleFrontal3_TS_ASYM': 'Rostral Middle Frontal Gyrus Cortical Thickness Standard Deviation Asymmetry',
    'SuperiorFrontal3_TS_MEAN': 'Superior Frontal Gyrus Cortical Thickness Standard Deviation Mean',
    'SuperiorFrontal3_TS_ASYM': 'Superior Frontal Gyrus Cortical Thickness Standard Deviation Asymmetry',
    'SuperiorParietal3_TS_MEAN': 'Superior Parietal Cortex Cortical Thickness Standard Deviation Mean',
    'SuperiorParietal3_TS_ASYM': 'Superior Parietal Cortex Cortical Thickness Standard Deviation Asymmetry',
    'SuperiorTemporal2_TA_MEAN': 'Superior Temporal Gyrus Cortical Thickness Average Mean'
}

tarte_mapping_cortical_4 = {
    'SuperiorTemporal2_TA_ASYM': 'Superior Temporal Gyrus Cortical Thickness Average Asymmetry',
    'SuperiorTemporal3_TS_MEAN': 'Superior Temporal Gyrus Cortical Thickness Standard Deviation Mean',
    'SuperiorTemporal3_TS_ASYM': 'Superior Temporal Gyrus Cortical Thickness Standard Deviation Asymmetry',
    'Supramarginal2_TA_MEAN': 'Supramarginal Gyrus Cortical Thickness Average Mean',
    'Supramarginal2_TA_ASYM': 'Supramarginal Gyrus Cortical Thickness Average Asymmetry',
    'Supramarginal3_TS_MEAN': 'Supramarginal Gyrus Cortical Thickness Standard Deviation Mean',
    'Supramarginal3_TS_ASYM': 'Supramarginal Gyrus Cortical Thickness Standard Deviation Asymmetry',
    'TemporalPole2_TA_MEAN': 'Temporal Pole Cortex Cortical Thickness Average Mean',
    'TemporalPole2_TA_ASYM': 'Temporal Pole Cortex Cortical Thickness Average Asymmetry',
    'TemporalPole3_TS_MEAN': 'Temporal Pole Cortex Cortical Thickness Standard Deviation Mean',
    'TemporalPole3_TS_ASYM': 'Temporal Pole Cortex Cortical Thickness Standard Deviation Asymmetry',
    'TransverseTemporal2_TA_MEAN': 'Transverse Temporal Gyrus Cortical Thickness Average Mean',
    'TransverseTemporal2_TA_ASYM': 'Transverse Temporal Gyrus Cortical Thickness Average Asymmetry',
    'TransverseTemporal3_TS_MEAN': 'Transverse Temporal Gyrus Cortical Thickness Standard Deviation Mean',
    'TransverseTemporal3_TS_ASYM': 'Transverse Temporal Gyrus Cortical Thickness Standard Deviation Asymmetry',
    'INSULA2_TA_MEAN': 'Insula Cortex Cortical Thickness Average Mean',
    'INSULA2_TA_ASYM': 'Insula Cortex Cortical Thickness Average Asymmetry',
    'INSULA3_TS_MEAN': 'Insula Cortex Cortical Thickness Standard Deviation Mean',
    'INSULA3_TS_ASYM': 'Insula Cortex Cortical Thickness Standard Deviation Asymmetry',
    'Accumbens_SV_MEAN': 'Accumbens Area Structural Volume Mean',
    'Accumbens_SV_ASYM': 'Accumbens Area Structural Volume Asymmetry',
    'Amygdala_SV_MEAN': 'Amygdala Structural Volume Mean',
    'Amygdala_SV_ASYM': 'Amygdala Structural Volume Asymmetry',
    'ChoroidPlexus_SV_MEAN': 'Choroid Plexus Structural Volume Mean',
    'ChoroidPlexus_SV_ASYM': 'Choroid Plexus Structural Volume Asymmetry',
    'InferiorLateralVentricle_SV_MEAN': 'Inferior Lateral Ventricle Structural Volume Mean',
    'InferiorLateralVentricle_SV_ASYM': 'Inferior Lateral Ventricle Structural Volume Asymmetry',
    'Bankssts_CV_MEAN': 'Banks of the Superior Temporal Sulcus Cortical Volume Mean',
    'Bankssts_CV_ASYM': 'Banks of the Superior Temporal Sulcus Cortical Volume Asymmetry',
    'CaudalAnteriorCingulate_CV_MEAN': 'Caudal Anterior Cingulate Cortex Cortical Volume Mean',
    'CaudalAnteriorCingulate_CV_ASYM': 'Caudal Anterior Cingulate Cortex Cortical Volume Asymmetry',
    'CaudalMiddleFrontal_CV_MEAN': 'Caudal Middle Frontal Gyrus Cortical Volume Mean',
    'CaudalMiddleFrontal_CV_ASYM': 'Caudal Middle Frontal Gyrus Cortical Volume Asymmetry',
    'Cuneus_CV_MEAN': 'Cuneus Cortex Cortical Volume Mean',
    'Cuneus_CV_ASYM': 'Cuneus Cortex Cortical Volume Asymmetry',
    'Entorhinal_CV_MEAN': 'Entorhinal Cortex Cortical Volume Mean',
    'Entorhinal_CV_ASYM': 'Entorhinal Cortex Cortical Volume Asymmetry',
    'FrontalPole_CV_MEAN': 'Frontal Pole Cortex Cortical Volume Mean',
    'FrontalPole_CV_ASYM': 'Frontal Pole Cortex Cortical Volume Asymmetry',
    'InferiorParietal_CV_MEAN': 'Inferior Parietal Cortex Cortical Volume Mean'
}

tarte_mapping_final = {
    'InferiorParietal_CV_ASYM': 'Inferior Parietal Cortex Cortical Volume Asymmetry',
    'InferiorTemporal_CV_MEAN': 'Inferior Temporal Gyrus Cortical Volume Mean',
    'InferiorTemporal_CV_ASYM': 'Inferior Temporal Gyrus Cortical Volume Asymmetry',
    'IsthmusCingulate_CV_MEAN': 'Isthmus of Cingulate Gyrus Cortical Volume Mean',
    'IsthmusCingulate_CV_ASYM': 'Isthmus of Cingulate Gyrus Cortical Volume Asymmetry',
    'LateralOccipital_CV_MEAN': 'Lateral Occipital Cortex Cortical Volume Mean',
    'LateralOccipital_CV_ASYM': 'Lateral Occipital Cortex Cortical Volume Asymmetry',
    'LateralOrbitofrontal_CV_MEAN': 'Lateral Orbitofrontal Cortex Cortical Volume Mean',
    'LateralOrbitofrontal_CV_ASYM': 'Lateral Orbitofrontal Cortex Cortical Volume Asymmetry',
    'Lingual_CV_MEAN': 'Lingual Gyrus Cortical Volume Mean',
    'Lingual_CV_ASYM': 'Lingual Gyrus Cortical Volume Asymmetry',
    'MedialOrbitofrontal_CV_MEAN': 'Medial Orbitofrontal Cortex Cortical Volume Mean',
    'MedialOrbitofrontal_CV_ASYM': 'Medial Orbitofrontal Cortex Cortical Volume Asymmetry',
    'Paracentral_CV_MEAN': 'Paracentral Lobule Cortical Volume Mean',
    'Paracentral_CV_ASYM': 'Paracentral Lobule Cortical Volume Asymmetry',
    'Parahippocampal_CV_MEAN': 'Parahippocampal Gyrus Cortical Volume Mean',
    'Parahippocampal_CV_ASYM': 'Parahippocampal Gyrus Cortical Volume Asymmetry',
    'ParsOpercularis_CV_MEAN': 'Pars Opercularis Cortical Volume Mean',
    'ParsOpercularis_CV_ASYM': 'Pars Opercularis Cortical Volume Asymmetry',
    'ParsOrbitalis_CV_MEAN': 'Pars Orbitalis Cortical Volume Mean',
    'ParsOrbitalis_CV_ASYM': 'Pars Orbitalis Cortical Volume Asymmetry',
    'ParsTriangularis_CV_MEAN': 'Pars Triangularis Cortical Volume Mean',
    'ParsTriangularis_CV_ASYM': 'Pars Triangularis Cortical Volume Asymmetry',
    'Pericalcarine_CV_MEAN': 'Pericalcarine Cortex Cortical Volume Mean',
    'Pericalcarine_CV_ASYM': 'Pericalcarine Cortex Cortical Volume Asymmetry',
    'Postcentral_CV_MEAN': 'Postcentral Gyrus Cortical Volume Mean',
    'Postcentral_CV_ASYM': 'Postcentral Gyrus Cortical Volume Asymmetry',
    'PosteriorCingulate_CV_MEAN': 'Posterior Cingulate Cortex Cortical Volume Mean',
    'PosteriorCingulate_CV_ASYM': 'Posterior Cingulate Cortex Cortical Volume Asymmetry',
    'Precentral_CV_MEAN': 'Precentral Gyrus Cortical Volume Mean',
    'Precentral_CV_ASYM': 'Precentral Gyrus Cortical Volume Asymmetry',
    'Precuneus_CV_MEAN': 'Precuneus Cortex Cortical Volume Mean',
    'Precuneus_CV_ASYM': 'Precuneus Cortex Cortical Volume Asymmetry',
    'RostralAnteriorCingulate_CV_MEAN': 'Rostral Anterior Cingulate Cortex Cortical Volume Mean',
    'RostralAnteriorCingulate_CV_ASYM': 'Rostral Anterior Cingulate Cortex Cortical Volume Asymmetry',
    'RostralMiddleFrontal_CV_MEAN': 'Rostral Middle Frontal Gyrus Cortical Volume Mean',
    'RostralMiddleFrontal_CV_ASYM': 'Rostral Middle Frontal Gyrus Cortical Volume Asymmetry',
    'SuperiorFrontal_CV_MEAN': 'Superior Frontal Gyrus Cortical Volume Mean',
    'SuperiorFrontal_CV_ASYM': 'Superior Frontal Gyrus Cortical Volume Asymmetry',
    'SuperiorParietal_CV_MEAN': 'Superior Parietal Cortex Cortical Volume Mean',
    'SuperiorParietal_CV_ASYM': 'Superior Parietal Cortex Cortical Volume Asymmetry',
    'SuperiorTemporal_CV_MEAN': 'Superior Temporal Gyrus Cortical Volume Mean',
    'SuperiorTemporal_CV_ASYM': 'Superior Temporal Gyrus Cortical Volume Asymmetry',
    'Supramarginal_CV_MEAN': 'Supramarginal Gyrus Cortical Volume Mean',
    'Supramarginal_CV_ASYM': 'Supramarginal Gyrus Cortical Volume Asymmetry',
    'TemporalPole_CV_MEAN': 'Temporal Pole Cortex Cortical Volume Mean',
    'TemporalPole_CV_ASYM': 'Temporal Pole Cortex Cortical Volume Asymmetry',
    'TransverseTemporal_CV_MEAN': 'Transverse Temporal Gyrus Cortical Volume Mean',
    'TransverseTemporal_CV_ASYM': 'Transverse Temporal Gyrus Cortical Volume Asymmetry',
    'Pallidum_SV_MEAN': 'Pallidum Structural Volume Mean',
    'Pallidum_SV_ASYM': 'Pallidum Structural Volume Asymmetry',
    'Vessel_SV_MEAN': 'Blood Vessel Structural Volume Mean',
    'Vessel_SV_ASYM': 'Blood Vessel Structural Volume Asymmetry'
}

master_mapping = {
    **tarte_mapping,
    **tarte_mapping_multimodal,
    **tarte_mapping_surface_area,
    **tarte_mapping_cortical,
    **tarte_mapping_cortical_2,
    **tarte_mapping_cortical_3,
    **tarte_mapping_cortical_4,
    **tarte_mapping_final
}

def translate_freesurfer(column_name):
    match = re.search(r'ST(\d+)([A-Z]{2})_', column_name)
    
    if not match:
        return column_name 
        
    region_id = match.group(1)
    metric_code = match.group(2)
    
    metrics = {
        'TA': 'Cortical Thickness Average',
        'TS': 'Cortical Thickness Standard Deviation',
        'SA': 'Surface Area',
        'SV': 'Structural Volume',
        'CV': 'Cortical Volume'
    }
    
    metric_text = metrics.get(metric_code, 'Measurement')
    return f"Brain MRI Region {region_id} {metric_text}"

def apply_tarte_mapping(df, master_mapping):
    new_columns = []
    for col in df.columns:
        if col in master_mapping:
            new_columns.append(master_mapping[col])
        else:
            new_columns.append(translate_freesurfer(col))
            
    df_mapped = df.copy()
    df_mapped.columns = new_columns
    
    # Deduplicate column names to prevent skrub/tarte-ai from failing when receiving DataFrames instead of Series
    unique_cols = []
    counts = {}
    for col in df_mapped.columns:
        if col in counts:
            counts[col] += 1
            unique_cols.append(f"{col} ({counts[col]})")
        else:
            counts[col] = 0
            unique_cols.append(col)
            
    df_mapped.columns = unique_cols
    return df_mapped

# ==========================================
# 3. STRICT PREPROCESSING
# ==========================================
def preprocess_data(df, label_encoder=None, is_train=False):
    df = df[df["DX"].isin([0, 1, 2])].copy()
    mapping = {2: "NL", 1: "MCI", 0: "AD"}
    df["DX"] = df["DX"].map(mapping)
    
    cols_to_drop = ['DX', 'PTID', 'EXAMDATE', 'CDRSB']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y_raw = df["DX"]
    
    if is_train:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
    else:
        y = label_encoder.transform(y_raw)
        
    return X, y, label_encoder

# ==========================================
# 4. DATA LOADING AND MAIN FLOW
# ==========================================
print("Loading and preprocessing real TADPOLE data...")
df_train = pd.read_csv("TADPOLE_D1_D2_BL_RF_TRAIN.csv")
df_val = pd.read_csv("TADPOLE_D1_D2_BL_RF_VALIDATION.csv")
df_test = pd.read_csv("TADPOLE_D1_D2_BL_RF_TEST.csv")

print("\n--- Class Distribution (DX) ---")
print("TRAIN:\n", df_train["DX"].value_counts())
print("TEST:\n", df_test["DX"].value_counts())
print("VAL:\n", df_val["DX"].value_counts())

X_train, y_train, le = preprocess_data(df_train, is_train=True)
X_val, y_val, _ = preprocess_data(df_val, label_encoder=le)
X_test, y_test, _ = preprocess_data(df_test, label_encoder=le)

# Apply mapping to features
X_train = apply_tarte_mapping(X_train, master_mapping)
X_val   = apply_tarte_mapping(X_val, master_mapping)
X_test  = apply_tarte_mapping(X_test, master_mapping)

print(f"\nDimensions -> TRAIN: {X_train.shape}, VAL: {X_val.shape}, TEST: {X_test.shape}")

# Configure PredefinedSplit for training/validation
X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_train_val = np.concatenate([y_train, y_val])

test_fold = np.concatenate([np.full(X_train.shape[0], -1), np.zeros(X_val.shape[0])])
ps = PredefinedSplit(test_fold)

# ==========================================
# 5. TARTE ENCODER
# ==========================================
class TARTEFeaturizer:
    def __init__(self, output_dim=768, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.preprocessor = tarte_ai.TARTE_TablePreprocessor()
        self.encoder = tarte_ai.TARTE_TableEncoder(device=self.device, dim_embedding=output_dim)
        self.is_fitted_ = False

    def fit(self, X, y=None):
        X_prep = self.preprocessor.fit_transform(X)
        self.encoder.fit(X_prep)
        self.is_fitted_ = True
        return self

    def transform(self, X, batch_size=64):
        embeddings = []
        total_batches = (len(X) - 1) // batch_size + 1
        
        # Process in batches to prevent GPU/CPU from running out of memory
        for i in range(0, len(X), batch_size):
            batch_num = i // batch_size + 1
            print(f"    [Batch {batch_num}/{total_batches}] Extracting embeddings...")
            
            if isinstance(X, pd.DataFrame):
                X_batch = X.iloc[i:i+batch_size]
            else:
                X_batch = X[i:i+batch_size]
                
            X_prep_batch = self.preprocessor.transform(X_batch)
            emb_batch = self.encoder.transform(X_prep_batch)
            embeddings.append(emb_batch)
            
            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return np.vstack(embeddings)

    def fit_transform(self, X, y=None, batch_size=64):
        self.fit(X, y)
        return self.transform(X, batch_size=batch_size)

print("\n[+] Initializing TARTE and extracting real embeddings from the foundation model...")
start_emb = time.time()
tarte_model = TARTEFeaturizer()

print(" -> Extracting semantic embeddings for TRAIN + VAL...")
try:
    X_train_val_emb = tarte_model.fit_transform(X_train_val)
except torch.OutOfMemoryError:
    print("\n[WARNING] GPU ran out of memory (CUDA Out of Memory).")
    print(" -> Freeing CUDA cache and automatically retrying on CPU...")
    torch.cuda.empty_cache()
    tarte_model = TARTEFeaturizer(device='cpu')
    X_train_val_emb = tarte_model.fit_transform(X_train_val)

print(f" -> Completed in {time.time() - start_emb:.2f} seconds.")

start_test_emb = time.time()
print(" -> Extracting semantic embeddings for TEST...")
try:
    X_test_emb = tarte_model.transform(X_test)
except torch.OutOfMemoryError:
    print("\n[WARNING] GPU ran out of memory during TEST.")
    print(" -> Automatically retrying on CPU...")
    torch.cuda.empty_cache()
    tarte_model = TARTEFeaturizer(device='cpu')
    X_test_emb = tarte_model.transform(X_test)

print(f" -> Completed in {time.time() - start_test_emb:.2f} seconds.")
print(f"[+] Embedding extraction completed in {(time.time() - start_emb) / 60:.2f} minutes.")

X_train_val_orig_np = np.array(X_train_val)
X_test_orig_np = np.array(X_test)

# ==========================================
# 6. CONFIGURE DATA FOR TARTE BOOST
# ==========================================
# TARTE Boost combines original features with TARTE embeddings
X_train_val_boost = np.hstack([X_train_val_orig_np, X_train_val_emb])
X_test_boost = np.hstack([X_test_orig_np, X_test_emb])

# ==========================================
# 7. TRAINING OF TARTE BOOST + XGBOOST
# ==========================================
experiment_name = "TARTE Boost + XGBoost"
print(f"\n{'='*65}")
print(f"STARTING: {experiment_name} (SEED: {SEED})")
print(f"{'='*65}")

# Hyperparameters grid for XGBoost search
param_grid = {
    'modelo__n_estimators': [50, 100, 200],
    'modelo__max_depth': [3, 5, 7],
    'modelo__learning_rate': [0.01, 0.1, 0.2]
}

estimator = XGBClassifier(
    random_state=SEED, 
    use_label_encoder=False, 
    eval_metric='mlogloss'
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('modelo', estimator)
])

# GridSearchCV using PredefinedSplit
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=ps,
    n_jobs=2,          
    verbose=2,
    refit=True
)

start_time = time.time()
grid_search.fit(X_train_val_boost, y_train_val)
total_time = (time.time() - start_time) / 60
best_model = grid_search.best_estimator_

print(f"\n[+] Search finished in {total_time:.2f} min.")
print(f"[+] Best Hyperparameters: {grid_search.best_params_}")

y_test_pred = best_model.predict(X_test_boost)
y_test_prob = best_model.predict_proba(X_test_boost)

# ==========================================
# 8. FORMATTED PRINTING OF METRICS
# ==========================================
def print_formatted_metrics(title, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print("-" * 61)
    print(title)
    print("-" * 61)
    print(f"* Accuracy: {acc:.4f}")
    print(f"* Precision: {prec:.4f}")
    print(f"* Recall: {rec:.4f}")
    print(f"* F1 Score: {f1:.4f}")

print_formatted_metrics("METRICS FOR THE TEST SET:", y_test, y_test_pred)
print("-" * 61)

# ==========================================
# 9. EXPORT OF RESULTS AND PLOTS
# ==========================================
# Visual Confusion Matrix
cm_test = confusion_matrix(y_test, y_test_pred)
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=le.classes_)
disp.plot(cmap="Reds", ax=ax_cm, values_format='d')
plt.title(f"Confusion Matrix - TARTE Boost + XGBoost\n(Seed: {SEED})", fontsize=12)
plt.xlabel("Predicted Diagnosis")
plt.ylabel("True Diagnosis")
plt.tight_layout()
plt.savefig(f"Matriz_Confusion_TARTE_Boost_XGBoost_Real_Seed{SEED}.svg", dpi=300)
plt.close()

# Multiclass ROC Curve (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))
n_classes = y_test_bin.shape[1]
fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c'])
lw = 2
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_prob[:, i])
    roc_auc_val = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color=color, lw=lw,
                label=f'ROC Curve for {le.classes_[i]} (AUC = {roc_auc_val:0.3f})')
ax_roc.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity / Recall)')
plt.title(f'Multiclass ROC Curve - TARTE Boost + XGBoost (Seed: {SEED})', fontsize=12)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"Curva_ROC_TARTE_Boost_XGBoost_Real_Seed{SEED}.svg", dpi=300)
plt.close()