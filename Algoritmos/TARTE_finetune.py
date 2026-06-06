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

# MONKEY PATCH: Fixes Out of Memory (OOM) bug in TARTEFinetuneClassifier during validation and testing
# By default, the library evaluates the entire validation/test dataset in a single giant batch.
def patched_eval(self, model, ds_eval):
    with torch.no_grad():
        model.eval()
        batch_size = 32
        num_samples = ds_eval[0].size(0)
        for i in range(0, num_samples, batch_size):
            ds_batch = [
                ds_eval[0][i:i+batch_size].to(self.device_),
                ds_eval[1][i:i+batch_size].to(self.device_),
                ds_eval[2][i:i+batch_size].to(self.device_),
                ds_eval[-1][i:i+batch_size].to(self.device_)
            ]
            out = model(ds_batch[0], ds_batch[1], ds_batch[2])
            target = ds_batch[-1].view(-1).to(torch.float32)
            if self.loss == "categorical_crossentropy":
                target = target.to(torch.long)
            if self.output_dim_ == 1:
                out = out.view(-1).to(torch.float32)
                target = target.to(torch.float32)
            self.valid_loss_metric_.update(out, target)
            
        loss_eval = self.valid_loss_metric_.compute()
        loss_eval = loss_eval.detach().item()
        if self.valid_loss_flag_ == "neg":
            loss_eval = -1 * loss_eval
        self.valid_loss_metric_.reset()
    return loss_eval

def patched_generate_output(self, X, model_list, weights):
    from tarte_ai.tarte_finetune_estimator import TARTETabularDataset
    from torch.utils.data import DataLoader
    from scipy.special import softmax
    import numpy as np
    
    ds_test = TARTETabularDataset(X)
    batch_size_test = min(32, len(X))
    test_loader = DataLoader(ds_test, batch_size=batch_size_test, shuffle=False)

    out_total = []
    with torch.no_grad():
        for ds_predict_eval in test_loader:
            ds_predict_eval[0] = ds_predict_eval[0].to(self.device_)
            ds_predict_eval[1] = ds_predict_eval[1].to(self.device_)
            ds_predict_eval[2] = ds_predict_eval[2].to(self.device_)
            ds_predict_eval[-1] = ds_predict_eval[-1].to(self.device_)
            
            out = [
                model(ds_predict_eval[0], ds_predict_eval[1], ds_predict_eval[2])
                .cpu()
                .detach()
                .numpy()
                for model in model_list
            ]
            out = np.average(out, weights=weights, axis=0)
            out_total.append(out)
            
    out = np.concatenate(out_total, axis=0)

    if self.loss == "binary_crossentropy":
        out = 1 / (1 + np.exp(-out))
    elif self.loss == "categorical_crossentropy":
        out = softmax(out, axis=1)

    if np.isnan(out).sum() > 0:
        mean_pred = np.mean(self.y_)
        out[np.isnan(out)] = mean_pred

    if out.ndim == 2 and out.shape[1] == 1:
        out = out.squeeze(axis=1)

    return out

tarte_ai.tarte_finetune_estimator.BaseTARTEFinetuneEstimator._eval = patched_eval
tarte_ai.tarte_finetune_estimator.BaseTARTEFinetuneEstimator._generate_output = patched_generate_output

from sklearn.metrics import balanced_accuracy_score, roc_curve, auc
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
)
from sklearn.decomposition import PCA

# ==========================================
# 1. CONFIGURATION AND REPRODUCIBILITY
# ==========================================
SEED = 42

print(f"\n{'='*75}")
print(f"[*] USING RANDOM SEED FOR THIS RUN: {SEED}")
print(f"{'='*75}\n")

def set_seeds(seed):
    """Sets all seeds to guarantee the reproducibility of this specific run."""
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
    'IsthmusCingulate2_TS_ASYM': 'Isthmus of Cingulate Gyrus Cortical Thickness Standard Deviation Asymmetry (V2)',
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

print(f"\nDimensions -> TRAIN: {X_train.shape}, TEST: {X_test.shape}")

# ==========================================
# 5. PREPROCESSING FOR TARTE (GRAPHS)
# ==========================================
print("\n[+] Combining Train and Validation for Fine-Tuning...")
X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_train_val = np.concatenate([y_train, y_val])

print("\n[+] Preprocessing joint tables to TARTE graph format...")
start_prep = time.time()
preprocessor = tarte_ai.TARTE_TablePreprocessor()
X_train_graphs = preprocessor.fit_transform(X_train_val, y_train_val)
X_test_graphs = preprocessor.transform(X_test)
print(f" -> Preprocessing completed in {time.time() - start_prep:.2f} seconds.")

# ==========================================
# 6. TARTE FINE-TUNING MODEL
# ==========================================
print("\n[+] Initializing TARTEFinetuneClassifier...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Enhanced Fine-Tuning model configuration
ft_model = tarte_ai.TARTEFinetuneClassifier(
    loss='categorical_crossentropy',
    scoring='accuracy',
    finetune_strategy='freeze', # 'freeze' to avoid destroying transformer weights (Catastrophic Forgetting)
    learning_rate=1e-3,         # Increased to better converge the last layer
    batch_size=32,
    max_epoch=150,              # More epochs
    early_stopping_patience=20, # More patience
    random_state=SEED,
    device=device,
    disable_pbar=False
)

# TARTEFinetuneClassifier sometimes forgets to initialize _estimator_type
ft_model._estimator_type = "classifier"

# --- GLOBAL TRAINING (Train + Validation) ---
start_time = time.time()
print("\n[+] Training final Fine-Tuned model (TRAIN + VAL)...")
print(" -> This may require significant GPU VRAM and time.")

try:
    ft_model.fit(X_train_graphs, y_train_val)
except Exception as e:
    print(f"\n[!] Error during final training: {e}")
    print("[!] Trying to reduce batch_size...")
    ft_model.batch_size = 16
    ft_model.fit(X_train_graphs, y_train_val)

total_time = (time.time() - start_time) / 60
print(f"\n[+] Training completed in {total_time:.2f} minutes.")

print("\n[+] Evaluating Fine-Tuned model on TEST...")
y_pred = ft_model.predict(X_test_graphs)
y_prob = ft_model.predict_proba(X_test_graphs)

# Metrics for internal report
acc = accuracy_score(y_test, y_pred)
b_acc = balanced_accuracy_score(y_test, y_pred)
f1_mac = f1_score(y_test, y_pred, average='macro', zero_division=0)
try:
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
except:
    roc_auc = np.nan

# ==========================================
# 7. FORMATTED PRINTING OF METRICS
# ==========================================
def print_formatted_metrics(title, y_true, y_pred):
    acc_val = accuracy_score(y_true, y_pred)
    prec_val = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_val = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_val = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print("-" * 61)
    print(title)
    print("-" * 61)
    print(f"* Accuracy: {acc_val:.4f}")
    print(f"* Precision: {prec_val:.4f}")
    print(f"* Recall: {rec_val:.4f}")
    print(f"* F1 Score: {f1_val:.4f}")

print_formatted_metrics("METRICS FOR THE TEST SET:", y_test, y_pred)
print("-" * 61)

print("\n--- Classification Report (Fine-Tuning) ---")
print(classification_report(y_test, y_pred, zero_division=0))
print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# Visual Confusion Matrix
cm_test = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=le.classes_)
disp.plot(cmap="Purples", ax=ax_cm, values_format='d')
plt.title(f"Confusion Matrix - TARTE Fine-Tuning\n(Seed: {SEED})", fontsize=12)
plt.xlabel("Predicted Diagnosis")
plt.ylabel("True Diagnosis")
plt.tight_layout()
plt.savefig(f"Matriz_Confusion_TARTE_FineTuning_Seed{SEED}.svg", dpi=300)
plt.close()

# ROC Curve
y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))
n_classes = y_test_bin.shape[1]
fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c'])
lw = 2
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc_val = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color=color, lw=lw,
                label=f'ROC Curve for {le.classes_[i]} (AUC = {roc_auc_val:0.3f})')
ax_roc.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - Fine-Tuning (Seed: {SEED})', fontsize=12)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"Curva_ROC_TARTE_FineTuning_Seed{SEED}.svg", dpi=300)
plt.close()
