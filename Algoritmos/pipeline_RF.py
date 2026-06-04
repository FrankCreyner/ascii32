import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import os
from pathlib import Path

def missing_data(df, threshold=0.75):
    missing = df.isna().mean().sort_values(ascending=False)
    columnas_muy_vacias = missing[missing > threshold]
    print("\nColumnas con más del 75% de NaN:", len(columnas_muy_vacias))
    return columnas_muy_vacias.index.tolist()

def constant_columns(df):
    columnas_constantes = []
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            columnas_constantes.append(col)
    print("Columnas constantes:", len(columnas_constantes))
    return columnas_constantes

def duplicate_columns(df):
    duplicadas = df.T.duplicated()
    columnas_duplicadas = df.columns[duplicadas]
    print("Columnas duplicadas:", len(columnas_duplicadas))
    return list(columnas_duplicadas)

def reducir(df, df_st_regions, df_volumen, df_baipetnmrc, df_otras):
    new_cols = {}
    cols_to_drop = []

    # ---- ST regions ------
    for _, row in df_st_regions.iterrows():
        region = row["Region"]
        left = row["Left"]
        right = row["Right"]
        if left in df.columns and right in df.columns:
            new_cols[f"{region}_MEAN"] = (df[left] + df[right]) / 2
            new_cols[f"{region}_ASYM"] = (df[left] - df[right]) / (df[left] + df[right])
            cols_to_drop.extend([left, right])
            
    # ----- Volumen -------
    for _, row in df_volumen.iterrows():
        region = row["Region"]
        left = row["Left"]
        right = row["Right"]
        if left in df.columns and right in df.columns:
            mean_value = (df[left] + df[right]) / 2
            if "ICV" in df.columns:
                mean_value = mean_value / df["ICV"]
            new_cols[f"{region}_MEAN"] = mean_value
            new_cols[f"{region}_ASYM"] = (df[left] - df[right]) / (df[left] + df[right])
            cols_to_drop.extend([left, right])

    # --- BAIPETNMRC ----
    for _, row in df_baipetnmrc.iterrows():
        region = row["Region"]
        left = row["Left"]
        right = row["Right"]
        if left in df.columns and right in df.columns:
            new_cols[f"{region}_MEAN"] = (df[left] + df[right]) / 2
            new_cols[f"{region}_ASYM"] = (df[left] - df[right]) / (df[left] + df[right])
            cols_to_drop.extend([left, right])
    
    # --- Otras medidas ----
    for _, row in df_otras.iterrows():
        region = row["Region"]
        left = row["Left"]
        right = row["Right"]
        if left in df.columns and right in df.columns:
            new_cols[f"{region}_MEAN"] = (df[left] + df[right]) / 2
            new_cols[f"{region}_ASYM"] = (df[left] - df[right]) / (df[left] + df[right])
            cols_to_drop.extend([left, right])
            
    # Solución de Desempeño: Concatenar todo de una vez
    if new_cols:
        new_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_df], axis=1)
        
    df.drop(columns=list(set(cols_to_drop)), inplace=True, errors='ignore')
    return df

def eliminar_correlacion_alta(df, threshold=0.85):
    df_num = df.select_dtypes(include=[np.number])
    corr_matrix = df_num.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print("Columnas altamente correlacionadas:", len(to_drop))
    return to_drop

if __name__ == "__main__":
    # Ignoramos alertas de warning innecesarias de pandas (por si acaso quedan fragmentaciones)
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    print(f"Script directory: {script_dir}")
    
    # Read CSV files from the script's directory
    df = pd.read_csv(script_dir / "TADPOLE_D1_D2.csv", low_memory=False, na_values=["", " ", "-4", -4, "-1", -1, "NA"])

    df_baipetnmrc = pd.read_csv(script_dir / "BAIPETNMRC.csv", low_memory=False)
    df_volumen = pd.read_csv(script_dir / "volumen.csv")
    df_st_regions = pd.read_csv(script_dir / "st_regions.csv")
    df_otras = pd.read_csv(script_dir / "otras.csv")
    
    df_baipetnmrc.columns = df_baipetnmrc.columns.str.strip()
    df_volumen.columns = df_volumen.columns.str.strip()
    df_st_regions.columns = df_st_regions.columns.str.strip()
    df_otras.columns = df_otras.columns.str.strip()

    df_bl = df.copy()
    df_bl["PTID"] = df_bl["PTID"].str.strip().str.upper() 

    columnas_a_quitar = [
        "FLDSTRENG",
        "FSVERSION",
        "EXAMDATE_bl",
        "update_stamp",
        "EXAMDATE_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "VERSION_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "FLDSTRENG_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "LONISID_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "update_stamp_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "EXAMDATE_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "VERSION_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "FLDSTRENG_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "STATUS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "OVERALLQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "TEMPQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "FRONTQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "PARQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "INSULAQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "OCCQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "BGQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "CWMQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "VENTQC_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "STATUS_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "OVERALLQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "TEMPQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "FRONTQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "INSULAQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "update_stamp_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "EXAMDATE_BAIPETNMRC_09_12_16",
        "VERSION_BAIPETNMRC_09_12_16",
        "RUNDATE_BAIPETNMRC_09_12_16",
        "STATUS_BAIPETNMRC_09_12_16",
        "update_stamp_BAIPETNMRC_09_12_16",
        "EXAMDATE_UCBERKELEYAV45_10_17_16",
        "update_stamp_UCBERKELEYAV45_10_17_16",
        "EXAMDATE_UPENNBIOMK9_04_19_17",
        "PHASE_UPENNBIOMK9_04_19_17",
        "BATCH_UPENNBIOMK9_04_19_17",
        "KIT_UPENNBIOMK9_04_19_17",
        "STDS_UPENNBIOMK9_04_19_17",
        "RUNDATE_UPENNBIOMK9_04_19_17",
        "update_stamp_UPENNBIOMK9_04_19_17",
        "ORIGPROT",
        "D1",
        "D2",
        "SITE",
        "RID",
        "DX_bl",
        "COLPROT",
        "PTMARRY",
        "DXCHANGE",
        "BASETP1_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "BASETP2_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "BASETP3_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "BASETP4_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "BASETP5_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "BASETP6_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "BASETP7_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "BASETP8_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "ADAS11",
        "RUNDATE_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "CDRSB",
        "CDRSB_bl",
        "FLDSTRENG_bl",
        "FSVERSION_bl",
        "LONIUID_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "RUNDATE_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "PARQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "OCCQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "BGQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16",
        "CWMQC_UCSFFSL_02_01_16_UCSFFSL51ALL_08_01_16"
    ]
    
    columnas_a_quitar = [c for c in columnas_a_quitar if c in df_bl.columns]
    df_bl.drop(columns=columnas_a_quitar, inplace=True)

    df_bl["EXAMDATE"] = pd.to_datetime(df_bl["EXAMDATE"], format='%Y-%m-%d', errors='coerce') 
    
    df_bl = df_bl.sort_values(by=["PTID","EXAMDATE"])
    df_bl = df_bl.drop_duplicates(subset=["PTID"], keep="first").reset_index(drop=True)

    basura = ['', ' ', 'NA', 'N/A', 'null', 'None', '-', '?']
    df_bl.replace(basura, np.nan, inplace=True)

    codigos_missing = [-4, -1, 999, -999]
    columnas_numericas = df_bl.select_dtypes(include=[np.number]).columns
    for col in columnas_numericas:
        df_bl[col] = df_bl[col].mask(df_bl[col].isin(codigos_missing), np.nan)

    columnas_texto = df_bl.select_dtypes(include=["object"]).columns
    for col in columnas_texto:
        df_bl[col] = df_bl[col].str.strip()

    if "DX" in df_bl.columns:
        df_bl = df_bl.dropna(subset=["DX"]).reset_index(drop=True)
        
        # Filter to keep only the three main diagnostic classes
        print("\n--- Filtering DX classes ---")
        print("Original DX distribution:")
        for dx_val, count in df_bl["DX"].value_counts().sort_index().items():
            print(f"  {dx_val}: {count} samples")
        
        # Keep only stable diagnoses: Dementia (AD), MCI, and NL
        # Remove transition states like "MCI to Dementia", "NL to MCI", etc.
        valid_dx_values = ['Dementia', 'MCI', 'NL']
        df_bl = df_bl[df_bl["DX"].isin(valid_dx_values)].reset_index(drop=True)
        
        print("\nFiltered DX distribution (kept only stable diagnoses):")
        for dx_val, count in df_bl["DX"].value_counts().sort_index().items():
            print(f"  {dx_val}: {count} samples")
        
        # Stratified split: 60% train, 20% validation, 20% test
        print("\nPerforming stratified split: 60% train, 20% validation, 20% test")
        
        # First split: 60% train, 40% temp (which will become 20% val + 20% test)
        train_df, temp_df = train_test_split(
            df_bl,
            test_size=0.4,
            random_state=42,
            stratify=df_bl["DX"]
        )
        
        # Second split: Split temp into 50% validation and 50% test (20% each of original)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df["DX"]
        )
    else:
        # First split: 70% train, 30% temp
        train_df, temp_df = train_test_split(df_bl, test_size=0.3, random_state=42)
        
        # Second split: 50% validation, 50% test
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    print(f"\n--- Dataset Split ---")
    print(f"Train size: {len(train_df)} ({len(train_df)/len(df_bl)*100:.1f}%)")
    print(f"Validation size: {len(val_df)} ({len(val_df)/len(df_bl)*100:.1f}%)")
    print(f"Test size: {len(test_df)} ({len(test_df)/len(df_bl)*100:.1f}%)")

    print("\n--- Analizando Data en TRAIN Dataset ---")
    columnas_muy_vacias = missing_data(train_df)
    columnas_constantes = constant_columns(train_df)
    columnas_duplicadas = duplicate_columns(train_df)

    columnas_eliminar = list(set(columnas_muy_vacias + columnas_constantes + columnas_duplicadas))
    print("Columnas totales a eliminar (Train):", len(columnas_eliminar))
    
    # Apply column removal to all three sets
    train_df.drop(columns=columnas_eliminar, inplace=True, errors="ignore")
    val_df.drop(columns=columnas_eliminar, inplace=True, errors="ignore")
    test_df.drop(columns=columnas_eliminar, inplace=True, errors="ignore")

    to_drop_corr = eliminar_correlacion_alta(train_df, threshold=0.85)
    train_df.drop(columns=to_drop_corr, inplace=True, errors="ignore")
    val_df.drop(columns=to_drop_corr, inplace=True, errors="ignore")
    test_df.drop(columns=to_drop_corr, inplace=True, errors="ignore")

    print("Columnas restantes en Train/Val/Test:", len(train_df.columns))

    # Save IDs for all three sets before dropping them
    train_ids = train_df[["PTID", "EXAMDATE"]].copy()
    val_ids = val_df[["PTID", "EXAMDATE"]].copy()
    test_ids = test_df[["PTID", "EXAMDATE"]].copy()
    
    train_df = train_df.drop(columns=["PTID", "EXAMDATE"])
    val_df = val_df.drop(columns=["PTID", "EXAMDATE"])
    test_df = test_df.drop(columns=["PTID", "EXAMDATE"])

    cat_cols = train_df.select_dtypes(include=["object"]).columns
    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        
        # Fit on training data only
        train_df[col] = train_df[col].astype(str)
        le.fit(train_df[col])
        train_df[col] = le.transform(train_df[col])
        
        # Transform validation set (handle unseen categories)
        val_df[col] = val_df[col].astype(str)
        val_df[col] = val_df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
        val_df[col] = val_df[col].replace(-1, np.nan)
        
        # Transform test set (handle unseen categories)
        test_df[col] = test_df[col].astype(str)
        test_df[col] = test_df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
        test_df[col] = test_df[col].replace(-1, np.nan)
        
        label_encoders[col] = le

    print("\n=== PASO 1: Imputación Iterativa (Random Forest) ===")
    print("Este paso puede tardar varios minutos...")
    imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=80, max_depth=10, n_jobs=-1), max_iter=5, random_state=42)
    
    # Fit imputer ONLY on training data
    train_df[:] = imputer.fit_transform(train_df)
    print("-> Imputación en Train completada.")
    
    # Transform validation and test using the fitted imputer
    val_df[:] = imputer.transform(val_df)
    print("-> Imputación en Validation completada.")
    
    test_df[:] = imputer.transform(test_df)
    print("-> Imputación en Test completada.")

    print("\nMissing Data después de imputar:")
    print(f"  Train: {train_df.isna().sum().sum()}")
    print(f"  Validation: {val_df.isna().sum().sum()}")
    print(f"  Test: {test_df.isna().sum().sum()}")

    print("\n=== PASO 2: Feature Engineering (reducir) ===")
    print("Aplicando transformaciones de atributos a cada conjunto por separado...")
    
    # Apply feature engineering to each set independently
    train_df = reducir(train_df, df_st_regions, df_volumen, df_baipetnmrc, df_otras)
    print("-> Feature engineering en Train completado.")
    
    val_df = reducir(val_df, df_st_regions, df_volumen, df_baipetnmrc, df_otras)
    print("-> Feature engineering en Validation completado.")
    
    test_df = reducir(test_df, df_st_regions, df_volumen, df_baipetnmrc, df_otras)
    print("-> Feature engineering en Test completado.")

    # Re-insert IDs after all transformations
    train_df.insert(0, "PTID", train_ids["PTID"])
    train_df.insert(1, "EXAMDATE", train_ids["EXAMDATE"])
    
    val_df.insert(0, "PTID", val_ids["PTID"])
    val_df.insert(1, "EXAMDATE", val_ids["EXAMDATE"])
    
    test_df.insert(0, "PTID", test_ids["PTID"])
    test_df.insert(1, "EXAMDATE", test_ids["EXAMDATE"])

    print("\n=== Guardando Datasets ===")
    # Save output files in the same directory as the script
    train_df.to_csv(script_dir / "TADPOLE_D1_D2_BL_RF_TRAIN.csv", index=False)
    val_df.to_csv(script_dir / "TADPOLE_D1_D2_BL_RF_VALIDATION.csv", index=False)
    test_df.to_csv(script_dir / "TADPOLE_D1_D2_BL_RF_TEST.csv", index=False)
    
    print(f"\n Train dataset guardado: {len(train_df)} filas, {len(train_df.columns)} columnas")
    print(f"  Ubicación: {script_dir / 'TADPOLE_D1_D2_BL_RF_TRAIN.csv'}")
    print(f" Validation dataset guardado: {len(val_df)} filas, {len(val_df.columns)} columnas")
    print(f"  Ubicación: {script_dir / 'TADPOLE_D1_D2_BL_RF_VALIDATION.csv'}")
    print(f" Test dataset guardado: {len(test_df)} filas, {len(test_df.columns)} columnas")
    print(f"  Ubicación: {script_dir / 'TADPOLE_D1_D2_BL_RF_TEST.csv'}")
    print("\nPipeline completado exitosamente!")