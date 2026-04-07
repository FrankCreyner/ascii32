import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

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

    df = pd.read_csv("TADPOLE_D1_D2.csv", low_memory=False, na_values=["", " ", "-4", -4, "-1", -1, "NA"])

    df_baipetnmrc = pd.read_csv("BAIPETNMRC.csv", low_memory=False)
    df_volumen = pd.read_csv("volumen.csv")
    df_st_regions = pd.read_csv("st_regions.csv")
    df_otras = pd.read_csv("otras.csv")
    
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
        "ADAS11"
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

    df_bl = reducir(df_bl, df_st_regions, df_volumen, df_baipetnmrc, df_otras)

    if "DX" in df_bl.columns:
        df_bl = df_bl.dropna(subset=["DX"]).reset_index(drop=True)
        # ----------------------------------------------------------------------------------
        # SOLUCIÓN DE ERROR: Eliminar clases con muy pocos miembros (ej. "Dementia to MCI")
        # Esto previene el error 'The least populated classes in y have only 1 member'
        # al usar stratify. 
        # ----------------------------------------------------------------------------------
        class_counts = df_bl["DX"].value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        
        # Filtramos para quedarnos solo con clases aptas para entrenamiento
        df_bl = df_bl[df_bl["DX"].isin(valid_classes)].reset_index(drop=True)
        train_df, test_df = train_test_split(df_bl, test_size=0.2, random_state=42, stratify=df_bl["DX"])
    else:
        train_df, test_df = train_test_split(df_bl, test_size=0.2, random_state=42)
    
    train_df = train_df.copy()
    test_df = test_df.copy()

    print("\n--- Analizando Data en TRAIN Dataset ---")
    columnas_muy_vacias = missing_data(train_df)
    columnas_constantes = constant_columns(train_df)
    columnas_duplicadas = duplicate_columns(train_df)

    columnas_eliminar = list(set(columnas_muy_vacias + columnas_constantes + columnas_duplicadas))
    print("Columnas totales a eliminar (Train):", len(columnas_eliminar))
    
    train_df.drop(columns=columnas_eliminar, inplace=True, errors="ignore")
    test_df.drop(columns=columnas_eliminar, inplace=True, errors="ignore")

    to_drop_corr = eliminar_correlacion_alta(train_df, threshold=0.85)
    train_df.drop(columns=to_drop_corr, inplace=True, errors="ignore")
    test_df.drop(columns=to_drop_corr, inplace=True, errors="ignore")

    print("Columnas restantes en Train/Test:", len(train_df.columns))

    train_ids = train_df[["PTID", "EXAMDATE"]].copy()
    test_ids = test_df[["PTID", "EXAMDATE"]].copy()
    
    train_df = train_df.drop(columns=["PTID", "EXAMDATE"])
    test_df = test_df.drop(columns=["PTID", "EXAMDATE"])

    cat_cols = train_df.select_dtypes(include=["object"]).columns
    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        
        train_df[col] = train_df[col].astype(str)
        le.fit(train_df[col])
        train_df[col] = le.transform(train_df[col])
        
        test_df[col] = test_df[col].astype(str)
        test_df[col] = test_df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
        test_df[col] = test_df[col].replace(-1, np.nan)
        
        label_encoders[col] = le

    print("\nIniciando Imputación Iterativa (Random Forest)... (Este paso puede tardar varios minutos)")
    imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=80, max_depth=10, n_jobs=-1), max_iter=5, random_state=42)
    
    train_df[:] = imputer.fit_transform(train_df)
    print("-> Imputación en Train completada.")
    
    test_df[:] = imputer.transform(test_df)
    print("-> Imputación en Test completada.")

    train_df.insert(0, "PTID", train_ids["PTID"])
    train_df.insert(1, "EXAMDATE", train_ids["EXAMDATE"])    
    
    test_df.insert(0, "PTID", test_ids["PTID"])
    test_df.insert(1, "EXAMDATE", test_ids["EXAMDATE"])

    print("\nMissing Data en Train después de imputar:", train_df.isna().sum().sum())
    print("Missing Data en Test después de imputar:", test_df.isna().sum().sum())

    train_df.to_csv("TADPOLE_D1_D2_BL_RF_TRAIN.csv", index=False)
    test_df.to_csv("TADPOLE_D1_D2_BL_RF_TEST.csv", index=False)
    print("\nSe han guardado exitosamente los datasets aislados para Train y Test.")