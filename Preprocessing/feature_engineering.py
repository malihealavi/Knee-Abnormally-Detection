import ast
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from configs import get_config
cfg = get_config()

def clean_and_encode_dicom_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans, aggregates, and One-Hot Encodes categorical DICOM metadata columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing columns: Manufacturer, Laterality, PatientSex,
        ScanningSequence, SequenceVariant, MRAcquisitionType
        
    Returns:
    --------
    df_encoded : pd.DataFrame
        DataFrame with original metadata columns cleaned and One-Hot Encoded.
    """
    df = df.copy()
    
    # -------------------------------------------------------------------------
    # 1. Clean & Aggregate Manufacturer
    # -------------------------------------------------------------------------

    df['Manufacturer_Clean'] = df['Manufacturer'].copy()

    # -------------------------------------------------------------------------
    # 2. Clean & Aggregate Laterality
    # -------------------------------------------------------------------------
    def clean_laterality(val):
        if pd.isna(val) or str(val).strip() == '':
            return 'Unknown'
        val = str(val).strip().upper()
        mapping = {
            'R': 'RIGHT',
            'RIGHT': 'RIGHT',
            'L': 'LEFT',
            'LEFT': 'LEFT',
            'B': 'BILATERAL',
            'BILATERAL': 'BILATERAL'
        }
        return mapping.get(val, 'Unknown')

    df['Laterality_Clean'] = df['Laterality'].apply(clean_laterality)

    # -------------------------------------------------------------------------
    # 3. Clean & Aggregate PatientSex
    # -------------------------------------------------------------------------
    def clean_patient_sex(val):
        if pd.isna(val):
            return 'Undefined'
        val = str(val).strip().upper()
        if val == 'M':
            return 'Male'
        elif val == 'F':
            return 'Female'
        return 'Undefined'

    df['PatientSex_Clean'] = df['PatientSex'].apply(clean_patient_sex)

    # -------------------------------------------------------------------------
    # 4. Clean & Aggregate ScanningSequence
    # -------------------------------------------------------------------------
    def parse_sequence_list(val):
        if pd.isna(val):
            return []
        val_str = str(val).strip()
        # Parse string representation of list (e.g. "['SE', 'IR']")
        if val_str.startswith('['):
            try:
                parsed = ast.literal_eval(val_str)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed]
            except (ValueError, SyntaxError):
                pass
        return [val_str] if val_str else []
    
    df['ScanningSequence_Clean'] = df['ScanningSequence'].apply(parse_sequence_list)
    # Binary column for each sequence type (1 if present in the list)
    for seq in ['SE', 'GR', 'IR', 'RM']:
        df[f'ScanningSequence_{seq}'] = df['ScanningSequence_Clean'].apply(
            lambda x: int(seq in x)
        ).astype('int8')
    df.drop(columns=['ScanningSequence_Clean'], inplace=True)

    # -------------------------------------------------------------------------
    # 5. Clean & Aggregate SequenceVariant
    # -------------------------------------------------------------------------
    def parse_variant_list(val):
        if pd.isna(val):
            return []
        val_str = str(val).strip()
        if val_str.startswith('['):
            try:
                parsed = ast.literal_eval(val_str)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip() != 'NONE']
            except (ValueError, SyntaxError):
                pass
        return [val_str] if val_str not in ('', 'NONE') else []

    df['SequenceVariant_Clean'] = df['SequenceVariant'].apply(parse_variant_list)
    # Presence flags for each variant type
    for var in ['SK', 'SP', 'OSP', 'SS']:
        df[f'SequenceVariant_{var}'] = df['SequenceVariant_Clean'].apply(
            lambda x: int(var in x)
        ).astype('int8')
    df.drop(columns=['SequenceVariant_Clean'], inplace=True)

    # -------------------------------------------------------------------------
    # 6. Clean & Aggregate MRAcquisitionType
    # -------------------------------------------------------------------------
    def clean_acq_type(val):
        if pd.isna(val) or str(val).strip() == '':
            return 'UNKNOWN'
        val_str = str(val).strip().upper()
        return val_str if val_str in ['2D', '3D'] else 'UNKNOWN'

    df['MRAcquisitionType_Clean'] = df['MRAcquisitionType'].apply(clean_acq_type)

    # -------------------------------------------------------------------------
    # 7. One-Hot Encoding
    # -------------------------------------------------------------------------
    clean_cols = [
        'Manufacturer_Clean',
        'Laterality_Clean',
        'PatientSex_Clean',
        # 'ScanningSequence_Clean',
        # 'SequenceVariant_Clean',
        'MRAcquisitionType_Clean'
    ]
    
    # Perform One-Hot Encoding and convert boolean outputs to integers (0 or 1)
    df_encoded = pd.get_dummies(df, columns=clean_cols, prefix_sep='_', dtype=int)
    
    # Drop original uncleaned raw columns
    raw_cols = ['Manufacturer', 'Laterality', 'PatientSex', 'ScanningSequence', 'SequenceVariant', 'MRAcquisitionType']
    df_encoded = df_encoded.drop(columns=[col for col in raw_cols if col in df_encoded.columns])

    return df_encoded

sample_df = cfg.data.LIGHT_DF[['Manufacturer',
                'Laterality',
                'PatientSex',
                 'ScanningSequence',
                'SequenceVariant',
                'MRAcquisitionType']].copy()
# Run cleaning and encoding
encoded_df = clean_and_encode_dicom_metadata(sample_df)
encoded_df['SliceNumberRate'] = cfg.glob.NUM_SLICES/cfg.data.LIGHT_DF['InstanceNumber']
encoded_df['InstanceNumber'] = cfg.data.LIGHT_DF['InstanceNumber']
encoded_df['Rows'] = np.log(cfg.data.LIGHT_DF['Rows'])
encoded_df['Columns'] = np.log(cfg.data.LIGHT_DF['Columns'])
encoded_df[[ 'MagneticFieldStrength','FlipAngle' , 'Fluid_Sensitive', 'Fat_Suppression'	]] = cfg.data.LIGHT_DF[[ 'MagneticFieldStrength','FlipAngle' , 'Fluid_Sensitive', 'Fat_Suppression'	]]
print("Encoded Feature Columns:")
print(encoded_df.columns.tolist())


num_cols = ['InstanceNumber', 'Rows', 'Columns', 'MagneticFieldStrength',
            'FlipAngle', 'SliceNumberRate']

std_scaler = StandardScaler()
encoded_df[num_cols] = std_scaler.fit_transform(encoded_df[num_cols])

# Impute AFTER scaling so missing becomes an out-of-range sentinel
encoded_df[num_cols] = encoded_df[num_cols].fillna(-1)
encoded_df [['StudyInstanceUID','SeriesInstanceUID','file_path', 'Anatomical_Plane']] = cfg.data.LIGHT_DF[['StudyInstanceUID','SeriesInstanceUID','file_path', 'Anatomical_Plane']]
LIGHT_DF = None
LIGHT_DF = encoded_df.copy()
LIGHT_DF.info()