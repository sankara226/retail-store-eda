import pandas as pd
from pathlib import Path

def load_clean_data(file_path: Path) -> pd.DataFrame:
    """Loads dataset and strips localized formatting from numeric string fields."""
    if not file_path.exists():
        raise FileNotFoundError(f"Target dataset missing at specified path: {file_path}")
        
    df = pd.read_csv(file_path, sep=',', encoding='utf-8-sig')
    
    # Standardize float strings (comma to dot notation)
    cols_to_float = ['UnitPrice', 'TotalPrice', 'ShippingCost', 'Discount']
    for col in cols_to_float:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
            
    # Standardize temporal elements
    date_cols = ['Date', 'OrderDate', 'DeliveryDate']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')
            
    return df
