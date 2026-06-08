import pandas as pd

def engineering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies domain-driven functional engineering to incoming datasets."""
    df = df.copy()
    
    # Financial and operational KPI attributes
    df['NetUnitPrice'] = df['UnitPrice'] * (1 - df['Discount'])
    df['ComputedRevenue'] = df['NetUnitPrice'] * df['Quantity']
    df['DeliveryDelay'] = (df['DeliveryDate'] - df['OrderDate']).dt.days
    df['IsReturned'] = (df['Returned'] == 1).astype(int)
    df['HasPromotion'] = df['Promotion'].notna().astype(str)
    
    # Calendar sub-feature splitting
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    return df
