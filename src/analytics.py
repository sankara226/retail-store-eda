import pandas as pd

def compute_top_performers(df: pd.DataFrame) -> dict:
    """Aggregates revenue performance slices across corporate entities."""
    dimensions = ['Product', 'Region', 'Salesperson', 'StoreLocation', 'Month', 'Year']
    return {dim: df.groupby(dim)['ComputedRevenue'].sum().sort_values(ascending=False) for dim in dimensions}

def get_dashboard_kpis(df: pd.DataFrame) -> dict:
    """Calculates operational high-level executive performance indicators."""
    return {
        "total_revenue": df['ComputedRevenue'].sum(),
        "aov": df['ComputedRevenue'].mean(),
        "return_rate": df['IsReturned'].mean() * 100,
        "revenue_per_region": df.groupby('Region')['ComputedRevenue'].sum()
    }
