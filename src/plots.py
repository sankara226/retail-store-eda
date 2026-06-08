import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_all_plots(df: pd.DataFrame, top_performers: dict, output_dir: Path):
    """Generates and silent-saves analytical visual distribution plots onto disk."""
    output_dir.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Product Performance Barplot
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_performers['Product'].values, y=top_performers['Product'].index, palette="viridis")
    plt.title("Top Products by Revenue")
    plt.tight_layout()
    plt.savefig(output_dir / "top_products.png")
    plt.close()

    # 2. Regional Outlier Boxplot
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Region', y='ComputedRevenue', data=df, palette="Set2")
    plt.title('Revenue Distribution By Region')
    plt.tight_layout()
    plt.savefig(output_dir / "revenue_distribution_region.png")
    plt.close()

    # 3. Structural Entity Heatmap
    plt.figure(figsize=(10, 6))
    pivot = df.pivot_table(values='ComputedRevenue', index='Region', columns='Product', aggfunc='sum')
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap='coolwarm')
    plt.title('Revenue by Region and Product')
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_region_product.png")
    plt.close()

    # 4. Global Structural Pairplot
    pp = sns.pairplot(df[['UnitPrice', 'Quantity', 'ComputedRevenue']])
    pp.savefig(output_dir / "pairplot_metrics.png")
    plt.close()
