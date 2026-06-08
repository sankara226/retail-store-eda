import sys
import os
from pathlib import Path

# Helper to try to re-run the script with the repo local venv Python
def _try_rerun_with_venv():
    # Avoid infinite re-exec loop
    if os.environ.get("SALES_EDA_REEXECED") == "1":
        return False

    repo_root = Path(__file__).resolve().parent
    venv_py = repo_root / ".venv" / "bin" / "python"
    if venv_py.exists():
        print("\nDetected local .venv. Re-launching script with .venv Python to satisfy dependencies...")
        # Set flag so the child process won't re-exec again
        os.environ["SALES_EDA_REEXECED"] = "1"
        os.execv(str(venv_py), [str(venv_py)] + sys.argv)
        return True
    return False

try:
    from src.data_loader import load_clean_data
    from src.features import engineering_features
    from src.analytics import compute_top_performers, get_dashboard_kpis
    from src.plots import generate_all_plots
    from src.models import train_return_classifier, train_revenue_regressor, run_prophet_forecast
except Exception as e:
    # If a local .venv is present, attempt to re-run the script under it.
    if _try_rerun_with_venv():
        # execv replaces the current process; this line is normally not reached.
        pass

    print("\nERROR: Import failed — missing dependencies or virtualenv not activated.")
    print(f"Detail: {e}\n")
    print("Quick fixes:")
    print("  1) Activate the project's virtual environment:")
    print("     source .venv/bin/activate")
    print("  2) Install dependencies (if not already installed):")
    print("     pip install -r requirements.txt")
    print("  3) Then re-run the script:")
    print("     python main.py\n")
    sys.exit(1)

def main():
    # Absolute dynamic root routing
    ROOT_DIR = Path(__file__).resolve().parent
    DATA_PATH = ROOT_DIR / "data" / "product_sales_region.csv"
    OUTPUT_PATH = ROOT_DIR / "outputs"
    
    print("=== STARTING PIPELINE: E-COMMERCE INDUSTRIAL ARCHITECTURE ===")
    
    # Step 1: Ingestion & Transmutation
    df_raw = load_clean_data(DATA_PATH)
    df_transformed = engineering_features(df_raw)
    print(" Layer 1: Data normalized, cleaned and features mapped.")
    
    # Step 2: Aggregated Statistical Slicing
    tops = compute_top_performers(df_transformed)
    kpis = get_dashboard_kpis(df_transformed)
    
    print("\n" + "#"*40)
    print("         EXECUTIVE KPIS DASHBOARD")
    print("#"*40)
    print(" ")
    print(f" Total Realized Revenue : ${kpis['total_revenue']:,.2f}")
    print(f" Average Order Value   : ${kpis['aov']:,.2f}")
    print(f" Operational Return Rate: {kpis['return_rate']:.2f}%")
    print(" ")
    print("#"*40)
    
    # Step 3: Graphical Exportation
    generate_all_plots(df_transformed, tops, OUTPUT_PATH)
    print(f"\n Layer 2: Vector graphics compiled and saved to: {OUTPUT_PATH}")
    
    # Step 4: Machine Learning Executions
    print("\n Layer 3: Initializing predictive modules...")
    train_return_classifier(df_transformed)
    train_revenue_regressor(df_transformed)
    run_prophet_forecast(df_transformed)
    
    print("\n=== PIPELINE EXECUTION COMPLETELY SUCCESSFUL ===")

if __name__ == "__main__":
    main()