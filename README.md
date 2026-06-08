# Sales EDA + Baseline Models

Lightweight analysis pipeline that loads sales data, computes KPIs, generates plots, and runs simple baseline models (classification, regression, and Prophet forecast).

**Quick start — macOS / zsh**
- **Create & activate venv**: (done in this repo as `.venv`) 
```bash
python3 -m venv .venv
source .venv/bin/activate
```
- **Install dependencies**:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
- **Run the pipeline**:
```bash
python main.py
```

**Files of interest**
- `main.py`: pipeline entrypoint — ingestion, features, analytics, plots, and models.
- `requirements.txt`: project dependencies.
- `src/`: project modules (`data_loader.py`, `features.py`, `analytics.py`, `plots.py`, `models.py`).
- `data/`: example dataset `product_sales_region.csv` (ignored by default; remove from `.gitignore` to track real data).
- `.venv/`: local virtual environment (ignored).

**Notes & tips**
- Make sure to activate the repository `.venv` before running. If you see `ModuleNotFoundError`, run the Activate + Install steps above.
- The repository contains a small example CSV used for demonstration; real datasets should be kept out of the repo and stored securely.
- Plots are written to `outputs/` (also ignored by default).

**Publish to GitHub**
- Create a new repository on GitHub (no README). Then push locally:
```bash
git remote add origin https://github.com/<your-username>/<repo>.git
git branch -M main
git push -u origin main
```
- Or use the GitHub CLI:
```bash
brew install gh
gh auth login
gh repo create <your-username>/<repo> --public --source=. --remote=origin --push
```


