# Data Directory

This folder contains datasets for the project.  

## Structure
- `raw/` → raw input datasets (ignored by Git).  
- `processed/` → preprocessed datasets (`X_train.pkl`, `y_train.pkl`, etc., ignored by Git).  

## Example files
- `data/raw/diabetes.csv`
- `data/processed/X_train.pkl`
- `data/processed/y_train.pkl`
- `data/processed/X_test.pkl`
- `data/processed/y_test.pkl`

Note: These files are **not included in the repository** (see `.gitignore`).  

## Usage
1. Ensure that `data/raw/` contains the raw dataset (e.g., `diabetes.csv`). 
2. Run the notebooks in order:
   - **`notebooks/1_eda.ipynb`** – Perform exploratory data analysis (EDA) on the raw dataset.
   - **OPTIONAL: `notebooks/2_preprocessing.ipynb`** – Preprocess the raw dataset (impute missing values, add interactions, bin features, encode, log-transform, scale). This will generate processed data in `data/processed/`.  **You do not need to run the preprocessing notebook, since notebook 3 includes preprocessing internally. Notebook 2 is provided to build and test the preprocessing pipeline separately. Run it only if you want to experiment with preprocessing steps (e.g., imputation, scaling) or save standalone transformed datasets.**
   - **`notebooks/3_XGBoost_model.ipynb`** – Train and evaluate an XGBoost classifier on the processed dataset.
3. (Optional) Extend the workflow by adding additional models or evaluation metrics in new notebooks.

