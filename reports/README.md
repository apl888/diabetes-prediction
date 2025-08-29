# Reports

This directory contains written reports that accompany the notebooks in the repository.  
The reports summarize exploratory data analysis (EDA), preprocessing steps, modeling, and evaluation results.  

## Files 
- `Pima Indian Diabetes Dataset Report.pdf` – Read-only PDF version of the report (recommended for quick viewing).  

## Connection to Notebooks
- [EDA notebook](..notebooks/1_eda.ipynb) → EDA section of the report.  
- [Preprocessing notebook](..notebooks/2_preprocessing.ipynb) → Preprocessing and feature engineering section.  
- [XGBoost modeling notebook](..notebooks/3_XGBoost_model.ipynb) → Modeling and evaluation section.  

## Usage
- Open the PDF report for a summary of the project, including figures and interpretations.  
- Use the notebooks if you want to reproduce the analysis or extend the workflow.  

## Note
- Multiple models were tested, optimized, and compared, including:
    - Logistic Regression
    - KNN
    - Random Forest
    - XGBoost
    - ANN (MLP)
- Comparative metrics for these models are provided in the report, forming the basis for selecting **XGBoost** as the best-performing model for this dataset.    
- Only the notebook for **XGBoost** is included in this repository.    
