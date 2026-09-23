# Machine Learning Homework 2026

**Authors:** Giacomo Pogliana (ID: 308411), Lorenzo Ponzone (ID: 301776)  
**Institution:** Politecnico di Milano — Machine Learning Course (A.Y. 2025/2026)

---

## Project Overview
Machine learning project on the *Online Shoppers Purchasing Intention* dataset: exploratory data analysis, regression for missing `ExitRates` imputation, purchase intention classification under class imbalance, and comparison with clustering-based methods.

---

## Description
This project focuses on analyzing online browsing behavior and session metrics to predict whether an online shopper intends to make a purchase (`Revenue`). The dataset, sourced from the UCI Machine Learning Repository ("Online Shoppers Purchasing Intention Dataset"), contains browsing sessions from different users collected over a 1-year period to avoid campaign and seasonal bias.

The goal is to explore browsing dynamics, recover corrupted feature data (`ExitRates`) using regression, build classification models to predict purchase intention, and compare them with clustering-based methods.

## Dataset
The dataset consists of 12,330 sessions across training and test splits and features a binary classification task:

* **Revenue classification** – Classifying whether a session results in an online purchase (`True` / `False`), characterized by a strong class imbalance (~15.5% purchases vs. ~84.5% non-purchases).

**Features & Structure:**
* 18 attributes per sample: 10 numerical features, 7 categorical features, and 1 binary target (`Revenue`).
* **Training set:** 9,248 samples (`training_set_online_shoppers_intention.csv`).
* **Test set:** 3,082 samples (`test_set_online_shoppers_intention.csv`).
* **Missing Data:** The `ExitRates` variable is corrupted and missing in ~30% of samples in both the training set (2,782 samples) and the test set (917 samples).

## Tasks

### 1. Data Analysis & Clustering
* Visualizing feature distributions and analyzing class imbalance.
* Identifying feature correlations (e.g., a strong 0.91 correlation between `BounceRates` and `ExitRates`, and `PageValues` as the strongest positive predictor of `Revenue` at 0.49).
* Applying and evaluating clustering methods to identify user segments and comparing their performance against supervised classification models.

### 2. Regression for Missing Data Recovery
* Training a regressor to predict missing `ExitRates` values using the remaining non-target features on complete training rows (6,466 samples).
* Comparing regression algorithms (Linear Regression, Ridge, Lasso) evaluated via 5-fold cross-validation using the $R^2$ metric.
* Integrating feature scaling within cross-validation pipelines to prevent data leakage.

### 3. Classification & Impact of Missing Data Recovery
* Imputing missing `ExitRates` values across both training and test sets using the best-performing regression model.
* Applying feature preprocessing (one-hot encoding for nominal categories, standardization for continuous variables) and feature selection.
* Training and comparing classification algorithms to predict `Revenue`.
* Evaluating the impact of missing data recovery by assessing whether a model built with recovered `ExitRates` outperforms a model trained without it.
* Testing the final classification model on the provided independent test set.

## Implementation
* **Language:** Python
* **Libraries Used:** `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`
* **Notebook:** The entire pipeline, analysis, and models are implemented in the Jupyter Notebook `MachineLearning_Project.ipynb`.

## Results
The project evaluates regression models using 5-fold cross-validation $R^2$ scores (~0.86), successfully imputing missing `ExitRates` values. Classification and clustering approaches are evaluated using accuracy, precision, recall, F1-score, and ROC-AUC metrics to address class imbalance. Final classification performance is assessed both before and after recovering missing `ExitRates`, as well as against unsupervised clustering benchmarks.

## Usage
To reproduce the results:

1. Install dependencies:
```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn notebook
```

2. Run the Jupyter Notebook:
```bash
jupyter notebook MachineLearning_Project.ipynb
```
