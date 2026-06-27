# data_engine.py
import pandas as pd
import numpy as np

def get_data_health(df):
    """Calculates missing values and duplicates."""
    null_counts = df.isnull().sum()
    missing_data = null_counts[null_counts > 0]
    duplicate_count = df.duplicated().sum()
    return missing_data, duplicate_count

def get_statistical_summary(df):
    """Returns the df.describe() dataframe."""
    return df.describe()

def get_column_summary(df):
    """Returns a summary table of each column: dtype, unique values, and a sample value."""
    summary = pd.DataFrame({
        "Data Type": df.dtypes,
        "Unique Values": df.nunique(),
        "Sample Value": df.iloc[0]
    })
    return summary

def suggest_missing_value_handling(df):
    """
    For each column with missing values, suggests whether to use
    mean, median, mode, or drop — based on data type and skewness.
    """
    suggestions = []
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing == 0:
            continue

        missing_pct = round((missing / len(df)) * 100, 1)

        if missing_pct > 50:
            strategy = "Drop column"
            reason = "More than 50% data is missing — not reliable to fill."
        elif df[col].dtype == "object":
            strategy = "Fill with Mode"
            reason = "Categorical column — use the most frequent value."
        else:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                strategy = "Fill with Median"
                reason = f"Skewed distribution (skew={skewness:.2f}) — median is more robust than mean."
            else:
                strategy = "Fill with Mean"
                reason = f"Roughly normal distribution (skew={skewness:.2f}) — mean is appropriate."

        suggestions.append({
            "Column": col,
            "Missing Count": missing,
            "Missing %": f"{missing_pct}%",
            "Suggested Strategy": strategy,
            "Reason": reason
        })

    return pd.DataFrame(suggestions).set_index("Column")


def get_outliers(df):
    """
    Detects outliers using the 3σ rule and explains what impact
    they have on machine learning models.
    """
    numeric_df = df.select_dtypes(include="number")
    results = []

    for col in numeric_df.columns:
        mean = numeric_df[col].mean()
        std = numeric_df[col].std()
        outlier_rows = numeric_df[
            (numeric_df[col] < mean - 3 * std) | (numeric_df[col] > mean + 3 * std)
        ]
        count = len(outlier_rows)
        if count > 0:
            pct = round((count / len(df)) * 100, 1)
            # Explain impact based on how many outliers exist
            if pct > 5:
                impact = "High impact — can heavily skew model predictions and distort the mean."
            elif pct > 1:
                impact = "Moderate impact — may affect linear models; tree-based models handle better."
            else:
                impact = "Low impact — small number, but worth reviewing before training."

            results.append({
                "Column": col,
                "Outlier Count": count,
                "Outlier %": f"{pct}%",
                "Impact on Model": impact
            })

    return pd.DataFrame(results).set_index("Column")

def get_correlation_matrix(df):
    """Returns the correlation matrix for numeric columns."""
    return df.select_dtypes(include="number").corr()

def suggest_train_test_split(df):
    """Suggests a train/test split ratio based on dataset size."""
    rows = len(df)
    if rows < 500:
        split = "60 / 40"
        reason = "Small dataset — more data kept for testing to get reliable evaluation."
    elif rows < 5000:
        split = "70 / 30"
        reason = "Medium dataset — balanced split for training and evaluation."
    else:
        split = "80 / 20"
        reason = "Large dataset — more data given to training since test set is still large enough."
    return split, reason

def suggest_models_logic(df, target_column):
    """Analyzes the target column and suggests machine learning models."""
    unique_values = df[target_column].nunique()
    is_numeric = np.issubdtype(df[target_column].dtype, np.number)

    if not is_numeric or unique_values < 10:
        problem_type = "Classification"
        models = [
            "Logistic Regression (Great for simple, linear baselines)",
            "Random Forest Classifier (Excellent for handling non-linear data)",
            "XGBoost Classifier (Industry standard for maximum accuracy)"
        ]
    else:
        problem_type = "Regression"
        models = [
            "Linear Regression (Standard baseline model)",
            "Random Forest Regressor (Robust to outliers and highly flexible)",
            "XGBoost Regressor (Top performer for complex numerical prediction)"
        ]

    return problem_type, models