# Automated Data Profiling & Model Recommendation Engine

You upload a CSV. The engine does the rest.

It scans your dataset for problems, explains what it found, visualises the data, and tells you exactly which machine learning model to start with — all before you write a single line of model code.

![Python](https://img.shields.io/badge/Python-3.9+-black?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-black?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-black?style=flat-square)

---

## Why I built this

Every ML project starts the same way — you get a dataset, and before you can do anything interesting, you spend hours just understanding it. Checking for missing values, spotting weird outliers, figuring out what kind of problem you're solving. It's repetitive and tedious.

This tool automates that entire first phase. Drop in any CSV and within seconds you have a complete picture of your data — what's broken, what's suspicious, how things relate to each other, and where to go next.

---

## What it does, and how

### 1 · Dataset Upload & Column Summary

The moment you upload a CSV, the engine reads it into a Pandas DataFrame and immediately shows you:
- A preview of the first 5 rows
- Total row and column count
- A column summary table — the data type, number of unique values, and a sample value for every single column

This gives you an instant structural overview before any analysis begins.

---

### 2 · Data Integrity

This is where the engine starts finding real problems.

**Duplicate rows** — Uses `df.duplicated().sum()` to count rows that are exact copies. Duplicates silently inflate your training data and give models a false sense of confidence.

**Missing values** — Uses `df.isnull().sum()` to count gaps per column. But it doesn't just count — it tells you what to do about them:

| Situation | Suggested Strategy | Why |
|---|---|---|
| More than 50% missing | Drop the column | Too unreliable to fill |
| Categorical column | Fill with Mode | Use the most frequent category |
| Numeric, skewed data | Fill with Median | Median ignores extreme values |
| Numeric, normal data | Fill with Mean | Mean is accurate when data is balanced |

The skewness check uses `df[col].skew()` — if the skew is above 1 or below -1, the distribution is lopsided and the median is safer than the mean.

**Outlier detection** — Uses the **3-sigma rule**: any value that sits more than 3 standard deviations away from the mean is flagged as an outlier. For each affected column, the engine tells you:
- How many outliers exist
- What percentage of the dataset they represent
- What impact they'll have on your model

| Outlier % | Impact |
|---|---|
| > 5% | High — will skew predictions, distort the mean |
| 1–5% | Moderate — linear models suffer, tree models handle it better |
| < 1% | Low — worth reviewing but unlikely to cause major issues |

**Descriptive statistics** — Full `df.describe()` output: count, mean, standard deviation, min, max, and the 25th / 50th / 75th percentile for every numeric column.

---

### 3 · Distribution & Correlation Analysis

**Histogram** — Select any numeric column and see its full distribution rendered as an interactive Plotly chart. This tells you whether the data is normally distributed, skewed, or has multiple peaks — all of which affect how you should preprocess it.

**Correlation heatmap** — Uses `df.corr()` to compute the correlation coefficient between every pair of numeric columns, then renders it as a heatmap with `px.imshow()`. Values close to 1 or -1 mean two features move together — useful for spotting redundant features or strong predictors before training.

---

### 4 · Model Recommendation

The engine inspects the column you choose as your target variable (Y) and automatically determines the problem type:

- **Classification** — if the target is non-numeric, or has fewer than 10 unique values
- **Regression** — if the target is numeric with 10 or more unique values

Based on that, it recommends three models ranked from simple to powerful:

**Classification:** Logistic Regression → Random Forest Classifier → XGBoost Classifier

**Regression:** Linear Regression → Random Forest Regressor → XGBoost Regressor

It also suggests a train/test split ratio based on dataset size:

| Dataset Size | Split | Reason |
|---|---|---|
| Under 500 rows | 60 / 40 | Small data — need more for reliable testing |
| 500 – 5000 rows | 70 / 30 | Standard balanced split |
| 5000+ rows | 80 / 20 | Large enough that 20% test set is still substantial |

---

## Project Structure

```
├── app.py           # Streamlit UI — handles all display and user interaction
├── data_engine.py   # Pure Python backend — all analysis logic lives here
```

The two files are intentionally separated. `data_engine.py` has no UI dependency — every function takes a DataFrame and returns a result. This means the logic is independently testable and reusable outside of Streamlit.

---

## Run Locally

```bash
git clone https://github.com/your-username/data-profiling-engine.git
cd data-profiling-engine
pip install streamlit pandas numpy plotly
streamlit run app.py
```

---

## Author

**Ankita Dhara** · [LinkedIn]([https://linkedin.com](https://www.linkedin.com/in/ankita-dhara-7333a2299/)) · [GitHub]((https://github.com/ankitadhara28))
