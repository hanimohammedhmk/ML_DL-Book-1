# %% [markdown]
"""
Chapter 4 — Section 4.9
Full Mini-Project Walkthrough — Predict House Prices

This file is organized as a runnable Jupyter-style script (compatible with VS Code/Notebook cell markers).
Each markdown block explains the next code cell. The notebook is self-contained: it attempts to load
"synthetic_house_prices.csv" from the working directory and falls back to an internally generated
synthetic dataset if the file is not present.

Instructions:
- Open this file in JupyterLab, Jupyter Notebook, or VS Code. Run cells sequentially.
- Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib.

"""

# %% [markdown]
"""
## 4.9.1 Required libraries

Import the standard Python libraries used in the walkthrough. If you are missing any package,
install it using `pip install pandas numpy matplotlib seaborn scikit-learn joblib`.
"""

# %%
# Standard libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn utilities
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional: save the trained pipeline
import joblib

# Plot configuration for readable output in notebooks
%matplotlib inline
plt.rcParams.update({"figure.figsize": (8, 5), "font.size": 11})
sns.set_style("whitegrid")

# %% [markdown]
"""
## 4.9.2 Load the data

The code below attempts to read `synthetic_house_prices.csv` from the current working directory.
If the file is not present (for readers running the notebook without the book resources), the
cell creates a synthetic dataset with the same schema so the remainder of the notebook runs.
"""

# %%
# Path to dataset supplied with the book
csv_path = "synthetic_house_prices.csv"

try:
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset from {csv_path}")
except FileNotFoundError:
    # Fallback: create a synthetic dataset for demonstration
    print(f"{csv_path} not found. Creating a synthetic dataset for demonstration.")
    rng = np.random.default_rng(seed=42)
    n = 500
    area = rng.uniform(40, 250, size=n).round(1)
    rooms = rng.integers(1, 6, size=n)
    age = rng.integers(0, 51, size=n)
    location = rng.choice(["Downtown", "Suburb", "Countryside", "Seaside"], size=n, p=[0.3, 0.4, 0.2, 0.1])

    # Simple generative formula + noise to produce a realistic-seeming price
    base_price = 1500 * area
    room_premium = rooms * 10000
    age_discount = -300 * age
    location_factor = np.select(
        [location == 'Downtown', location == 'Suburb', location == 'Countryside', location == 'Seaside'],
        [1.25, 1.0, 0.85, 1.15]
    )
    noise = rng.normal(0, 20000, size=n)

    price = (base_price + room_premium + age_discount) * location_factor + noise
    price = np.maximum(price, 10000).round(0)

    df = pd.DataFrame({
        'area': area,
        'rooms': rooms,
        'age': age,
        'location': location,
        'price': price
    })

# Quick summary
print("Dataset shape:", df.shape)

# Display first rows (in a notebook this cell's output will render the table)
df.head()

# %% [markdown]
"""
## 4.9.3 Basic exploration

Inspect the first rows, data types, and descriptive statistics. Checking for missing values and
valid ranges is an essential early step.
"""

# %%
print("First five rows:")
display(df.head())

print("\nData types and missing values:")
display(df.info())

print("\nDescriptive statistics (numerical columns):")
display(df.describe().T)

# %% [markdown]
"""
## 4.9.4 Visual exploration

Visual checks help reveal relationships between predictors and the target. The correlation matrix
and plots below are useful initial diagnostics.
"""

# %%
# Correlation heatmap for numeric features
numeric_cols = ['area', 'rooms', 'age', 'price']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation matrix (numeric features)")
plt.show()

# Scatter: price vs area, colored by location
sns.scatterplot(data=df, x='area', y='price', hue='location', alpha=0.7)
plt.title("Price vs Area (colored by location)")
plt.show()

# Boxplot: price distribution by location
sns.boxplot(data=df, x='location', y='price')
plt.title("Price distribution by location")
plt.show()

# %% [markdown]
"""
## 4.9.5 Preprocessing: encoding and pipeline

We will use one-hot encoding for the `location` column and keep numeric columns as-is.
A `Pipeline` keeps preprocessing and the model together which reduces the risk of leakage and
simplifies saving/loading the full transformation chain.
"""

# %%
# Features and target
X = df.drop(columns=['price'])
y = df['price']

# Identify feature types
categorical_features = ['location']
numeric_features = ['area', 'rooms', 'age']

# ColumnTransformer: one-hot encode location, passthrough numeric columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

# Full pipeline: preprocessing followed by linear regression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# The train/test split is performed next; do not fit the pipeline before splitting.

# %% [markdown]
"""
## 4.9.6 Train / test split (best practice)

Always split the data before fitting the model. This prevents information from the test set
leaking into model training and provides a realistic estimate of generalization performance.
"""

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set:", X_train.shape, "Test set:", X_test.shape)

# %% [markdown]
"""
## 4.9.7 Train the model

Fit the pipeline on the training set. After fitting, extract coefficients and the intercept to
interpret how the model uses features to predict price.
"""

# %%
pipeline.fit(X_train, y_train)

# Extract the trained linear regression model
trained_model = pipeline.named_steps['model']

# Get one-hot encoder inside the preprocessor to build feature names
ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
# get_feature_names_out is available in recent sklearn versions; fall back if needed
try:
    ohe_feature_names = ohe.get_feature_names_out(categorical_features)
except Exception:
    ohe_feature_names = ohe.get_feature_names(categorical_features)

feature_names = list(ohe_feature_names) + numeric_features

coefficients = pd.Series(trained_model.coef_, index=feature_names)
intercept = trained_model.intercept_

print("Model intercept:", intercept)
print("\nModel coefficients:")
display(coefficients.sort_values(ascending=False))

# %% [markdown]
"""
## 4.9.8 Predictions and evaluation

Evaluate model performance using MAE, MSE, RMSE, and R². Compare training and test results to spot
underfitting or overfitting.
"""

# %%
# Predictions
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# Metrics helper
def regression_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

train_metrics = regression_metrics(y_train, y_train_pred)
test_metrics = regression_metrics(y_test, y_test_pred)

print("Training set metrics:")
for k, v in train_metrics.items():
    print(f"  {k}: {v:,.2f}")

print("\nTest set metrics:")
for k, v in test_metrics.items():
    print(f"  {k}: {v:,.2f}")

# %% [markdown]
"""
## 4.9.9 Visualization: Actual vs Predicted

A scatter plot of actual vs predicted values provides a quick qualitative assessment of fit.
The red dashed diagonal indicates perfect predictions.
"""

# %%
plt.figure(figsize=(7,7))
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Test Set)")
plt.show()

# %% [markdown]
"""
## 4.9.10 Residual analysis

Residual plots help detect bias and heteroscedasticity. Residuals should ideally be centered
around zero with no clear pattern.
"""

# %%
residuals = y_test - y_test_pred

# Residuals vs predicted
plt.figure()
sns.scatterplot(x=y_test_pred, y=residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals vs Predicted Price (Test Set)")
plt.show()

# Histogram of residuals
plt.figure()
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Residual")
plt.title("Distribution of Residuals (Test Set)")
plt.show()

# %% [markdown]
"""
## 4.9.11 Interpret coefficients

Interpret the learned coefficients to understand feature influence. For one-hot encoded location
variables, coefficients are relative to the dropped reference category.
"""

# %%
coefficients_abs_sorted = coefficients.abs().sort_values(ascending=False)
display(coefficients_abs_sorted)

print("\nTop contributing features (by absolute coefficient):")
display(coefficients.loc[coefficients_abs_sorted.index].head(10))

# %% [markdown]
"""
## 4.9.12 Save the trained pipeline (optional)

Saving the pipeline preserves both preprocessing and model steps so that the same transformation
is applied at inference time. This is critical for reproducible predictions and deployment.
"""

# %%
model_path = "house_price_pipeline.joblib"
joblib.dump(pipeline, model_path)
print(f"Saved trained pipeline to {model_path}")

# %% [markdown]
"""
## 4.9.13 Next steps and suggested exercises

1. Feature engineering: create features such as `rooms_per_area`, `age_bucket`, or interaction terms.
2. Model comparison: evaluate Ridge, Lasso, DecisionTreeRegressor, and RandomForestRegressor.
3. Cross-validation: use `cross_val_score` or `GridSearchCV` for robust performance estimation and
   hyperparameter tuning.
4. Target transformation: if residuals increase with price, consider `log(price)` as the target.
5. Deploy: create a simple Streamlit app that loads the saved pipeline and exposes a web-based
   input form for predictions.


This notebook is ready to run and is organized to teach the complete end-to-end regression
workflow for the synthetic house price dataset included with the book.
"""
