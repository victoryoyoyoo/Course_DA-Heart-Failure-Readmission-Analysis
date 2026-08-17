#%%
# --- Module 1: Data Ingestion & Quality Assurance ---
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

FILE_ID = '1bSa2A5xgYUv4QHC1U8zz0UF-lngOUvNi'
FILE_NAME = 'D01_heart_failure_readmission_dataset.csv'

import gdown
if not os.path.exists(FILE_NAME):
    print("Fetching dataset using Python...")
    gdown.download(id=FILE_ID, output=FILE_NAME, quiet=False)

try:
    df = pd.read_csv(FILE_NAME)
except Exception as e:
    raise RuntimeError(f"Data loading failed: {e}")

df = df.dropna(subset=['bnp', 'readmitted_30d'])

exclude_cols = ['patient_id', 'readmitted_30d', 'bnp']
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
# Bug fix: cat_cols previously kept patient_id (and any categorical target
# columns) instead of excluding them like num_cols does above. If patient_id
# is stored as a string ID, it was being one-hot encoded and fed into the
# models as a "feature" — a data leakage / meaningless-feature bug.
cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in exclude_cols]

# 1. Missingness Diagnostics
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
msno.matrix(df, sparkline=False, ax=axes[0], color=(0.15, 0.35, 0.55), fontsize=10)
axes[0].set_title("Missingness Matrix", fontweight='bold', fontsize=12)
msno.heatmap(df, ax=axes[1], fontsize=10, cmap="YlGnBu")
axes[1].set_title("Missingness Correlation", fontweight='bold', fontsize=12)
plt.tight_layout()
plt.show()

# 2. Robust Outlier Handling (IQR)
for col in num_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (df[col] < lower) | (df[col] > upper)
    
    if mask.sum() > 0:
        df.loc[mask, col] = np.nan

# 3. Clinical Imputation Strategy
for col in df.columns:
    if df[col].isnull().any():
        if col in num_cols:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

print(f"Pipeline Stage 1 complete. Data shape: {df.shape}")


# --- Module 2: Preprocessing Benchmark & Baseline Modeling ---
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, f1_score, roc_auc_score, confusion_matrix

target_reg, target_clf = 'bnp', 'readmitted_30d'
features = num_cols + cat_cols

# 1. Scaler Benchmark
scalers = {'Standard': StandardScaler(), 'MinMax': MinMaxScaler(), 'Robust': RobustScaler()}
scaler_scores = []

for name, scaler in scalers.items():
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', scaler)]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])
    
    cv_scores = []
    for i in range(3):
        X_tr, X_te, y_tr, y_te = train_test_split(df[features], df[target_clf], test_size=0.2, random_state=i*42)
        clf = Pipeline([('pre', preprocessor), ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=i*42))])
        clf.fit(X_tr, y_tr)
        cv_scores.append(f1_score(y_te, clf.predict(X_te), average='weighted'))
    
    scaler_scores.append({'Scaler': name, 'Weighted F1': np.mean(cv_scores)})

scaler_df = pd.DataFrame(scaler_scores)
best_scaler_name = scaler_df.loc[scaler_df['Weighted F1'].idxmax()]['Scaler']
best_scaler = scalers[best_scaler_name]

opt_preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', best_scaler)]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# 2. 10-Run Validation to ensure model stability
metrics = []
for i in range(10):
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(df[features], df[target_reg], test_size=0.2, random_state=i*42)
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(df[features], df[target_clf], test_size=0.2, random_state=i*42)

    reg_model = Pipeline([('pre', opt_preprocessor), ('svr', SVR(kernel='rbf', C=10))]).fit(Xr_tr, yr_tr)
    clf_model = Pipeline([('pre', opt_preprocessor), ('rf', RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=i*42))]).fit(Xc_tr, yc_tr)

    r_pred = reg_model.predict(Xr_te)
    c_pred = clf_model.predict(Xc_te)

    metrics.extend([
        {'Task': 'Regression', 'Metric': 'RMSE', 'Value': np.sqrt(mean_squared_error(yr_te, r_pred))},
        {'Task': 'Regression', 'Metric': 'R2', 'Value': r2_score(yr_te, r_pred)},
        {'Task': 'Classification', 'Metric': 'F1-Score', 'Value': f1_score(yc_te, c_pred, average='weighted')},
        {'Task': 'Classification', 'Metric': 'ROC-AUC', 'Value': roc_auc_score(yc_te, clf_model.predict_proba(Xc_te)[:, 1])}
    ])

res_df = pd.DataFrame(metrics)
print(res_df.groupby(['Task', 'Metric'])['Value'].agg(['mean', 'std']).round(4))

# 3. Model Analytics Dashboard
# NOTE: yr_te/r_pred/yc_te/c_pred are intentionally reused here from the
# LAST iteration (i=9) of the 10-run loop above — this plots one
# representative run, not an aggregate. Kept as-is since it's a deliberate
# choice, but flagging so it's not mistaken for an accident.
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].scatter(yr_te, r_pred, alpha=0.5, edgecolors='w', color='#2ca02c')
axes[0].plot([yr_te.min(), yr_te.max()], [yr_te.min(), yr_te.max()], 'r--', lw=2)
axes[0].set_title(f"SVR Prediction Parity (Target: {target_reg})", fontweight='bold')
axes[0].set_xlabel("Actual Values")
axes[0].set_ylabel("Predicted Values")

cm = confusion_matrix(yc_te, c_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
axes[1].set_title(f"RF Confusion Matrix (Target: {target_clf})", fontweight='bold')
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()


# --- Module 3: Advanced Feature Selection Strategies ---
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.svm import LinearSVR

X_raw = pd.get_dummies(df.drop(columns=[target_reg, target_clf, 'patient_id']), drop_first=True)
X_scaled = pd.DataFrame(best_scaler.fit_transform(SimpleImputer(strategy='median').fit_transform(X_raw)), columns=X_raw.columns)
y_reg, y_clf = df[target_reg], df[target_clf]

def evaluate_subset(X_sub, y, task='reg'):
    scores = []
    for i in range(5):
        X_tr, X_te, y_tr, y_te = train_test_split(X_sub, y, test_size=0.2, random_state=i*42)
        if task == 'reg':
            model = SVR(kernel='rbf', C=10).fit(X_tr, y_tr)
            scores.append(np.sqrt(mean_squared_error(y_te, model.predict(X_te))))
        else:
            model = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=i*42).fit(X_tr, y_tr)
            scores.append(f1_score(y_te, model.predict(X_te), average='weighted'))
    return np.mean(scores), np.std(scores)

K = 5
base_rmse, _ = evaluate_subset(X_scaled, y_reg, 'reg')
base_f1, _ = evaluate_subset(X_scaled, y_clf, 'clf')

class EmbeddedSelector:
    def __init__(self, estimator, k): self.estimator, self.k = estimator, k
    def fit(self, X, y):
        model = self.estimator.fit(X, y)
        coefs = np.abs(getattr(model, 'coef_', getattr(model, 'feature_importances_', None)))
        if coefs.ndim > 1: coefs = coefs[0]
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[np.argsort(coefs)[-self.k:]] = True
        return self
    def get_support(self): return self.support_

strategies = {
    'Filter: ANOVA F-Test': (SelectKBest(f_regression, k=K).fit(X_scaled, y_reg), SelectKBest(f_classif, k=K).fit(X_scaled, y_clf)),
    'Embedded: Lasso / LogReg': (EmbeddedSelector(LassoCV(cv=5), K).fit(X_scaled, y_reg), EmbeddedSelector(LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear'), K).fit(X_scaled, y_clf)),
    'Wrapper: RFE': (RFE(LinearSVR(random_state=42, dual='auto'), n_features_to_select=K).fit(X_scaled, y_reg), RFE(RandomForestClassifier(n_estimators=20), n_features_to_select=K).fit(X_scaled, y_clf))
}

results = []
for name, (sel_reg, sel_clf) in strategies.items():
    m_reg, s_reg = evaluate_subset(X_scaled.iloc[:, sel_reg.get_support()], y_reg, 'reg')
    m_clf, s_clf = evaluate_subset(X_scaled.iloc[:, sel_clf.get_support()], y_clf, 'clf')
    results.extend([
        {'Strategy': name, 'Task': 'Regression (RMSE)', 'Score': m_reg, 'Std': s_reg},
        {'Strategy': name, 'Task': 'Classification (F1)', 'Score': m_clf, 'Std': s_clf}
    ])

df_res = pd.DataFrame(results)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
for i, task in enumerate(['Regression (RMSE)', 'Classification (F1)']):
    ax = axes[i]
    data = df_res[df_res['Task'] == task].reset_index(drop=True)
    ax.barh(data['Strategy'], data['Score'], xerr=data['Std'], color=sns.color_palette('viridis', len(data)), alpha=0.85)
    
    base = base_rmse if 'Regression' in task else base_f1
    ax.axvline(base, color='#d9534f', linestyle='--', lw=2, label=f'Baseline ({base:.3f})')
    
    for j, (val, err) in enumerate(zip(data['Score'], data['Std'])):
        ax.annotate(f'{val:.3f}±{err:.3f}', (val + err + 0.01, j), va='center', fontsize=10)
        
    ax.set_title(f'Feature Selection Benchmark: {task}', fontweight='bold')
    ax.legend()

plt.tight_layout()
plt.show()


# --- Module 4: Dimensionality Reduction & Manifold Learning ---
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

tf.random.set_seed(42)
np.random.seed(42)

df['bnp_level'] = pd.qcut(df['bnp'], q=4, labels=[0, 1, 2, 3])
target_dr = 'bnp_level'
feature_cols = [c for c in df.columns if c not in ['bnp', 'readmitted_30d', 'bnp_level', 'patient_id']]

X_dr = best_scaler.fit_transform(ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
]).fit_transform(df[feature_cols]))
y_dr = df[target_dr]

# 1. PCA
X_pca = PCA(n_components=10, random_state=42).fit_transform(X_dr)

# 2. Deep Autoencoder
input_dim = X_dr.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dropout(0.2)(encoded)
encoded = Dense(10, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_dr, X_dr, epochs=100, batch_size=32, shuffle=True, verbose=0, validation_split=0.2)

encoder = Model(input_layer, encoded)
X_ae = encoder.predict(X_dr, verbose=0)

def test_extraction(X_data, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X_data, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42).fit(X_tr, y_tr)
    return f1_score(y_te, model.predict(X_te), average='weighted'), model.predict(X_te)

f1_full, _ = test_extraction(X_dr, y_dr)
f1_pca, _ = test_extraction(X_pca, y_dr)
f1_ae, preds_ae = test_extraction(X_ae, y_dr)

print(f"Full Feature Space F1: {f1_full:.4f}")
print(f"PCA (10 components) F1: {f1_pca:.4f}")
print(f"Deep Autoencoder F1: {f1_ae:.4f}")

# 3. Manifold Learning Visualization (t-SNE)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
methods = [(X_dr, 'Raw Features'), (X_pca, 'PCA (10)'), (X_ae, 'Deep Autoencoder')]

for ax, (data, title) in zip(axes, methods):
    _, X_test, _, y_test = train_test_split(data, y_dr, test_size=0.2, random_state=42)
    _, preds = test_extraction(data, y_dr)
    
    errors = (preds != y_test.values)
    embedded = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_test)
    
    sns.scatterplot(
        x=embedded[:, 0], y=embedded[:, 1], 
        hue=y_test, style=errors, 
        palette='magma', ax=ax, markers={False: 'o', True: 'X'}, alpha=0.8
    )
    ax.set_title(f"t-SNE: {title}", fontweight='bold')
    ax.legend(title='BNP Quartile / Misclass', loc='best', fontsize='small')

plt.tight_layout()
plt.show()
# %%
