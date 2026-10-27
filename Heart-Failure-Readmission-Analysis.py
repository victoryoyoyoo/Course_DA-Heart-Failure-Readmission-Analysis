#%%

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import gdown
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, f1_score, roc_auc_score, confusion_matrix

warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

FILE_ID = '1bSa2A5xgYUv4QHC1U8zz0UF-lngOUvNi'
FILE_NAME = 'D01_heart_failure_readmission_dataset.csv'
RANDOM_STATE = 42

if not os.path.exists(FILE_NAME):
    print("Downloading dataset...")
    gdown.download(id=FILE_ID, output=FILE_NAME, quiet=False)

try:
    df = pd.read_csv(FILE_NAME)
except Exception as e:
    raise RuntimeError(f"Data loading failed: {e}")

df = df.dropna(subset=['bnp', 'readmitted_30d'])

target_reg, target_clf = 'bnp', 'readmitted_30d'
exclude_cols = ['patient_id', target_clf, target_reg]
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in exclude_cols]
features = num_cols + cat_cols

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
msno.matrix(df, sparkline=False, ax=axes[0], color=(0.15, 0.35, 0.55), fontsize=10)
axes[0].set_title("Missingness Matrix", fontweight='bold', fontsize=12)
msno.heatmap(df, ax=axes[1], fontsize=10, cmap="YlGnBu")
axes[1].set_title("Missingness Correlation", fontweight='bold', fontsize=12)
plt.tight_layout()
plt.show()

df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[target_clf]
)
X_train, X_test = df_train[features], df_test[features]
yr_train, yr_test = df_train[target_reg], df_test[target_reg]
yc_train, yc_test = df_train[target_clf], df_test[target_clf]


class IQROutlierMasker(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        q1, q3 = np.nanpercentile(np.asarray(X, dtype=float), [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        X[(X < self.lower_) | (X > self.upper_)] = np.nan
        return X


def build_preprocessor(scaler):
    return ColumnTransformer([
        ('num', Pipeline([
            ('outlier', IQROutlierMasker()),
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler)
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ])


print(f"Pipeline Stage 1 complete. Train shape: {df_train.shape}, test shape: {df_test.shape}")


scalers = {'Standard': StandardScaler(), 'MinMax': MinMaxScaler(), 'Robust': RobustScaler()}
scaler_scores = []
clf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, scaler in scalers.items():
    clf = Pipeline([
        ('pre', build_preprocessor(scaler)),
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE))
    ])
    cv_scores = cross_val_score(clf, X_train, yc_train, cv=clf_cv, scoring='f1_weighted')
    scaler_scores.append({'Scaler': name, 'Weighted F1': cv_scores.mean()})

scaler_df = pd.DataFrame(scaler_scores)
best_scaler_name = scaler_df.loc[scaler_df['Weighted F1'].idxmax()]['Scaler']
best_scaler = scalers[best_scaler_name]
print(scaler_df.round(4).to_string(index=False))
print(f"Best scaler (5-fold CV on training set): {best_scaler_name}")

reg_model = Pipeline([('pre', build_preprocessor(best_scaler)), ('svr', SVR(kernel='rbf', C=10))])
clf_model = Pipeline([
    ('pre', build_preprocessor(best_scaler)),
    ('rf', RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=RANDOM_STATE))
])

reg_cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
clf_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
reg_scores = cross_validate(reg_model, X_train, yr_train, cv=reg_cv,
                            scoring={'RMSE': 'neg_root_mean_squared_error', 'R2': 'r2'})
clf_scores = cross_validate(clf_model, X_train, yc_train, cv=clf_cv,
                            scoring={'F1-Score': 'f1_weighted', 'ROC-AUC': 'roc_auc'})

metrics = []
for fold in range(10):
    metrics.extend([
        {'Task': 'Regression', 'Metric': 'RMSE', 'Value': -reg_scores['test_RMSE'][fold]},
        {'Task': 'Regression', 'Metric': 'R2', 'Value': reg_scores['test_R2'][fold]},
        {'Task': 'Classification', 'Metric': 'F1-Score', 'Value': clf_scores['test_F1-Score'][fold]},
        {'Task': 'Classification', 'Metric': 'ROC-AUC', 'Value': clf_scores['test_ROC-AUC'][fold]}
    ])

res_df = pd.DataFrame(metrics)
print("10-fold cross-validation on the training set")
print(res_df.groupby(['Task', 'Metric'])['Value'].agg(['mean', 'std']).round(4))

reg_model.fit(X_train, yr_train)
clf_model.fit(X_train, yc_train)
r_pred = reg_model.predict(X_test)
c_pred = clf_model.predict(X_test)

print("Held-out test set")
print(f"RMSE: {np.sqrt(mean_squared_error(yr_test, r_pred)):.4f}")
print(f"R2: {r2_score(yr_test, r_pred):.4f}")
print(f"F1-Score: {f1_score(yc_test, c_pred, average='weighted'):.4f}")
print(f"ROC-AUC: {roc_auc_score(yc_test, clf_model.predict_proba(X_test)[:, 1]):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].scatter(yr_test, r_pred, alpha=0.5, edgecolors='w', color='#2ca02c')
axes[0].plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], 'r--', lw=2)
axes[0].set_title(f"SVR Prediction Parity (Target: {target_reg})", fontweight='bold')
axes[0].set_xlabel("Actual Values")
axes[0].set_ylabel("Predicted Values")

cm = confusion_matrix(yc_test, c_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
axes[1].set_title(f"RF Confusion Matrix (Target: {target_clf})", fontweight='bold')
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()


from sklearn.feature_selection import SelectKBest, SelectFromModel, f_regression, f_classif, RFE
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.svm import LinearSVR

K = 5


def evaluate_subset(selector, task='reg'):
    steps = [('pre', build_preprocessor(best_scaler))]
    if selector is not None:
        steps.append(('select', selector))
    if task == 'reg':
        steps.append(('model', SVR(kernel='rbf', C=10)))
        cv, scoring, y = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), 'neg_root_mean_squared_error', yr_train
    else:
        steps.append(('model', RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=RANDOM_STATE)))
        cv, scoring, y = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), 'f1_weighted', yc_train
    scores = cross_val_score(Pipeline(steps), X_train, y, cv=cv, scoring=scoring)
    if task == 'reg':
        scores = -scores
    return scores.mean(), scores.std()


base_rmse, _ = evaluate_subset(None, 'reg')
base_f1, _ = evaluate_subset(None, 'clf')

strategies = {
    'Filter: Univariate F-test': (
        SelectKBest(f_regression, k=K),
        SelectKBest(f_classif, k=K)
    ),
    'Embedded: Lasso / LogReg': (
        SelectFromModel(LassoCV(cv=5), max_features=K, threshold=-np.inf),
        SelectFromModel(LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear'), max_features=K, threshold=-np.inf)
    ),
    'Wrapper: RFE': (
        RFE(LinearSVR(random_state=RANDOM_STATE, dual='auto'), n_features_to_select=K),
        RFE(RandomForestClassifier(n_estimators=20, random_state=RANDOM_STATE), n_features_to_select=K)
    )
}

results = []
for name, (sel_reg, sel_clf) in strategies.items():
    m_reg, s_reg = evaluate_subset(sel_reg, 'reg')
    m_clf, s_clf = evaluate_subset(sel_clf, 'clf')
    results.extend([
        {'Strategy': name, 'Task': 'Regression (RMSE)', 'Score': m_reg, 'Std': s_reg},
        {'Strategy': name, 'Task': 'Classification (F1)', 'Score': m_clf, 'Std': s_clf}
    ])

df_res = pd.DataFrame(results)
print(f"Baseline RMSE: {base_rmse:.4f}, baseline F1: {base_f1:.4f}")
print(df_res.round(4).to_string(index=False))

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


import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

train_levels, level_edges = pd.qcut(df_train[target_reg], q=4, labels=[0, 1, 2, 3], retbins=True)
level_edges[0], level_edges[-1] = -np.inf, np.inf
y_dr_train = train_levels.astype(int)
y_dr_test = pd.cut(df_test[target_reg], bins=level_edges, labels=[0, 1, 2, 3]).astype(int)

dr_preprocessor = build_preprocessor(best_scaler)
X_dr_train = dr_preprocessor.fit_transform(X_train)
X_dr_test = dr_preprocessor.transform(X_test)

pca = PCA(n_components=10, random_state=RANDOM_STATE).fit(X_dr_train)
X_pca_train, X_pca_test = pca.transform(X_dr_train), pca.transform(X_dr_test)

input_dim = X_dr_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dropout(0.2)(encoded)
encoded = Dense(10, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_dr_train, X_dr_train, epochs=100, batch_size=32, shuffle=True, verbose=0, validation_split=0.2)

encoder = Model(input_layer, encoded)
X_ae_train = encoder.predict(X_dr_train, verbose=0)
X_ae_test = encoder.predict(X_dr_test, verbose=0)


def test_extraction(X_data_train, X_data_test):
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE)
    model.fit(X_data_train, y_dr_train)
    preds = model.predict(X_data_test)
    return f1_score(y_dr_test, preds, average='weighted'), preds


methods = [
    (X_dr_train, X_dr_test, 'Raw Features'),
    (X_pca_train, X_pca_test, 'PCA (10)'),
    (X_ae_train, X_ae_test, 'Deep Autoencoder')
]
extraction = {title: test_extraction(data_train, data_test) for data_train, data_test, title in methods}

print(f"Full Feature Space F1: {extraction['Raw Features'][0]:.4f}")
print(f"PCA (10 components) F1: {extraction['PCA (10)'][0]:.4f}")
print(f"Deep Autoencoder F1: {extraction['Deep Autoencoder'][0]:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (_, data_test, title) in zip(axes, methods):
    preds = extraction[title][1]
    errors = (preds != y_dr_test.values)
    embedded = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30).fit_transform(data_test)

    sns.scatterplot(
        x=embedded[:, 0], y=embedded[:, 1],
        hue=y_dr_test.values, style=errors,
        palette='magma', ax=ax, markers={False: 'o', True: 'X'}, alpha=0.8
    )
    ax.set_title(f"t-SNE: {title}", fontweight='bold')
    ax.legend(title='BNP Quartile / Misclass', loc='best', fontsize='small')

plt.tight_layout()
plt.show()


from sklearn.inspection import permutation_importance

os.makedirs('models', exist_ok=True)
joblib.dump(reg_model, 'models/svr_bnp_regressor.joblib')
joblib.dump(clf_model, 'models/rf_readmission_classifier.joblib')
print("Saved trained pipelines to models/")

perm = permutation_importance(
    clf_model, X_test, yc_test, n_repeats=10, random_state=RANDOM_STATE, scoring='f1_weighted'
)
importance_df = pd.DataFrame({
    'feature': features,
    'importance_mean': perm.importances_mean,
    'importance_std': perm.importances_std
}).sort_values('importance_mean', ascending=False).head(10)

plt.figure(figsize=(9, 5))
plt.barh(importance_df['feature'][::-1], importance_df['importance_mean'][::-1],
         xerr=importance_df['importance_std'][::-1], color='#4c72b0', alpha=0.85)
plt.title("Top 10 Features by Permutation Importance (30-day Readmission)", fontweight='bold')
plt.xlabel("Mean F1 drop when feature is shuffled")
plt.tight_layout()
plt.show()

print(importance_df.to_string(index=False))

from sklearn.metrics import precision_recall_curve, average_precision_score

probs = clf_model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(yc_test, probs)
ap = average_precision_score(yc_test, probs)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color='#c44e52', lw=2, label=f'Average Precision = {ap:.3f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve: 30-day Readmission", fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Average Precision (PR-AUC): {ap:.4f}")
# %%
