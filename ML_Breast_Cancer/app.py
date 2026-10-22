"""
Streamlit app - Breast Cancer Classification (KNN, Logistic Regression, Decision Tree)

EDUCATIONAL USE ONLY. This app is NOT a real breast cancer prediction or diagnosis tool.

How to run:
    1. Put this file in the same folder as the 3 .pkl files + 2 .txt files
       (or put those files in a "models/" subfolder)
    2. pip install -r requirements.txt
    3. streamlit run app.py
"""
import urllib.request
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Classifier (Educational)", page_icon="🩺", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILES = {
    "K-Nearest Neighbors": "breast_knn_model.pkl",
    "Logistic Regression": "breast_logistic_model.pkl",
    "Decision Tree": "breast_tree_model.pkl",
}
# Split used in the training notebook (the Decision Tree reaches 100% train accuracy on it)
TEST_SIZE = 0.2
RANDOM_STATE = 42

SKLEARN_LOGO_URL = "https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png"
SKLEARN_DATASET_URL = "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html"
UCI_DATASET_URL = "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"

SCOPE_TEST = "Test set (20%)"
SCOPE_ALL = "Entire dataset"

PRESET_MEAN_ALL = "Mean of all data"
PRESET_MEAN_MAL = "Mean of Malignant class"
PRESET_MEAN_BEN = "Mean of Benign class"
PRESET_ROW = "Dataset row number"


# ----------------------------------------------------------------------------
# Educational-use disclaimer (pop-up shown before entering the app)
# ----------------------------------------------------------------------------
@st.dialog("⚠️ Educational Use Only")
def disclaimer_dialog():
    st.markdown(
        """
**This app is for learning purposes only.**

- It demonstrates how three machine learning algorithms (KNN, Logistic Regression and
  Decision Tree) classify the public *Breast Cancer Wisconsin (Diagnostic)* dataset.
- It is **NOT** a real breast cancer prediction or diagnosis tool.
- The results **must not** be used for medical decisions, self-diagnosis or treatment.
- If you have any health concern, please consult a qualified doctor or healthcare professional.
        """
    )
    if st.button("I understand, continue", type="primary"):
        st.session_state["disclaimer_accepted"] = True
        st.rerun()


if not st.session_state.get("disclaimer_accepted", False):
    disclaimer_dialog()
    st.title("🩺 Breast Cancer Classifier (Educational)")
    st.info("Please read and accept the notice to continue. If you closed it by accident, reload the page.")
    st.stop()


def find_file(name: str) -> Path:
    for folder in (BASE_DIR, BASE_DIR / "models"):
        p = folder / name
        if p.exists():
            return p
    st.error(f"File `{name}` was not found. Put it in the same folder as app.py or in a `models/` subfolder.")
    st.stop()


# ----------------------------------------------------------------------------
# Loaders (cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    return {name: joblib.load(find_file(fname)) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_names():
    feats = [l.strip() for l in find_file("breast_feature_names.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    targets = [l.strip() for l in find_file("breast_target_names.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    return feats, targets


@st.cache_data
def load_data():
    data = load_breast_cancer()  # dataset bundled with scikit-learn
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X, y, X_train, X_test, y_train, y_test, data.DESCR


@st.cache_data(ttl=86400)
def logo_available() -> bool:
    """Check (server side) that the logo URL is reachable, so we never show a broken image."""
    try:
        with urllib.request.urlopen(SKLEARN_LOGO_URL, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


MODELS = load_models()
FEATURES, TARGET_NAMES = load_names()
X, y, X_train, X_test, y_train, y_test, DATASET_DESCR = load_data()
X = X[FEATURES]
X_train, X_test = X_train[FEATURES], X_test[FEATURES]
ALGOS = list(MODELS.keys())
COLORS = {"K-Nearest Neighbors": "#4C78A8", "Logistic Regression": "#F58518", "Decision Tree": "#54A24B"}


@st.cache_data
def evaluate_all(scope: str):
    """Compute metrics for all models on the test set or on the entire dataset."""
    Xe, ye = (X_test, y_test) if scope == SCOPE_TEST else (X, y)
    rows, details = [], {}
    for name, model in MODELS.items():
        pred = model.predict(Xe)
        proba = model.predict_proba(Xe)[:, 1]  # probability of class 1 (Benign)
        fpr, tpr, _ = roc_curve(ye, proba)
        rows.append(
            {
                "Algorithm": name,
                "Accuracy": accuracy_score(ye, pred),
                "Precision (Malignant)": precision_score(ye, pred, pos_label=0),
                "Recall (Malignant)": recall_score(ye, pred, pos_label=0),
                "F1 (Malignant)": f1_score(ye, pred, pos_label=0),
                "F1 (macro)": f1_score(ye, pred, average="macro"),
                "ROC-AUC": roc_auc_score(ye, proba),
                "Train Accuracy": accuracy_score(y_train, model.predict(X_train)),
                "Test Accuracy": accuracy_score(y_test, model.predict(X_test)),
            }
        )
        details[name] = {
            "pred": pred,
            "report": pd.DataFrame(
                classification_report(ye, pred, target_names=TARGET_NAMES, output_dict=True)
            ).T,
            "cm": confusion_matrix(ye, pred),
            "fpr": fpr,
            "tpr": tpr,
            "n": len(ye),
        }
    return pd.DataFrame(rows).set_index("Algorithm"), details


# ----------------------------------------------------------------------------
# Feature input state
# ----------------------------------------------------------------------------
def fkey(i: int) -> str:
    return f"feat_{i}"


def set_values(series: pd.Series):
    for i, f in enumerate(FEATURES):
        lo, hi = float(X[f].min()), float(X[f].max())
        st.session_state[fkey(i)] = float(np.clip(series[f], lo, hi))


def apply_preset():
    preset = st.session_state.get("preset", PRESET_MEAN_ALL)
    if preset == PRESET_MEAN_ALL:
        set_values(X.mean())
    elif preset == PRESET_MEAN_MAL:
        set_values(X[y == 0].mean())
    elif preset == PRESET_MEAN_BEN:
        set_values(X[y == 1].mean())
    else:  # dataset row number
        idx = int(st.session_state.get("row_idx", 0))
        set_values(X.iloc[idx])


if fkey(0) not in st.session_state:
    set_values(X.mean())

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    if logo_available():
        st.image(SKLEARN_LOGO_URL, width=140)
    st.title("🩺 Breast Cancer")
    algo = st.selectbox("Choose algorithm", ALGOS, key="algo")

    st.divider()
    st.subheader("Quick fill inputs")
    preset = st.selectbox(
        "Value source",
        [PRESET_MEAN_ALL, PRESET_MEAN_MAL, PRESET_MEAN_BEN, PRESET_ROW],
        key="preset",
    )
    if preset == PRESET_ROW:
        row_idx = st.number_input("Row number (0 - %d)" % (len(X) - 1), 0, len(X) - 1, 0, key="row_idx")
        st.caption(f"True label of this row: **{TARGET_NAMES[int(y.iloc[int(row_idx)])]}**")
    st.button("Apply to inputs", on_click=apply_preset)

    st.divider()
    scope = st.radio(
        "Evaluate report on",
        [SCOPE_TEST, SCOPE_ALL],
        help="Test set = 114 rows not used during training (80/20 split, random_state=42, stratified). "
        "'Entire dataset' can be too optimistic because many rows were already seen by the models during training.",
    )
    st.caption("⚠️ For learning purposes only. Not a medical diagnosis tool.")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
st.title("Breast Cancer Classification: KNN vs Logistic Regression vs Decision Tree")
st.warning("Educational demo only. These results are not a real breast cancer prediction or diagnosis.", icon="⚠️")
tab_pred, tab_report, tab_cmp, tab_data = st.tabs(
    ["🔮 Prediction", "📄 Report per Algorithm", "⚖️ Comparison", "📊 Dataset"]
)

# ---------------------------- Prediction tab --------------------------------
with tab_pred:
    st.subheader("Change the feature values")
    st.caption("Slider ranges follow the minimum and maximum values found in the dataset.")

    groups = {
        "Mean": [i for i, f in enumerate(FEATURES) if f.startswith("mean ")],
        "Error (standard error)": [i for i, f in enumerate(FEATURES) if f.endswith(" error")],
        "Worst (largest value)": [i for i, f in enumerate(FEATURES) if f.startswith("worst ")],
    }
    for title, idxs in groups.items():
        with st.expander(title, expanded=title.startswith("Mean")):
            cols = st.columns(3)
            for n, i in enumerate(idxs):
                f = FEATURES[i]
                lo, hi = float(X[f].min()), float(X[f].max())
                step = (hi - lo) / 1000
                with cols[n % 3]:
                    st.slider(f, lo, hi, step=step, format="%.4f", key=fkey(i))

    input_df = pd.DataFrame([[st.session_state[fkey(i)] for i in range(len(FEATURES))]], columns=FEATURES)

    st.divider()
    st.subheader(f"Prediction result: {algo}")
    model = MODELS[algo]
    pred = int(model.predict(input_df)[0])
    proba = model.predict_proba(input_df)[0]

    c1, c2 = st.columns([1, 1])
    with c1:
        if pred == 0:
            st.error(f"### {TARGET_NAMES[0]}")
        else:
            st.success(f"### {TARGET_NAMES[1]}")
        st.metric("Model confidence", f"{proba[pred] * 100:.1f}%")
    with c2:
        st.bar_chart(pd.DataFrame({"Probability": proba}, index=TARGET_NAMES))

    st.markdown("**Predictions of all algorithms for the same input**")
    cols = st.columns(len(ALGOS))
    for col, name in zip(cols, ALGOS):
        p = int(MODELS[name].predict(input_df)[0])
        pr = MODELS[name].predict_proba(input_df)[0]
        with col:
            st.markdown(f"**{name}**")
            st.write(TARGET_NAMES[p])
            st.caption(f"Malignant {pr[0] * 100:.1f}% · Benign {pr[1] * 100:.1f}%")

    preds_set = {int(MODELS[n].predict(input_df)[0]) for n in ALGOS}
    if len(preds_set) > 1:
        st.warning("The algorithms disagree on this input, so the result should not be trusted blindly.")

# ---------------------------- Report tab ------------------------------------
with tab_report:
    metrics_df, details = evaluate_all(scope)
    st.subheader(f"Report: {algo}")
    st.caption(f"Evaluated on: **{scope}** ({details[algo]['n']} rows)")

    m = metrics_df.loc[algo]
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{m['Accuracy'] * 100:.2f}%")
    b.metric("Recall Malignant", f"{m['Recall (Malignant)'] * 100:.2f}%")
    c.metric("F1 macro", f"{m['F1 (macro)'] * 100:.2f}%")
    d.metric("ROC-AUC", f"{m['ROC-AUC']:.4f}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Classification report**")
        st.dataframe(details[algo]["report"].style.format("{:.3f}"))
        st.caption(
            f"Train accuracy: {m['Train Accuracy'] * 100:.2f}% · Test accuracy: {m['Test Accuracy'] * 100:.2f}%"
        )
        if algo == "Decision Tree" and m["Train Accuracy"] - m["Test Accuracy"] > 0.05:
            st.info(
                "The gap between train and test accuracy is large: the tree is overfitting (its depth is not limited)."
            )
    with right:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4, 3.6))
        ConfusionMatrixDisplay(details[algo]["cm"], display_labels=["Malignant", "Benign"]).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Model details**")
    model = MODELS[algo]
    if algo == "K-Nearest Neighbors":
        st.write(f"n_neighbors = **{model.n_neighbors}**, metric = **{model.metric}**, weights = **{model.weights}**")
        st.caption("KNN was trained without a scaler, so features with large scales (e.g. area) dominate the distance.")
    elif algo == "Logistic Regression":
        coef = pd.Series(model.coef_[0], index=FEATURES)
        top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.barh(top.index, top.values, color=["#E45756" if v < 0 else "#4C78A8" for v in top.values])
        ax.set_title("Top 10 coefficients (positive → leans Benign)")
        st.pyplot(fig)
        plt.close(fig)
    else:
        depth = st.slider("Tree depth to display", 1, int(model.get_depth()), 3)
        fig, ax = plt.subplots(figsize=(16, 7))
        plot_tree(model, feature_names=FEATURES, class_names=TARGET_NAMES, filled=True, rounded=True,
                  fontsize=8, max_depth=depth, ax=ax)
        st.pyplot(fig)
        plt.close(fig)
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False).head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.barh(imp.index, imp.values, color=COLORS[algo])
        ax.set_title("Top 10 most important features")
        st.pyplot(fig)
        plt.close(fig)

# ---------------------------- Comparison tab --------------------------------
with tab_cmp:
    metrics_df, details = evaluate_all(scope)
    st.subheader("Comparison of all algorithms")
    st.caption(f"Evaluated on: **{scope}**")

    show_cols = ["Accuracy", "Precision (Malignant)", "Recall (Malignant)", "F1 (Malignant)", "F1 (macro)", "ROC-AUC"]
    st.dataframe(
        metrics_df[show_cols].style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"),
    )
    best = metrics_df["Accuracy"].idxmax()
    st.success(f"Highest accuracy with the current settings: **{best}** ({metrics_df.loc[best, 'Accuracy'] * 100:.2f}%)")
    st.caption(
        "In a cancer setting, **Recall (Malignant)** matters: a low value means malignant cases are being missed (false negatives)."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Metric comparison**")
        fig, ax = plt.subplots(figsize=(7, 4))
        w = 0.25
        xs = np.arange(len(show_cols))
        for k, name in enumerate(ALGOS):
            ax.bar(xs + (k - 1) * w, metrics_df.loc[name, show_cols].values, w, label=name, color=COLORS[name])
        ax.set_xticks(xs)
        ax.set_xticklabels([c.replace(" (", "\n(") for c in show_cols], fontsize=8)
        ax.set_ylim(max(0.0, metrics_df[show_cols].values.min() - 0.05), 1.01)
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.markdown("**ROC curves**")
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for name in ALGOS:
            ax.plot(details[name]["fpr"], details[name]["tpr"], color=COLORS[name],
                    label=f"{name} (AUC={metrics_df.loc[name, 'ROC-AUC']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, loc="lower right")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Confusion matrices side by side**")
    cols = st.columns(len(ALGOS))
    for col, name in zip(cols, ALGOS):
        with col:
            fig, ax = plt.subplots(figsize=(3.6, 3.3))
            ConfusionMatrixDisplay(details[name]["cm"], display_labels=["Malignant", "Benign"]).plot(
                ax=ax, cmap="Blues", colorbar=False
            )
            ax.set_title(name, fontsize=10)
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("**Train vs Test accuracy (overfitting check)**")
    st.bar_chart(metrics_df[["Train Accuracy", "Test Accuracy"]])

# ---------------------------- Dataset tab -----------------------------------
with tab_data:
    head_logo, head_text = st.columns([1, 5])
    with head_logo:
        if logo_available():
            st.image(SKLEARN_LOGO_URL, width=120)
    with head_text:
        st.subheader("Breast Cancer Wisconsin (Diagnostic) Dataset")
        st.write(
            "Loaded directly from **scikit-learn** with `sklearn.datasets.load_breast_cancer()`. "
            "scikit-learn's copy comes from the UCI Machine Learning Repository."
        )
        st.markdown(
            f"- 📘 [scikit-learn documentation: load_breast_cancer]({SKLEARN_DATASET_URL})\n"
            f"- 🗄️ [Original dataset on UCI Machine Learning Repository]({UCI_DATASET_URL})"
        )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", len(X))
    k2.metric("Features", len(FEATURES))
    k3.metric(TARGET_NAMES[0], int((y == 0).sum()))
    k4.metric(TARGET_NAMES[1], int((y == 1).sum()))

    st.markdown("**Data table**")
    view = X.copy()
    view["label"] = y.map(dict(enumerate(TARGET_NAMES)))
    st.dataframe(view, height=400)

    st.markdown("**Descriptive statistics**")
    st.dataframe(X.describe().T)

    with st.expander("Full dataset description (from scikit-learn)"):
        st.text(DATASET_DESCR)