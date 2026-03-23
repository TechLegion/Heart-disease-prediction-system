"""
Streamlit Web App for Heart Disease Prediction System.

Three pages:
  1. Predict Heart Disease  – enter patient data, pick a model, get prediction.
  2. Model Comparison       – metrics table, bar chart, ROC curves, CV results.
  3. About / Methodology    – overview of the research approach.

Run with:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
RESULTS_CSV = os.path.join(RESULTS_DIR, 'comparison_results.csv')
CV_CSV = os.path.join(RESULTS_DIR, 'cv_results.csv')
ROC_FILE = os.path.join(RESULTS_DIR, 'roc_data.joblib')

MODEL_FILES = {
    'Baseline ANN': 'baseline_ann.joblib',
    'GA-Optimized ANN': 'ga_ann.joblib',
    'PSO-Optimized ANN': 'pso_ann.joblib',
    'Hybrid GA-PSO-ANN': 'hybrid_ann.joblib',
}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))


@st.cache_resource
def load_label_encoders():
    return joblib.load(os.path.join(MODELS_DIR, 'label_encoders.joblib'))


@st.cache_resource
def load_feature_names():
    return joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))


def available_models():
    """Return list of model names whose .joblib files exist."""
    found = []
    for name, fname in MODEL_FILES.items():
        if os.path.exists(os.path.join(MODELS_DIR, fname)):
            found.append(name)
    if not found:
        legacy = os.path.join(MODELS_DIR, 'ann_model.joblib')
        if os.path.exists(legacy):
            found.append('Baseline ANN')
    return found


# ---------------------------------------------------------------------------
# Normal ranges for risk-factor highlighting
# ---------------------------------------------------------------------------
NORMAL_RANGES = {
    'age': (20, 65),
    'trestbps': (90, 140),
    'chol': (125, 200),
    'thalach': (100, 190),
    'oldpeak': (0.0, 2.0),
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title='Heart Disease Prediction',
    page_icon='heart',
    layout='wide',
)

page = st.sidebar.radio(
    'Navigation',
    ['Predict Heart Disease', 'Model Comparison', 'About / Methodology'],
)


# ===================================================================
# PAGE 1 – Predict Heart Disease
# ===================================================================
def page_predict():
    st.title('Heart Disease Prediction')
    st.write('Enter patient medical data below and click **Predict**.')

    models = available_models()
    if not models:
        st.error('No trained models found. Run `python main.py` first.')
        return

    chosen = st.selectbox('Select prediction model', models)
    fname = MODEL_FILES.get(chosen, 'ann_model.joblib')
    model = load_model(fname)
    if model is None:
        model = load_model('ann_model.joblib')
    scaler = load_scaler()
    label_encoders = load_label_encoders()
    feature_names = load_feature_names()

    with st.form('patient_form'):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider('Age', 20, 100, 55)
            sex = st.selectbox('Sex', ['Male', 'Female'])
            cp = st.selectbox('Chest Pain Type', [
                '0 - Typical Angina', '1 - Atypical Angina',
                '2 - Non-anginal Pain', '3 - Asymptomatic'])
            trestbps = st.slider('Resting Blood Pressure (mm Hg)', 80, 220, 130)
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])

        with col2:
            chol = st.slider('Serum Cholesterol (mg/dl)', 100, 600, 245)
            restecg = st.selectbox('Resting ECG', [
                '0 - Normal', '1 - ST-T Wave Abnormality',
                '2 - Left Ventricular Hypertrophy'])
            thalach = st.slider('Max Heart Rate Achieved', 60, 220, 150)
            exang = st.selectbox('Exercise-Induced Angina', ['No', 'Yes'])

        with col3:
            oldpeak = st.number_input('ST Depression (Oldpeak)', 0.0, 7.0, 1.0, step=0.1)
            slope = st.selectbox('Slope of Peak Exercise ST', [
                '0 - Upsloping', '1 - Flat', '2 - Downsloping'])
            ca = st.selectbox('Major Vessels Colored by Fluoroscopy',
                              ['0', '1', '2', '3'])
            thal = st.selectbox('Thalassemia', [
                '3 - Normal', '6 - Fixed Defect', '7 - Reversible Defect'])

        submitted = st.form_submit_button('Predict', width='stretch')

    if submitted:
        raw = {
            'age': float(age),
            'sex': 1.0 if sex == 'Male' else 0.0,
            'cp': float(cp[0]),
            'trestbps': float(trestbps),
            'chol': float(chol),
            'fbs': 1.0 if fbs == 'Yes' else 0.0,
            'restecg': float(restecg[0]),
            'thalach': float(thalach),
            'exang': 1.0 if exang == 'Yes' else 0.0,
            'oldpeak': float(oldpeak),
            'slope': float(slope[0]),
            'ca': float(ca),
            'thal': float(thal[0]),
        }

        for col_name in ['sex', 'cp', 'fbs', 'restecg', 'exang',
                          'slope', 'ca', 'thal']:
            if col_name in label_encoders:
                le = label_encoders[col_name]
                val_str = str(int(raw[col_name]))
                if val_str in le.classes_:
                    raw[col_name] = float(le.transform([val_str])[0])
                else:
                    raw[col_name] = float(le.transform([le.classes_[0]])[0])

        feature_df = pd.DataFrame(
            [[raw[f] for f in feature_names]], columns=feature_names)
        scaled = scaler.transform(feature_df)

        prediction = model.predict(scaled)[0]
        probabilities = model.predict_proba(scaled)[0]
        confidence = probabilities[1] if prediction == 1 else probabilities[0]

        st.divider()
        if prediction == 1:
            st.error('### Heart Disease Detected')
        else:
            st.success('### No Heart Disease Detected')
        st.metric('Confidence', f'{confidence * 100:.1f}%')
        st.caption(f'Model used: **{chosen}**')

        st.subheader('Risk Factor Analysis')
        risk_flags = []
        for feat, (lo, hi) in NORMAL_RANGES.items():
            val = raw.get(feat)
            if val is not None and (val < lo or val > hi):
                risk_flags.append((feat, val, lo, hi))

        if risk_flags:
            for feat, val, lo, hi in risk_flags:
                label = (feat.replace('trestbps', 'Blood Pressure')
                             .replace('chol', 'Cholesterol')
                             .replace('thalach', 'Max Heart Rate')
                             .replace('oldpeak', 'ST Depression')
                             .replace('age', 'Age'))
                direction = 'above' if val > hi else 'below'
                st.warning(
                    f'**{label}**: {val} is {direction} normal range '
                    f'({lo} - {hi})')
        else:
            st.info('All key indicators are within normal ranges.')


# ===================================================================
# PAGE 2 – Model Comparison
# ===================================================================
def page_compare():
    st.title('Model Comparison')

    if not os.path.exists(RESULTS_CSV):
        st.error('Results not found. Run `python main.py` first.')
        return

    df = pd.read_csv(RESULTS_CSV)
    metric_cols = [c for c in ['Accuracy','Precision','Recall','F1-Score','AUC']
                   if c in df.columns]

    best_idx = df['Accuracy'].idxmax()
    st.success(
        f"**Best model:** {df.loc[best_idx, 'Model']} — "
        f"**{df.loc[best_idx, 'Accuracy'] * 100:.2f}%** accuracy"
        + (f", **{df.loc[best_idx, 'AUC'] * 100:.2f}%** AUC"
           if 'AUC' in df.columns else ''))

    # --- Metrics table ---
    st.subheader('Single-Split Metrics')
    disp = df.copy()
    for c in metric_cols:
        disp[c] = disp[c].apply(lambda v: f'{v * 100:.2f}%')
    st.dataframe(disp, width='stretch', hide_index=True)

    # --- Bar chart ---
    st.subheader('Performance Chart')
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    fig = go.Figure()
    for mc, col in zip(metric_cols, colors):
        fig.add_trace(go.Bar(name=mc, x=df['Model'], y=df[mc],
                             marker_color=col))
    fig.update_layout(barmode='group', yaxis_title='Score',
                      yaxis_range=[0, 1], height=500,
                      legend_title='Metric')
    st.plotly_chart(fig, width='stretch')

    # --- ROC curves ---
    if os.path.exists(ROC_FILE):
        st.subheader('ROC Curves')
        roc_data = joblib.load(ROC_FILE)
        fig_roc = go.Figure()
        roc_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
        for (name, (fpr, tpr, auc)), col in zip(roc_data.items(), roc_colors):
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f'{name} (AUC={auc:.3f})', line=dict(color=col)))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            name='Random', line=dict(color='gray', dash='dash')))
        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500)
        st.plotly_chart(fig_roc, width='stretch')

    # --- Cross-validation results ---
    if os.path.exists(CV_CSV):
        st.subheader('5-Fold Cross-Validation Results')
        cv = pd.read_csv(CV_CSV)
        cv_display = cv[['Model']].copy()
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']:
            mcol = f'{metric}_mean'
            scol = f'{metric}_std'
            if mcol in cv.columns and scol in cv.columns:
                cv_display[metric] = cv.apply(
                    lambda r: f"{r[mcol]*100:.2f}% ± {r[scol]*100:.2f}%",
                    axis=1)
        st.dataframe(cv_display, width='stretch', hide_index=True)

    with st.expander('What do these metrics mean?'):
        st.markdown(
            '- **Accuracy** — proportion of all predictions that are correct.\n'
            '- **Precision** — of predicted positives, how many are truly positive.\n'
            '- **Recall** — of actual positives, how many were detected.\n'
            '- **F1-Score** — harmonic mean of precision and recall.\n'
            '- **AUC** — area under the ROC curve; measures discrimination ability '
            '(1.0 = perfect, 0.5 = random).'
        )


# ===================================================================
# PAGE 3 – About / Methodology
# ===================================================================
def page_about():
    st.title('About This Project')

    st.markdown("""
### Heart Disease Prediction Using a Hybrid GA-PSO-ANN Approach

This system predicts the presence of heart disease using patient medical data
from the **UCI Heart Disease dataset** (303 records, 13 clinical features).

---

#### Methodology

| Stage | Technique | Role |
|-------|-----------|------|
| **Data Preprocessing** | Cleaning, encoding, Z-score normalisation | Prepare raw data for modelling |
| **Baseline ANN** | Multi-Layer Perceptron (sklearn) | Reference model (no optimisation) |
| **GA-Optimised ANN** | Genetic Algorithm (DEAP) | Evolves architecture + learning rate |
| **PSO-Optimised ANN** | Particle Swarm Optimisation (PySwarms) | Fine-tunes learning rate + L2 regularisation |
| **Hybrid GA-PSO-ANN** | GA then PSO, chained | Two-stage optimisation for best performance |

---

#### How the Hybrid Works

1. **Genetic Algorithm** searches over ANN architectures (hidden-layer sizes)
   and learning rates using selection, crossover, and mutation — inspired by
   natural evolution.
2. **Particle Swarm Optimisation** takes the GA-selected architecture and
   fine-tunes continuous training parameters (learning rate, L2 penalty) —
   inspired by the collective behaviour of bird flocks.
3. The **final model** is trained with all optimised hyper-parameters and
   evaluated on a held-out test set.

---

#### Evaluation

- **Single-split** evaluation on a 80/20 stratified test set.
- **5-fold cross-validation** for statistical robustness.
- Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC.

---

#### Dataset Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | age | Age in years |
| 2 | sex | 1 = male, 0 = female |
| 3 | cp | Chest pain type (0-3) |
| 4 | trestbps | Resting blood pressure (mm Hg) |
| 5 | chol | Serum cholesterol (mg/dl) |
| 6 | fbs | Fasting blood sugar > 120 mg/dl |
| 7 | restecg | Resting ECG results (0-2) |
| 8 | thalach | Maximum heart rate achieved |
| 9 | exang | Exercise-induced angina |
| 10 | oldpeak | ST depression induced by exercise |
| 11 | slope | Slope of peak exercise ST segment |
| 12 | ca | Number of major vessels (0-3) coloured by fluoroscopy |
| 13 | thal | Thalassemia (3=normal, 6=fixed defect, 7=reversible defect) |

---

*Built as a Final Year Project — Heart Disease Prediction with Hybrid GA-PSO-ANN.*
""")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == 'Predict Heart Disease':
    page_predict()
elif page == 'Model Comparison':
    page_compare()
else:
    page_about()
