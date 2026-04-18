"""
Streamlit Web App for Heart Disease Prediction System.

Pages:
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

MODEL_DESCRIPTIONS = {
    'Baseline ANN': 'Standard neural network — no optimization applied.',
    'GA-Optimized ANN': 'Architecture and learning rate evolved by Genetic Algorithm.',
    'PSO-Optimized ANN': 'Learning rate and regularization tuned by Particle Swarm Optimization.',
    'Hybrid GA-PSO-ANN': 'GA selects architecture, then PSO fine-tunes training parameters.',
}

SAMPLE_PATIENTS = {
    'Healthy Patient (Low Risk)': {
        'age': 35, 'sex': 'Female', 'cp': '0 - Typical Angina',
        'trestbps': 120, 'chol': 180, 'fbs': 'No',
        'restecg': '0 - Normal', 'thalach': 170, 'exang': 'No',
        'oldpeak': 0.5, 'slope': '0 - Upsloping', 'ca': '0',
        'thal': '3 - Normal',
    },
    'At-Risk Patient (High Risk)': {
        'age': 62, 'sex': 'Male', 'cp': '3 - Asymptomatic',
        'trestbps': 160, 'chol': 290, 'fbs': 'Yes',
        'restecg': '1 - ST-T Wave Abnormality', 'thalach': 108, 'exang': 'Yes',
        'oldpeak': 3.5, 'slope': '2 - Downsloping', 'ca': '2',
        'thal': '7 - Reversible Defect',
    },
}

FEATURE_HELP = {
    'age': 'Patient age in years.',
    'sex': 'Biological sex of the patient.',
    'cp': 'Type of chest pain. Asymptomatic (type 3) is most associated with heart disease.',
    'trestbps': 'Blood pressure (mm Hg) at rest. Normal range: 90–140.',
    'chol': 'Serum cholesterol in mg/dl. Desirable: below 200.',
    'fbs': 'Is fasting blood sugar above 120 mg/dl? May indicate diabetes.',
    'restecg': 'Results of the resting electrocardiogram.',
    'thalach': 'Highest heart rate during exercise. Lower values may indicate risk.',
    'exang': 'Does exercise cause chest pain (angina)?',
    'oldpeak': 'ST depression induced by exercise vs rest. Higher = possible ischaemia. Normal: 0–2.',
    'slope': 'Slope of the peak exercise ST segment on the ECG.',
    'ca': 'Number of major coronary vessels (0–3) visible on fluoroscopy.',
    'thal': 'Thalassemia status. Reversible defect is a strong risk indicator.',
}

NORMAL_RANGES = {
    'age': (20, 65),
    'trestbps': (90, 140),
    'chol': (125, 200),
    'thalach': (100, 190),
    'oldpeak': (0.0, 2.0),
}

RISK_LABELS = {
    'age': 'Age',
    'trestbps': 'Blood Pressure',
    'chol': 'Cholesterol',
    'thalach': 'Max Heart Rate',
    'oldpeak': 'ST Depression',
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title='Heart Disease Prediction',
    page_icon='\u2764\uFE0F',
    layout='wide',
)

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
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title('Heart Disease Prediction')
st.sidebar.caption('GA-PSO Optimized ANN')

page = st.sidebar.radio(
    'Navigation',
    ['Predict Heart Disease', 'Model Comparison', 'About / Methodology'],
)

st.sidebar.divider()
st.sidebar.caption('Final Year Project — Redeemer\'s University, Ede')


# ===================================================================
# PAGE 1 – Predict Heart Disease
# ===================================================================
def page_predict():
    st.title('Predict Heart Disease')
    st.write(
        'Enter patient clinical data below and click **Predict**. '
        'Hover over the **?** icons for guidance on each field.'
    )

    models = available_models()
    if not models:
        st.error('No trained models found. Run `python main.py` first.')
        return

    col_model, col_sample = st.columns([2, 1])
    with col_model:
        chosen = st.selectbox(
            'Prediction Model', models,
            help='Choose which trained model to use.',
        )
        st.caption(MODEL_DESCRIPTIONS.get(chosen, ''))
    with col_sample:
        sample_choice = st.selectbox(
            'Load Sample Patient',
            ['(Custom input)'] + list(SAMPLE_PATIENTS.keys()),
            help='Pick a pre-filled example to quickly test the system.',
        )

    sample = SAMPLE_PATIENTS.get(sample_choice, {})

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
            st.markdown('**Demographics & History**')
            age = st.slider('Age', 20, 100, sample.get('age', 55),
                            help=FEATURE_HELP['age'])
            sex = st.selectbox('Sex', ['Male', 'Female'],
                               index=['Male', 'Female'].index(sample.get('sex', 'Male')),
                               help=FEATURE_HELP['sex'])
            cp_opts = ['0 - Typical Angina', '1 - Atypical Angina',
                       '2 - Non-anginal Pain', '3 - Asymptomatic']
            cp = st.selectbox('Chest Pain Type', cp_opts,
                              index=cp_opts.index(sample['cp']) if sample.get('cp') in cp_opts else 0,
                              help=FEATURE_HELP['cp'])
            trestbps = st.slider('Resting Blood Pressure (mm Hg)', 80, 220,
                                 sample.get('trestbps', 130),
                                 help=FEATURE_HELP['trestbps'])
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'],
                               index=['No', 'Yes'].index(sample.get('fbs', 'No')),
                               help=FEATURE_HELP['fbs'])

        with col2:
            st.markdown('**Cardiac Measurements**')
            chol = st.slider('Serum Cholesterol (mg/dl)', 100, 600,
                             sample.get('chol', 245),
                             help=FEATURE_HELP['chol'])
            restecg_opts = ['0 - Normal', '1 - ST-T Wave Abnormality',
                            '2 - Left Ventricular Hypertrophy']
            restecg = st.selectbox('Resting ECG', restecg_opts,
                                   index=restecg_opts.index(sample['restecg']) if sample.get('restecg') in restecg_opts else 0,
                                   help=FEATURE_HELP['restecg'])
            thalach = st.slider('Max Heart Rate Achieved', 60, 220,
                                sample.get('thalach', 150),
                                help=FEATURE_HELP['thalach'])
            exang = st.selectbox('Exercise-Induced Angina', ['No', 'Yes'],
                                 index=['No', 'Yes'].index(sample.get('exang', 'No')),
                                 help=FEATURE_HELP['exang'])

        with col3:
            st.markdown('**Exercise Test Results**')
            oldpeak = st.number_input('ST Depression (Oldpeak)', 0.0, 7.0,
                                      float(sample.get('oldpeak', 1.0)), step=0.1,
                                      help=FEATURE_HELP['oldpeak'])
            slope_opts = ['0 - Upsloping', '1 - Flat', '2 - Downsloping']
            slope = st.selectbox('Slope of Peak Exercise ST', slope_opts,
                                 index=slope_opts.index(sample['slope']) if sample.get('slope') in slope_opts else 0,
                                 help=FEATURE_HELP['slope'])
            ca = st.selectbox('Major Vessels Colored by Fluoroscopy',
                              ['0', '1', '2', '3'],
                              index=['0', '1', '2', '3'].index(sample.get('ca', '0')),
                              help=FEATURE_HELP['ca'])
            thal_opts = ['3 - Normal', '6 - Fixed Defect', '7 - Reversible Defect']
            thal = st.selectbox('Thalassemia', thal_opts,
                                index=thal_opts.index(sample['thal']) if sample.get('thal') in thal_opts else 0,
                                help=FEATURE_HELP['thal'])

        submitted = st.form_submit_button('Predict', use_container_width=True,
                                          type='primary')

    if submitted:
        with st.spinner('Running prediction...'):
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
            prob_disease = probabilities[1]
            prob_healthy = probabilities[0]

        st.divider()

        if prediction == 1:
            st.error('### Heart Disease Detected')
            st.write('The model predicts heart disease is **likely present**. '
                     'Please consult a medical professional.')
        else:
            st.success('### No Heart Disease Detected')
            st.write('The model predicts heart disease is **unlikely**. '
                     'Continue regular health check-ups.')

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric('Healthy Probability', f'{prob_healthy * 100:.1f}%')
        with c2:
            st.metric('Disease Probability', f'{prob_disease * 100:.1f}%')
        with c3:
            st.metric('Model Used', chosen)

        st.subheader('Risk Factor Analysis')
        risk_flags = []
        for feat, (lo, hi) in NORMAL_RANGES.items():
            val = raw.get(feat)
            if val is not None and (val < lo or val > hi):
                risk_flags.append((feat, val, lo, hi))

        if risk_flags:
            for feat, val, lo, hi in risk_flags:
                label = RISK_LABELS.get(feat, feat)
                direction = 'above' if val > hi else 'below'
                st.warning(
                    f'**{label}**: {val} is {direction} normal range ({lo}\u2013{hi})')
        else:
            st.info('All key indicators are within normal ranges.')

        st.caption(
            '*This tool is for educational/research purposes only '
            'and is not a substitute for professional medical advice.*')


# ===================================================================
# PAGE 2 – Model Comparison
# ===================================================================
def page_compare():
    st.title('Model Comparison')
    st.write('Side-by-side performance of all four models on the UCI Heart Disease test set.')

    if not os.path.exists(RESULTS_CSV):
        st.error('Results not found. Run `python main.py` first.')
        return

    df = pd.read_csv(RESULTS_CSV)
    metric_cols = [c for c in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
                   if c in df.columns]

    best_idx = df['Accuracy'].idxmax()
    best_name = df.loc[best_idx, 'Model']
    best_acc = df.loc[best_idx, 'Accuracy']
    auc_str = (f', **{df.loc[best_idx, "AUC"]*100:.2f}%** AUC'
               if 'AUC' in df.columns else '')
    st.success(
        f'**Best model:** {best_name} \u2014 '
        f'**{best_acc*100:.2f}%** accuracy{auc_str}')

    # --- Metrics table ---
    st.subheader('Single-Split Metrics')
    disp = df.copy()
    for c in metric_cols:
        disp[c] = disp[c].apply(lambda v: f'{v * 100:.2f}%')
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # --- Bar chart ---
    st.subheader('Performance Chart')
    fig = go.Figure()
    for mc in metric_cols:
        fig.add_trace(go.Bar(
            name=mc, x=df['Model'], y=df[mc],
            text=df[mc].apply(lambda v: f'{v*100:.1f}%'),
            textposition='outside',
        ))
    fig.update_layout(
        barmode='group', yaxis_title='Score',
        yaxis_range=[0, 1.08], height=500,
        legend_title='Metric', margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- ROC curves ---
    if os.path.exists(ROC_FILE):
        st.subheader('ROC Curves')
        roc_data = joblib.load(ROC_FILE)
        fig_roc = go.Figure()
        for name, (fpr, tpr, auc) in roc_data.items():
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f'{name} (AUC={auc:.3f})'))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            name='Random', line=dict(color='gray', dash='dash')))
        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500, margin=dict(t=30))
        st.plotly_chart(fig_roc, use_container_width=True)

    # --- Cross-validation ---
    if os.path.exists(CV_CSV):
        st.subheader('5-Fold Cross-Validation Results')
        cv = pd.read_csv(CV_CSV)
        cv_display = cv[['Model']].copy()
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']:
            mcol = f'{metric}_mean'
            scol = f'{metric}_std'
            if mcol in cv.columns and scol in cv.columns:
                cv_display[metric] = cv.apply(
                    lambda r, m=mcol, s=scol: f"{r[m]*100:.2f}% \u00B1 {r[s]*100:.2f}%",
                    axis=1)
        st.dataframe(cv_display, use_container_width=True, hide_index=True)

    with st.expander('What do these metrics mean?'):
        st.markdown(
            '- **Accuracy** \u2014 proportion of all predictions that are correct.\n'
            '- **Precision** \u2014 of predicted positives, how many are truly positive.\n'
            '- **Recall** \u2014 of actual positives, how many were detected.\n'
            '- **F1-Score** \u2014 harmonic mean of precision and recall.\n'
            '- **AUC** \u2014 area under the ROC curve; measures discrimination ability '
            '(1.0 = perfect, 0.5 = random).')


# ===================================================================
# PAGE 3 – About / Methodology
# ===================================================================
def page_about():
    st.title('About This Project')

    st.markdown(
        '> *An Improved Heart Disease Prediction System Using '
        'Genetic Algorithm and Swarm-Optimized Artificial Neural Network*')

    st.write(
        'This system predicts heart disease from patient clinical data using '
        'an ANN whose hyperparameters are optimized by bio-inspired algorithms '
        '(GA and PSO). Built as a final-year research project at '
        '**Redeemer\'s University, Ede**.')

    st.subheader('Methodology')

    st.markdown("""
**1. Data Collection & Preprocessing**

Combined **four UCI Heart Disease datasets** (Cleveland, Hungarian, Switzerland,
VA Long Beach) totalling **~920 patient records** with 13 clinical features.
Missing values imputed, duplicates removed, features Z-score normalised,
data split 80/20 stratified.

**2. Baseline ANN**

A standard Multi-Layer Perceptron (2 hidden layers: 100 & 50 neurons,
ReLU activation, Adam optimizer, early stopping). This is the **reference**
against which optimized models are compared.

**3. GA Optimization (DEAP)**

A Genetic Algorithm evolves a population of chromosomes encoding
*hidden-layer sizes* and *learning rate*. Tournament selection, two-point
crossover, and Gaussian mutation drive the search. Fitness = validation accuracy.

**4. PSO Optimization (PySwarms)**

Particle Swarm Optimization positions particles in a 2D space encoding
*learning rate* and *L2 regularization strength*. Particles converge toward the
global best via social and cognitive acceleration.

**5. Hybrid GA \u2192 PSO \u2192 ANN**

Two-stage hybrid: GA finds the best architecture, PSO fine-tunes the training
parameters for that architecture, and a final ANN is trained with all optimized
hyperparameters.
""")

    st.subheader('Evaluation')
    c1, c2 = st.columns(2)
    with c1:
        st.write(
            '**Single-split evaluation** \u2014 all four models evaluated on the same '
            '20% held-out stratified test set using Accuracy, Precision, Recall, '
            'F1-Score, and ROC AUC.')
    with c2:
        st.write(
            '**5-Fold Stratified Cross-Validation** \u2014 each model assessed across '
            'five independent folds to confirm performance is consistent.')

    st.subheader('Dataset Features')
    features_df = pd.DataFrame({
        '#': range(1, 14),
        'Feature': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
        'Description': [
            'Age in years',
            '1 = male, 0 = female',
            'Chest pain type (0\u20133)',
            'Resting blood pressure (mm Hg)',
            'Serum cholesterol (mg/dl)',
            'Fasting blood sugar > 120 mg/dl',
            'Resting ECG results (0\u20132)',
            'Maximum heart rate achieved',
            'Exercise-induced angina',
            'ST depression induced by exercise',
            'Slope of peak exercise ST segment',
            'Major vessels (0\u20133) coloured by fluoroscopy',
            'Thalassemia (3=normal, 6=fixed, 7=reversible)',
        ],
    })
    st.dataframe(features_df, use_container_width=True, hide_index=True)

    st.subheader('Technology Stack')
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.markdown(
            '**Machine Learning**\n'
            '- scikit-learn (MLPClassifier)\n'
            '- DEAP (Genetic Algorithm)\n'
            '- PySwarms (PSO)')
    with tc2:
        st.markdown(
            '**Web Interface**\n'
            '- Streamlit\n'
            '- Plotly (interactive charts)')
    with tc3:
        st.markdown(
            '**Data & Utilities**\n'
            '- pandas / NumPy\n'
            '- joblib (model persistence)\n'
            '- UCI ML Repository')

    st.divider()
    st.markdown(
        '**Authors**: Ogundana Moyinolawa, Adebayo Oluwatimilehin, '
        'Okorie Samuel, Adewale Olukolade  \n'
        '**Supervisor**: Dr. T. O. Ojewumi  \n'
        '*Redeemer\'s University, Ede, Osun State \u2014 2025/2026*')


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == 'Predict Heart Disease':
    page_predict()
elif page == 'Model Comparison':
    page_compare()
else:
    page_about()
