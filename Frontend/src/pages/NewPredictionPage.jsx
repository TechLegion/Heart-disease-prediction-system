import { useMemo, useState } from 'react';
import styled, { keyframes } from 'styled-components';
import { AlertTriangle, ArrowLeft, ArrowRight, CheckCircle2 } from 'lucide-react';
import { useLocation, useNavigate } from 'react-router-dom';
import Card from '../components/common/Card';
import { Input } from '../components/common/Input';
import { Select } from '../components/common/Select';
import { Button } from '../components/common/Button';
import Badge from '../components/common/Badge';
import { Skeleton } from '../components/common/Skeleton';
import { ProbabilityBar } from '../components/charts/ProbabilityBar';
import { RiskGauge } from '../components/charts/RiskGauge';
import { createPatientRecord } from '../services/patients';
import { predictHeartDisease } from '../services/api';
import { useAuth } from '../hooks/useAuth';
import StepIndicator from '../components/common/StepIndicator';

const Wrapper = styled.div`
  display: grid;
  gap: 14px;
`;

const StepCard = styled(Card)`
  padding: 16px;
`;

const Panel = styled(Card)`
  padding: 16px;
  position: relative;
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;

  @media (max-width: 980px) {
    grid-template-columns: 1fr;
  }
`;

const FieldCard = styled(Card)`
  padding: 14px;
`;

const ActionRow = styled.div`
  margin-top: 14px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
`;

const SexToggle = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
`;

const SexButton = styled.button`
  height: 44px;
  border-radius: ${({ theme }) => theme.radius.sm};
  border: 1px solid ${({ $active, theme }) => ($active ? theme.colors.orange : '#ebebeb')};
  background: ${({ $active, theme }) => ($active ? theme.colors.orange : '#fff')};
  color: ${({ $active }) => ($active ? '#fff' : '#888')};
`;

const fade = keyframes`
  0% { opacity: 0.4; }
  50% { opacity: 0.9; }
  100% { opacity: 0.4; }
`;

const LoadingOverlay = styled.div`
  position: absolute;
  inset: 0;
  border-radius: ${({ theme }) => theme.radius.lg};
  background: rgba(255, 255, 255, 0.86);
  display: grid;
  place-items: center;
  gap: 10px;
`;

const EcgLine = styled.div`
  width: 240px;
  height: 2px;
  background: linear-gradient(90deg, transparent, #e8734a, transparent);
  animation: ${fade} 1.4s infinite;
`;

const ResultBanner = styled.div`
  border-radius: ${({ theme }) => theme.radius.lg};
  color: #fff;
  padding: 16px;
  background: ${({ $detected }) => ($detected ? 'linear-gradient(135deg, #e8734a, #e8536e)' : 'linear-gradient(135deg, #3ab5a0, #2d9e8e)')};
`;

const Results = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;

  @media (max-width: 980px) {
    grid-template-columns: 1fr;
  }
`;

const RiskList = styled.div`
  display: grid;
  gap: 10px;
`;

const RiskItem = styled(Card)`
  padding: 12px;
`;

const twoStepFieldsLeft = [
  { key: 'cp', label: 'Chest Pain Type', options: [
    { value: '0 - Typical Angina', label: '0 - Typical Angina' },
    { value: '1 - Atypical Angina', label: '1 - Atypical Angina' },
    { value: '2 - Non-Anginal Pain', label: '2 - Non-Anginal Pain' },
    { value: '3 - Asymptomatic', label: '3 - Asymptomatic' },
  ], helperText: 'Normal range by symptom category.' },
  { key: 'trestbps', label: 'Resting Blood Pressure', type: 'number', helperText: 'Normal range: 90-140 mmHg' },
  { key: 'chol', label: 'Cholesterol', type: 'number', helperText: 'Normal range: 125-200 mg/dl' },
  { key: 'fbs', label: 'Fasting Blood Sugar', options: [
    { value: 'Yes', label: 'Yes' },
    { value: 'No', label: 'No' },
  ], helperText: 'Above 120 mg/dl threshold.' },
  { key: 'restecg', label: 'Resting ECG', options: [
    { value: '0 - Normal', label: '0 - Normal' },
    { value: '1 - ST-T Wave Abnormality', label: '1 - ST-T Wave Abnormality' },
    { value: '2 - Left Ventricular Hypertrophy', label: '2 - Left Ventricular Hypertrophy' },
  ], helperText: 'Clinical ECG interpretation.' },
  { key: 'thalach', label: 'Max Heart Rate', type: 'number', helperText: 'Unit: bpm' },
];

const twoStepFieldsRight = [
  { key: 'exang', label: 'Exercise Angina', options: [
    { value: 'Yes', label: 'Yes' },
    { value: 'No', label: 'No' },
  ], helperText: 'Exercise-induced chest pain.' },
  { key: 'oldpeak', label: 'ST Depression', type: 'number', step: '0.1', helperText: 'Decimal values allowed' },
  { key: 'slope', label: 'ST Slope', options: [
    { value: '0 - Upsloping', label: '0 - Upsloping' },
    { value: '1 - Flat', label: '1 - Flat' },
    { value: '2 - Downsloping', label: '2 - Downsloping' },
  ], helperText: 'Slope type' },
  { key: 'ca', label: 'Major Vessels (CA)', options: [
    { value: '0', label: '0' },
    { value: '1', label: '1' },
    { value: '2', label: '2' },
    { value: '3', label: '3' },
  ], helperText: 'Fluoroscopy vessels count' },
  { key: 'thal', label: 'Thalassemia', options: [
    { value: '3 - Normal', label: '3 - Normal' },
    { value: '6 - Fixed Defect', label: '6 - Fixed Defect' },
    { value: '7 - Reversible Defect', label: '7 - Reversible Defect' },
  ], helperText: 'Stress-test thal class' },
];

const emptyForm = {
  patientName: '',
  age: '',
  sex: '',
  cp: '',
  trestbps: '',
  chol: '',
  fbs: '',
  restecg: '',
  thalach: '',
  exang: '',
  oldpeak: '',
  slope: '',
  ca: '',
  thal: '',
};

export default function NewPredictionPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [step, setStep] = useState(1);
  const [form, setForm] = useState(() => ({ ...emptyForm, ...(location.state?.prefill || {}) }));
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);

  const clinicalPatient = useMemo(() => ({
    age: Number(form.age),
    sex: form.sex,
    cp: form.cp,
    trestbps: Number(form.trestbps),
    chol: Number(form.chol),
    fbs: form.fbs,
    restecg: form.restecg,
    thalach: Number(form.thalach),
    exang: form.exang,
    oldpeak: parseFloat(form.oldpeak),
    slope: form.slope,
    ca: String(form.ca),
    thal: form.thal,
  }), [form]);

  const onChange = (event) => {
    const { name, value } = event.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const runPrediction = async () => {
    setError('');
    setLoading(true);
    try {
      const response = await predictHeartDisease(clinicalPatient);
      setPrediction(response);
      setStep(3);
    } catch (submissionError) {
      setError(submissionError.message || 'Prediction failed. Please check your inputs and try again.');
    } finally {
      setLoading(false);
    }
  };

  const saveRecord = async () => {
    if (!prediction || !user) return;
    setSaving(true);
    try {
      const id = await createPatientRecord({
        name: form.patientName,
        uid: user.uid,
        patientData: clinicalPatient,
        ...prediction,
      });
      navigate(`/patients/${id}`);
    } finally {
      setSaving(false);
    }
  };

  const reset = () => {
    setForm({ ...emptyForm });
    setPrediction(null);
    setStep(1);
  };

  return (
    <Wrapper>
      <StepCard>
        <StepIndicator
          currentStep={step}
          steps={[
            { title: 'Patient Info' },
            { title: 'Clinical Data' },
            { title: 'Results' },
          ]}
        />
      </StepCard>

      {step === 1 ? (
        <Panel>
          <Grid>
            <Input label="Patient Name" name="patientName" value={form.patientName} onChange={onChange} />
            <Input label="Age" type="number" name="age" value={form.age} onChange={onChange} />
          </Grid>
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 11, letterSpacing: '0.05em', textTransform: 'uppercase', color: '#999', marginBottom: 6 }}>Sex</div>
            <SexToggle>
              <SexButton $active={form.sex === 'Male'} type="button" onClick={() => setForm((prev) => ({ ...prev, sex: 'Male' }))}>Male</SexButton>
              <SexButton $active={form.sex === 'Female'} type="button" onClick={() => setForm((prev) => ({ ...prev, sex: 'Female' }))}>Female</SexButton>
            </SexToggle>
          </div>
          <ActionRow>
            <div />
            <Button onClick={() => setStep(2)} disabled={!form.patientName || !form.age || !form.sex}>Continue <ArrowRight size={16} /></Button>
          </ActionRow>
        </Panel>
      ) : null}

      {step === 2 ? (
        <Panel>
          <Grid>
            <div style={{ display: 'grid', gap: 10 }}>
              {twoStepFieldsLeft.map((field) => (
                <FieldCard key={field.key}>
                  {field.options ? (
                    <Select label={field.label} name={field.key} value={form[field.key]} onChange={onChange} options={field.options} helperText={field.helperText} />
                  ) : (
                    <Input label={field.label} type={field.type || 'text'} step={field.step} name={field.key} value={form[field.key]} onChange={onChange} helperText={field.helperText} />
                  )}
                </FieldCard>
              ))}
            </div>
            <div style={{ display: 'grid', gap: 10 }}>
              {twoStepFieldsRight.map((field) => (
                <FieldCard key={field.key}>
                  {field.options ? (
                    <Select label={field.label} name={field.key} value={form[field.key]} onChange={onChange} options={field.options} helperText={field.helperText} />
                  ) : (
                    <Input label={field.label} type={field.type || 'text'} step={field.step} name={field.key} value={form[field.key]} onChange={onChange} helperText={field.helperText} />
                  )}
                </FieldCard>
              ))}
            </div>
          </Grid>
          {error ? <div style={{ color: '#e8536e', marginTop: 10, fontSize: 12 }}>{error}</div> : null}
          <ActionRow>
            <Button variant="ghost" onClick={() => setStep(1)}><ArrowLeft size={16} /> Back</Button>
            <Button loading={loading} onClick={runPrediction}>Run Prediction <ArrowRight size={16} /></Button>
          </ActionRow>
          {loading ? (
            <LoadingOverlay>
              <Skeleton width="240px" height="180px" />
              <EcgLine />
              <div style={{ color: '#888', fontSize: 12 }}>Analyzing patient data...</div>
            </LoadingOverlay>
          ) : null}
        </Panel>
      ) : null}

      {step === 3 && prediction ? (
        <>
          <ResultBanner $detected={Number(prediction.prediction) === 1}>
            <div style={{ fontSize: 18, fontWeight: 700 }}>{Number(prediction.prediction) === 1 ? 'Heart Disease Detected' : 'No Heart Disease Detected'}</div>
            <div style={{ opacity: 0.7, fontSize: 12, marginTop: 2 }}>{prediction.model_name}</div>
          </ResultBanner>
          <Results>
            <div style={{ display: 'grid', gap: 12 }}>
              <Card style={{ padding: 14 }}>
                <RiskGauge value={Number(prediction.probability_disease || 0) * 100} />
              </Card>
              <Card style={{ padding: 14, display: 'grid', gap: 10 }}>
                <ProbabilityBar label="Disease probability" value={Number(prediction.probability_disease || 0) * 100} color="#e8734a" />
                <ProbabilityBar label="Healthy probability" value={Number(prediction.probability_healthy || 0) * 100} color="#3ab5a0" />
              </Card>
            </div>
            <div>
              <h3 style={{ marginBottom: 10 }}>Elevated risk factors</h3>
              {prediction.risk_factors?.length ? (
                <RiskList>
                  {prediction.risk_factors.map((factor) => (
                    <RiskItem key={factor.feature}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <AlertTriangle color="#e8734a" size={16} />
                          <strong>{factor.label}</strong>
                        </div>
                        <div style={{ color: '#e8734a', fontFamily: 'Fira Code, monospace', fontSize: 18 }}>{factor.value}</div>
                      </div>
                      <div style={{ marginTop: 4, color: '#888', fontSize: 12 }}>Normal: {factor.low}-{factor.high}</div>
                      <div style={{ marginTop: 8 }}><Badge variant="risk">{factor.direction === 'above' ? 'Above Normal' : 'Below Normal'}</Badge></div>
                    </RiskItem>
                  ))}
                </RiskList>
              ) : (
                <Card style={{ padding: 14, background: '#e8f7f5', color: '#3ab5a0' }}>No elevated risk factors detected</Card>
              )}
            </div>
          </Results>
          <ActionRow>
            <Button loading={saving} onClick={saveRecord}><CheckCircle2 size={16} /> Save Patient Record</Button>
            <Button variant="ghost" onClick={reset}>Start New Prediction</Button>
          </ActionRow>
        </>
      ) : null}
    </Wrapper>
  );
}
