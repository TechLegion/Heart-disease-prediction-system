import { useEffect, useMemo, useState } from 'react';
import styled from 'styled-components';
import { useNavigate, useParams } from 'react-router-dom';
import { AlertTriangle, ArrowLeft } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import Card from '../components/common/Card';
import Badge from '../components/common/Badge';
import { Button } from '../components/common/Button';
import Modal from '../components/common/Modal';
import { Skeleton } from '../components/common/Skeleton';
import Avatar from '../components/common/Avatar';
import { ProbabilityBar } from '../components/charts/ProbabilityBar';
import { RiskGauge } from '../components/charts/RiskGauge';
import { fetchPatientById } from '../services/patients';
import { usePatients } from '../hooks/usePatients';
import { formatShortDate, getInitials, getSeverityLevel } from '../utils/formatters';

const Wrapper = styled.div`
  display: grid;
  gap: 14px;
`;

const BackLink = styled.button`
  border: 0;
  background: transparent;
  color: #888;
  font-size: 12px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
`;

const HeadCard = styled(Card)`
  padding: 16px;

  @media (max-width: 767px) {
    padding: 14px;
  }
`;

const Head = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;

  @media (max-width: 767px) {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }
`;

const Name = styled.h1`
  font-size: 22px;
  color: ${({ theme }) => theme.colors.textPrimary};
`;

const Meta = styled.div`
  font-size: 12px;
  color: #888;
  margin-top: 4px;
`;

const Content = styled.div`
  display: grid;
  grid-template-columns: 1.5fr 1fr;
  gap: 12px;

  @media (max-width: 1020px) {
    grid-template-columns: 1fr;
  }

  @media (max-width: 767px) {
    gap: 10px;
  }
`;

const Section = styled(Card)`
  padding: 14px;
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
`;

const Td = styled.td`
  padding: 10px;
  font-size: 12px;
  color: ${({ theme }) => theme.colors.textPrimary};
`;

const Key = styled(Td)`
  color: #888;
`;

const SeverityBar = styled.div`
  height: 6px;
  border-radius: 999px;
  background: linear-gradient(90deg, #3a8a3a, #f59e0b, #e8536e);
  margin-top: 6px;
  position: relative;
  overflow: hidden;

  @media (max-width: 767px) {
    margin-top: 4px;
  }
`;

const SeverityLabel = styled.div`
  font-size: 11px;
  color: ${({ theme }) => theme.colors.textSecondary};
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin-top: 6px;
  margin-bottom: 4px;

  @media (max-width: 767px) {
    font-size: 10px;
  }
`;

const SectionTitle = styled.h3`
  margin: 0 0 10px 0;
  font-size: 14px;
  font-weight: 600;
  color: ${({ theme }) => theme.colors.textPrimary};

  @media (max-width: 767px) {
    font-size: 13px;
    margin-bottom: 8px;
  }
`;

const RiskFactorCard = styled(Card)`
  padding: 12px;
  display: grid;
  gap: 6px;

  @media (max-width: 767px) {
    padding: 10px;
  }
`;

const RiskFactorRow = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
`;

const RiskFactorLabel = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
  color: ${({ theme }) => theme.colors.textPrimary};
`;

const RiskFactorValue = styled.div`
  font-family: 'Fira Code', monospace;
  color: ${({ theme }) => theme.colors.orange};
  font-weight: 600;

  @media (max-width: 767px) {
    font-size: 12px;
  }
`;

const RiskFactorNormal = styled.div`
  margin-top: 4px;
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: 12px;
`;

const EmptyState = styled(Card)`
  padding: 12px;
  background: ${({ theme }) => theme.colors.tealLight};
  color: ${({ theme }) => theme.colors.teal};
  text-align: center;
  font-size: 12px;
`;

const MetricRow = styled.div`
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 12px;

  @media (max-width: 1020px) {
    grid-template-columns: 1fr;
  }

  @media (max-width: 767px) {
    grid-template-columns: 1fr;
    gap: 10px;
  }
`;

const SeverityDot = styled.span`
  position: absolute;
  top: -2px;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #fff;
  transform: translateX(-50%);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
`;

export default function PatientDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { removePatient } = usePatients();
  const [patient, setPatient] = useState(null);
  const [loading, setLoading] = useState(true);
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => {
    let active = true;
    setLoading(true);
    fetchPatientById(id).then((record) => {
      if (active) {
        setPatient(record);
        setLoading(false);
      }
    });
    return () => {
      active = false;
    };
  }, [id]);

  const detected = Number(patient?.prediction) === 1;
  const severity = useMemo(() => getSeverityLevel(patient?.probability_disease || 0), [patient]);

  const features = useMemo(() => {
    if (!patient?.features_used) return [];
    return Object.entries(patient.features_used).map(([key, value]) => ({ key, value: Number(value) }));
  }, [patient]);

  const rows = useMemo(() => {
    if (!patient) return [];
    return [
      ['Chest Pain Type', patient.patientData?.cp],
      ['Resting Blood Pressure', `${patient.patientData?.trestbps} mmHg`],
      ['Cholesterol', `${patient.patientData?.chol} mg/dl`],
      ['Fasting Blood Sugar', patient.patientData?.fbs],
      ['Resting ECG', patient.patientData?.restecg],
      ['Max Heart Rate', patient.patientData?.thalach],
      ['Exercise Angina', patient.patientData?.exang],
      ['ST Depression', patient.patientData?.oldpeak],
      ['ST Slope', patient.patientData?.slope],
      ['Major Vessels', patient.patientData?.ca],
      ['Thalassemia', patient.patientData?.thal],
      ['Age', patient.patientData?.age],
      ['Sex', patient.patientData?.sex],
    ];
  }, [patient]);

  const onDelete = async () => {
    await removePatient(id);
    navigate('/patients');
  };

  if (loading) {
    return <Wrapper><Skeleton height="120px" /><Skeleton height="500px" /></Wrapper>;
  }

  if (!patient) {
    return <div>Patient record not found.</div>;
  }

  return (
    <Wrapper>
      <BackLink onClick={() => navigate('/patients')}><ArrowLeft size={14} /> Patients</BackLink>
      <HeadCard>
        <Head>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <Avatar initials={getInitials(patient.name)} size={64} variant={detected ? 'detected' : 'healthy'} />
            <div>
              <Name>{patient.name}</Name>
              <Meta>{patient.patientData?.age} · {patient.patientData?.sex} · {formatShortDate(patient.createdAt)}</Meta>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <Badge variant={detected ? 'detected' : 'healthy'} size="lg">{patient.label}</Badge>
            <Badge size="lg" variant={severity.level.toLowerCase()}>{severity.label}</Badge>
          </div>
        </Head>
      </HeadCard>

      <Content>
        <div style={{ display: 'grid', gap: 12 }}>
          <Section>
            <MetricRow>
              <Card style={{ padding: 14 }}>
                <RiskGauge value={Number(patient.probability_disease || 0) * 100} />
              </Card>
              <Card style={{ padding: 14, display: 'grid', gap: 10 }}>
                <ProbabilityBar label="Disease probability" value={Number(patient.probability_disease || 0) * 100} color="#e8734a" />
                <ProbabilityBar label="Healthy probability" value={Number(patient.probability_healthy || 0) * 100} color="#3ab5a0" />
                <div>
                  <SeverityLabel>Severity Level</SeverityLabel>
                  <SeverityBar>
                    <SeverityDot style={{ left: `${Math.max(0, Math.min(100, Number(patient.probability_disease || 0) * 100))}%` }} />
                  </SeverityBar>
                </div>
              </Card>
            </MetricRow>
          </Section>

          <Section>
            <SectionTitle>Elevated risk factors</SectionTitle>
            {patient.risk_factors?.length ? (
              <div style={{ display: 'grid', gap: 10 }}>
                {patient.risk_factors.map((factor) => (
                  <RiskFactorCard key={factor.feature}>
                    <RiskFactorRow>
                      <RiskFactorLabel>
                        <AlertTriangle size={16} color="#e8734a" />
                        {factor.label}
                      </RiskFactorLabel>
                      <RiskFactorValue>{factor.value}</RiskFactorValue>
                    </RiskFactorRow>
                    <RiskFactorNormal>Normal: {factor.low}-{factor.high}</RiskFactorNormal>
                  </RiskFactorCard>
                ))}
              </div>
            ) : (
              <EmptyState>No elevated risk factors detected</EmptyState>
            )}
          </Section>

          <Section>
            <SectionTitle>Clinical data submitted</SectionTitle>
            <Table>
              <tbody>
                {rows.map(([label, value], index) => (
                  <tr key={label} style={{ background: index % 2 ? '#fafafa' : '#fff' }}>
                    <Key>{label}</Key>
                    <Td style={{ fontWeight: 500 }}>{String(value ?? '--')}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </Section>
        </div>

        <div style={{ display: 'grid', gap: 12 }}>
          <Section>
            <SectionTitle>Model feature weights</SectionTitle>
            <div style={{ height: 320 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={features} layout="vertical">
                  <CartesianGrid stroke="#f1f1f1" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 10, fill: '#bbb' }} axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="key" tick={{ fontSize: 10, fill: '#999' }} axisLine={false} tickLine={false} width={78} />
                  <Tooltip />
                  <Bar dataKey="value" barSize={8} radius={[0, 4, 4, 0]}>
                    {features.map((entry) => <Cell key={entry.key} fill={entry.value > 0 ? '#e8734a' : '#3ab5a0'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Section>

          <Section style={{ background: '#1e1e30', color: '#fff' }}>
            <div style={{ fontSize: 11, color: '#888899' }}>Model</div>
            <div style={{ marginTop: 3 }}>Hybrid GA-PSO-ANN</div>
            <div style={{ fontSize: 11, color: '#888899', marginTop: 10 }}>Assessed</div>
            <div style={{ marginTop: 3 }}>{formatShortDate(patient.createdAt)}</div>
          </Section>
        </div>
      </Content>

      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <Button variant="danger" onClick={() => setConfirmDelete(true)}>Delete Record</Button>
        <Button onClick={() => navigate('/predict', { state: { prefill: { patientName: patient.name, ...patient.patientData } } })}>Re-run Prediction</Button>
      </div>

      <Modal
        open={confirmDelete}
        title="Delete record"
        danger
        confirmLabel="Delete"
        onClose={() => setConfirmDelete(false)}
        onConfirm={onDelete}
      >
        This will permanently remove the patient record.
      </Modal>
    </Wrapper>
  );
}
