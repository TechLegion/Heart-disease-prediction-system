import styled from 'styled-components';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { LayoutGrid, PlusCircle, Users } from 'lucide-react';
import Sidebar from './Sidebar';
import DarkCard from '../common/DarkCard';
import { Badge } from '../common/Badge';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { useMemo } from 'react';
import { usePatients } from '../../hooks/usePatients';
import { formatShortDate } from '../../utils/formatters';

const PageWrapper = styled.div`
  min-height: 100vh;
  background: ${({ theme }) => theme.colors.pageBg};
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: 24px;

  @media (max-width: 767px) {
    padding: 0;
  }
`;

const Shell = styled.div`
  display: flex;
  width: 100%;
  max-width: 1400px;
  min-height: calc(100vh - 48px);
  border-radius: ${({ theme }) => theme.radius.xl};
  overflow: hidden;
  box-shadow: ${({ theme }) => theme.shadows.shell};

  @media (max-width: 767px) {
    min-height: 100vh;
    border-radius: 0;
  }
`;

const MainContent = styled.main`
  flex: 1;
  background: ${({ theme }) => theme.colors.shellBg};
  padding: 28px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 20px;

  @media (max-width: 767px) {
    padding: 16px 16px 76px;
  }
`;

const DarkPanel = styled.aside`
  width: 280px;
  background: ${({ theme }) => theme.colors.darkPanel};
  padding: 24px 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  flex-shrink: 0;
  overflow-y: auto;

  @media (max-width: 767px) {
    position: fixed;
    inset: auto 0 0 0;
    width: auto;
    max-height: 70vh;
    border-radius: 18px 18px 0 0;
    z-index: 40;
  }
`;

const MobileTabs = styled.nav`
  display: none;

  @media (max-width: 767px) {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    position: fixed;
    left: 12px;
    right: 12px;
    bottom: 12px;
    border-radius: 14px;
    background: ${({ theme }) => theme.colors.sidebarBg};
    padding: 8px;
    z-index: 30;
  }
`;

const TabBtn = styled.button`
  height: 34px;
  border-radius: 10px;
  border: 0;
  display: grid;
  place-items: center;
  color: ${({ $active }) => ($active ? '#fff' : 'rgba(255,255,255,0.4)')};
  background: ${({ $active, theme }) => ($active ? theme.colors.orange : 'transparent')};
`;

const SectionTitle = styled.h4`
  color: ${({ theme }) => theme.colors.darkTextPrimary};
  font-size: 14px;
`;

const Muted = styled.div`
  color: ${({ theme }) => theme.colors.darkTextSecondary};
  font-size: 10px;
`;

const MarkerRow = styled.div`
  display: grid;
  gap: 8px;
`;

function DashboardAnalyticsPanel({ patients }) {
  const latest = patients[0];
  const detected = patients.filter((item) => Number(item.prediction) === 1).length;
  const healthy = patients.length - detected;
  const pieData = [
    { name: 'Detected', value: detected, color: '#e8734a' },
    { name: 'Healthy', value: healthy, color: '#3ab5a0' },
  ];

  return (
    <>
      <div>
        <SectionTitle>Risk analysis</SectionTitle>
        <Muted>Last updated 5 min ago</Muted>
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <Badge variant="detected">All</Badge>
        <Badge variant="neutral">Detected</Badge>
        <Badge variant="neutral">Healthy</Badge>
      </div>
      <DarkCard>
        <Muted>Disease probability</Muted>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8 }}>
          <input value={latest?.name || 'No patient'} readOnly style={{ background: '#151525', border: 0, borderRadius: 8, color: '#fff', padding: '8px 10px', fontSize: 11 }} />
          <input value={latest?.patientData?.age || '--'} readOnly style={{ background: '#151525', border: 0, borderRadius: 8, color: '#fff', padding: '8px 10px', fontSize: 11 }} />
        </div>
        <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <strong style={{ color: '#fff', fontSize: 20 }}>{((latest?.probability_disease || 0) * 100).toFixed(1)}%</strong>
          <Badge variant={Number(latest?.prediction) === 1 ? 'risk' : 'healthy'}>{latest?.label || 'N/A'}</Badge>
        </div>
        <div style={{ marginTop: 10, height: 6, borderRadius: 999, background: 'linear-gradient(90deg,#3ab5a0,#f0c56a,#e8536e)', position: 'relative' }}>
          <span style={{ position: 'absolute', top: -3, left: `${Math.max(0, Math.min(100, (latest?.probability_disease || 0) * 100))}%`, width: 10, height: 10, borderRadius: '50%', background: '#fff', transform: 'translateX(-50%)' }} />
        </div>
      </DarkCard>
      <DarkCard>
        <div style={{ height: 160 }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={pieData} dataKey="value" innerRadius={45} outerRadius={65}>
                {pieData.map((entry) => <Cell key={entry.name} fill={entry.color} />)}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div style={{ textAlign: 'center', marginTop: -92, marginBottom: 62, color: '#fff', fontSize: 18, fontWeight: 700 }}>{patients.length}</div>
        <div style={{ display: 'grid', gap: 8 }}>
          <div style={{ color: '#fff', fontSize: 11 }}>● Detected</div>
          <div style={{ color: '#fff', fontSize: 11 }}>● Healthy</div>
        </div>
      </DarkCard>
      <div>
        <Muted style={{ textTransform: 'uppercase', letterSpacing: '0.08em' }}>Clinical markers</Muted>
        <MarkerRow>
          {[
            { label: 'Blood Pressure', value: latest?.patientData?.trestbps, unit: 'mmHg', high: 140 },
            { label: 'Cholesterol', value: latest?.patientData?.chol, unit: 'mg/dl', high: 200 },
            { label: 'Max Heart Rate', value: latest?.patientData?.thalach, unit: 'bpm', high: 100 },
          ].map((item) => {
            const elevated = Number(item.value || 0) > item.high;
            const color = elevated ? '#e8734a' : '#3ab5a0';
            return (
              <DarkCard key={item.label}>
                <Muted>{item.label}</Muted>
                <div style={{ color: '#fff', fontSize: 16, fontWeight: 700, marginTop: 4 }}>{item.value || '--'} <span style={{ fontSize: 10 }}>{item.unit}</span> <span style={{ color }}>↑</span></div>
                <div style={{ marginTop: 8, height: 3, background: '#1e1e30', borderRadius: 999 }}><div style={{ width: `${Math.min(100, (Number(item.value || 0) / (item.high * 1.4)) * 100)}%`, height: '100%', background: color }} /></div>
              </DarkCard>
            );
          })}
        </MarkerRow>
      </div>
    </>
  );
}

export const AppLayout = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { patients } = usePatients();
  const showDarkPanel = useMemo(() => location.pathname === '/dashboard' || /^\/patients\/.+/.test(location.pathname), [location.pathname]);
  const activePath = location.pathname.startsWith('/patients') ? '/patients' : location.pathname.startsWith('/predict') ? '/predict' : '/dashboard';

  return (
    <PageWrapper>
      <Shell>
        <Sidebar />
        <MainContent>
          <Outlet />
        </MainContent>
        {showDarkPanel ? (
          <DarkPanel>
            <DashboardAnalyticsPanel patients={patients} />
          </DarkPanel>
        ) : null}
      </Shell>
      <MobileTabs>
        <TabBtn $active={activePath === '/dashboard'} onClick={() => navigate('/dashboard')}><LayoutGrid size={16} /></TabBtn>
        <TabBtn $active={activePath === '/patients'} onClick={() => navigate('/patients')}><Users size={16} /></TabBtn>
        <TabBtn $active={activePath === '/predict'} onClick={() => navigate('/predict')}><PlusCircle size={16} /></TabBtn>
      </MobileTabs>
    </PageWrapper>
  );
};

export default AppLayout;
