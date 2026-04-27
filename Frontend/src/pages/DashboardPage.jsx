import { useMemo } from 'react';
import styled from 'styled-components';
import { Bell, CalendarDays, HeartPulse, Percent, Search, Users } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import Card from '../components/common/Card';
import { Button } from '../components/common/Button';
import { Badge } from '../components/common/Badge';
import { Skeleton } from '../components/common/Skeleton';
import { usePatients } from '../hooks/usePatients';
import { formatShortDate, getInitials } from '../utils/formatters';
import MetricCard from '../components/charts/MetricCard';
import Avatar from '../components/common/Avatar';

const Wrapper = styled.div`
  display: grid;
  gap: 18px;

  @media (max-width: 767px) {
    gap: 14px;
  }
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;

  @media (max-width: 767px) {
    flex-direction: column;
    gap: 12px;
  }
`;

const Title = styled.h1`
  font-size: 22px;
  color: ${({ theme }) => theme.colors.textPrimary};

  @media (max-width: 767px) {
    font-size: 18px;
  }
`;

const Subtitle = styled.div`
  margin-top: 4px;
  font-size: 12px;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const HeaderActions = styled.div`
  display: flex;
  gap: 8px;

  @media (max-width: 767px) {
    width: 100%;
    justify-content: flex-end;
  }
`;

const Metrics = styled.div`
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;

  @media (max-width: 980px) {
    grid-template-columns: 1fr;
  }

  @media (max-width: 767px) {
    gap: 10px;
  }
`;

const ChartCard = styled(Card)`
  padding: 16px;

  @media (max-width: 767px) {
    padding: 14px;
  }
`;

const Row = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;

  @media (max-width: 767px) {
    flex-wrap: wrap;
    gap: 8px;
  }
`;

const SectionTitle = styled.h3`
  font-size: 14px;
  color: ${({ theme }) => theme.colors.textPrimary};
`;

const Selector = styled.button`
  border: 0;
  border-radius: ${({ theme }) => theme.radius.pill};
  background: ${({ theme }) => theme.colors.orangeLight};
  color: ${({ theme }) => theme.colors.orange};
  padding: 7px 12px;
  font-size: 11px;
`;

const AppointmentCard = styled(Card)`
  padding: 14px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
`;

const DateBadge = styled.div`
  background: #e8f4ff;
  color: #3a7bd5;
  border-radius: ${({ theme }) => theme.radius.sm};
  padding: 6px 10px;
  font-size: 11px;
  font-weight: 600;
`;

const TableCard = styled(Card)`
  padding: 16px;
  overflow: hidden;
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
`;

const Th = styled.th`
  text-align: left;
  font-size: 11px;
  text-transform: uppercase;
  color: #bbb;
  letter-spacing: 0.06em;
  padding: 12px 10px;
`;

const Td = styled.td`
  padding: 12px 10px;
  font-size: 12px;
  color: ${({ theme }) => theme.colors.textPrimary};
  vertical-align: middle;
`;

const RiskTrack = styled.div`
  height: 4px;
  background: #f0f0f0;
  border-radius: ${({ theme }) => theme.radius.pill};
  margin-top: 6px;
  overflow: hidden;
`;

const RiskFill = styled.div`
  height: 100%;
  width: ${({ $value }) => `${Math.max(0, Math.min(100, $value))}%`};
  background: ${({ $detected, theme }) => ($detected ? theme.colors.orange : theme.colors.teal)};
`;

const Empty = styled.div`
  padding: 22px;
  text-align: center;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const todayText = new Intl.DateTimeFormat('en-GB', { day: '2-digit', month: 'long', year: 'numeric' }).format(new Date());

export default function DashboardPage() {
  const { patients, loadingPatients } = usePatients();

  const stats = useMemo(() => {
    const total = patients.length;
    const detected = patients.filter((item) => Number(item.prediction) === 1).length;
    const rate = total ? (detected / total) * 100 : 0;
    return { total, detected, rate };
  }, [patients]);

  const sparkline = useMemo(() => {
    const days = Array.from({ length: 7 }, (_, index) => {
      const date = new Date();
      date.setDate(date.getDate() - (6 - index));
      return { key: date.toISOString().slice(0, 10), value: 0, detected: 0 };
    });
    const map = new Map(days.map((day) => [day.key, day]));
    patients.forEach((item) => {
      const date = item.createdAt?.toDate?.();
      if (!date) return;
      const key = date.toISOString().slice(0, 10);
      if (!map.has(key)) return;
      const entry = map.get(key);
      entry.value += 1;
      entry.detected += Number(item.prediction) === 1 ? 1 : 0;
    });
    return days;
  }, [patients]);

  const activity = useMemo(() => {
    const days = Array.from({ length: 14 }, (_, index) => {
      const date = new Date();
      date.setDate(date.getDate() - (13 - index));
      return {
        key: date.toISOString().slice(0, 10),
        date: date.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }),
        detected: 0,
        healthy: 0,
        highRisk: 0,
      };
    });
    const map = new Map(days.map((day) => [day.key, day]));
    patients.forEach((item) => {
      const date = item.createdAt?.toDate?.();
      if (!date) return;
      const key = date.toISOString().slice(0, 10);
      if (!map.has(key)) return;
      const entry = map.get(key);
      const detected = Number(item.prediction) === 1;
      if (detected) entry.detected += 1;
      else entry.healthy += 1;
      if (Number(item.probability_disease || 0) >= 0.75) entry.highRisk += 1;
    });
    return days;
  }, [patients]);

  const recent = useMemo(() => patients.slice(0, 8), [patients]);

  if (loadingPatients) {
    return (
      <Wrapper>
        <Skeleton height="50px" />
        <Metrics>{[0, 1, 2].map((i) => <Skeleton key={i} height="180px" />)}</Metrics>
        <Skeleton height="290px" />
        <Skeleton height="72px" />
        <Skeleton height="360px" />
      </Wrapper>
    );
  }

  return (
    <Wrapper>
      <Header>
        <div>
          <Title>Cardiac Overview</Title>
          <Subtitle>{todayText}</Subtitle>
        </div>
        <HeaderActions>
          <Button variant="icon"><Search size={16} /></Button>
          <Button variant="icon"><Bell size={16} /></Button>
        </HeaderActions>
      </Header>

      <Metrics>
        <MetricCard
          icon={<Users size={16} />}
          label="Total patients"
          value={stats.total}
          badge={{ variant: 'detected', label: 'Live' }}
          data={sparkline.map((item) => ({ value: item.value }))}
          stroke="#e8734a"
          fill="#fff3ee"
        />
        <MetricCard
          icon={<HeartPulse size={16} />}
          label="Detected"
          value={stats.detected}
          badge={{ variant: 'risk', label: 'At risk' }}
          data={sparkline.map((item) => ({ value: item.detected }))}
          stroke="#e8536e"
          fill="#fff0f3"
        />
        <MetricCard
          icon={<Percent size={16} />}
          label="Detection rate"
          value={stats.rate.toFixed(1)}
          unit="%"
          badge={{ variant: 'healthy', label: 'Stable' }}
          data={sparkline.map((item) => ({ value: item.value ? (item.detected / item.value) * 100 : 0 }))}
          stroke="#3ab5a0"
          fill="#e8f7f5"
        />
      </Metrics>

      <ChartCard>
        <Row>
          <SectionTitle>Prediction activity</SectionTitle>
          <Selector>This month</Selector>
        </Row>
        <div style={{ height: 240, marginTop: 10 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={activity}>
              <CartesianGrid stroke="#f5f5f5" vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 10, fill: '#bbb' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 10, fill: '#bbb' }} axisLine={false} tickLine={false} allowDecimals={false} />
              <Tooltip />
              <Legend iconType="circle" wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="detected" fill="#e8734a" radius={[4, 4, 0, 0]} barSize={7} />
              <Bar dataKey="healthy" fill="#3ab5a0" radius={[4, 4, 0, 0]} barSize={7} />
              <Bar dataKey="highRisk" fill="#e8536e" radius={[4, 4, 0, 0]} barSize={7} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      <AppointmentCard>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <DateBadge>Today 16:00</DateBadge>
          <span style={{ color: '#555', fontSize: 13 }}>Cardiology review with Dr. Lane for high-risk cohort follow-up.</span>
        </div>
        <CalendarDays size={16} color="#888" />
      </AppointmentCard>

      <TableCard>
        <Row><SectionTitle>Recent predictions</SectionTitle></Row>
        {recent.length === 0 ? (
          <Empty>No predictions yet.</Empty>
        ) : (
          <Table>
            <thead>
              <tr>
                <Th>Patient</Th>
                <Th>Age</Th>
                <Th>Sex</Th>
                <Th>Result</Th>
                <Th>Risk Score</Th>
                <Th>Date</Th>
                <Th>Action</Th>
              </tr>
            </thead>
            <tbody>
              {recent.map((item, index) => {
                const detected = Number(item.prediction) === 1;
                const score = Number(item.probability_disease || 0) * 100;
                return (
                  <tr key={item.id} style={{ background: index % 2 ? '#fafafa' : '#fff' }}>
                    <Td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <Avatar initials={getInitials(item.name)} size={30} variant={detected ? 'detected' : 'healthy'} />
                        <span>{item.name}</span>
                      </div>
                    </Td>
                    <Td>{item.patientData?.age || '--'}</Td>
                    <Td>{item.patientData?.sex || '--'}</Td>
                    <Td><Badge variant={detected ? 'detected' : 'healthy'}>{detected ? 'Detected' : 'Healthy'}</Badge></Td>
                    <Td>
                      <div>{score.toFixed(1)}%</div>
                      <RiskTrack><RiskFill $value={score} $detected={detected} /></RiskTrack>
                    </Td>
                    <Td>{formatShortDate(item.createdAt)}</Td>
                    <Td><a href={`/patients/${item.id}`} style={{ color: '#e8734a' }}>View</a></Td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        )}
      </TableCard>
    </Wrapper>
  );
}
