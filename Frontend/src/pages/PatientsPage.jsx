import { useMemo } from 'react';
import styled from 'styled-components';
import { Search } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import Card from '../components/common/Card';
import Badge from '../components/common/Badge';
import { Skeleton } from '../components/common/Skeleton';
import { usePatients } from '../hooks/usePatients';
import { formatShortDate, getInitials, getSeverityLevel } from '../utils/formatters';
import Avatar from '../components/common/Avatar';

const Wrapper = styled.div`
  display: grid;
  gap: 14px;
`;

const Header = styled.div`
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 12px;

  @media (max-width: 900px) {
    grid-template-columns: 1fr;
  }
`;

const SearchBox = styled(Card)`
  height: 46px;
  border-radius: ${({ theme }) => theme.radius.md};
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 14px;

  input {
    flex: 1;
    border: 0;
    background: transparent;
    outline: none;
  }
`;

const Filters = styled.div`
  display: flex;
  gap: 8px;
`;

const Filter = styled.button`
  border: 0;
  border-radius: ${({ theme }) => theme.radius.pill};
  padding: 8px 14px;
  background: ${({ $active, theme }) => ($active ? theme.colors.orange : '#fff')};
  color: ${({ $active }) => ($active ? '#fff' : '#888')};
  box-shadow: ${({ theme }) => theme.shadows.card};
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;

  @media (max-width: 980px) {
    grid-template-columns: 1fr;
  }

  @media (max-width: 767px) {
    gap: 10px;
  }
`;

const PatientCard = styled(Card)`
  padding: 16px;

  @media (max-width: 767px) {
    padding: 14px;
  }
`;

const Top = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
`;

const Name = styled.div`
  font-size: 15px;
  font-weight: 600;
`;

const Meta = styled.div`
  font-size: 12px;
  color: ${({ theme }) => theme.colors.textSecondary};
  margin-top: 4px;
`;

const Divider = styled.div`
  height: 1px;
  background: #f1f1f1;
  margin: 12px 0;
`;

const Progress = styled.div`
  height: 4px;
  border-radius: ${({ theme }) => theme.radius.pill};
  background: #f0f0f0;
  overflow: hidden;
`;

const Fill = styled.div`
  width: ${({ $value }) => `${Math.max(0, Math.min(100, $value))}%`};
  height: 100%;
  background: ${({ $detected, theme }) => ($detected ? theme.colors.orange : theme.colors.teal)};
`;

const Empty = styled(Card)`
  padding: 34px;
  text-align: center;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

export default function PatientsPage() {
  const navigate = useNavigate();
  const { patients, loadingPatients, searchQuery, filter, dispatch } = usePatients();

  const filtered = useMemo(() => patients.filter((patient) => {
    const matchesSearch = patient.name?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter = filter === 'all' || (filter === 'detected' && Number(patient.prediction) === 1) || (filter === 'healthy' && Number(patient.prediction) === 0);
    return matchesSearch && matchesFilter;
  }), [patients, searchQuery, filter]);

  if (loadingPatients) {
    return <Wrapper><Skeleton height="46px" /><Grid>{[0, 1, 2, 3].map((i) => <Skeleton key={i} height="210px" />)}</Grid></Wrapper>;
  }

  return (
    <Wrapper>
      <Header>
        <SearchBox>
          <Search size={15} color="#888" />
          <input
            value={searchQuery}
            onChange={(event) => dispatch({ type: 'SET_SEARCH_QUERY', payload: event.target.value })}
            placeholder="Search patients"
          />
        </SearchBox>
        <Filters>
          {['all', 'detected', 'healthy'].map((item) => (
            <Filter key={item} $active={filter === item} onClick={() => dispatch({ type: 'SET_FILTER', payload: item })}>
              {item === 'all' ? 'All' : item === 'detected' ? 'Detected' : 'Healthy'}
            </Filter>
          ))}
        </Filters>
      </Header>

      {filtered.length === 0 ? (
        <Empty>No patients match your filter.</Empty>
      ) : (
        <Grid>
          {filtered.map((patient) => {
            const detected = Number(patient.prediction) === 1;
            const risk = Number(patient.probability_disease || 0) * 100;
            return (
              <PatientCard key={patient.id} $clickable onClick={() => navigate(`/patients/${patient.id}`)}>
                <Top>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <Avatar initials={getInitials(patient.name)} variant={detected ? 'detected' : 'healthy'} size={44} />
                    <div>
                      <Name>{patient.name}</Name>
                      <Meta>Age {patient.patientData?.age || '--'} · {patient.patientData?.sex || '--'}</Meta>
                    </div>
                  </div>
                </Top>
                <Divider />
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: detected ? '#e8734a' : '#3ab5a0' }} />
                    <Badge variant={detected ? 'detected' : 'healthy'}>{detected ? 'Detected' : 'Healthy'}</Badge>
                  </div>
                </div>
                <div style={{ fontSize: 12, color: '#888' }}>Risk</div>
                <Progress><Fill $value={risk} $detected={detected} /></Progress>
                <div style={{ marginTop: 6, fontSize: 12 }}>{risk.toFixed(1)}%</div>
                <div style={{ marginTop: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: 12, color: '#888' }}>{formatShortDate(patient.createdAt)}</span>
                    {(() => {
                      const sev = getSeverityLevel(patient.probability_disease || 0);
                      return <Badge size="sm" variant={sev.level.toLowerCase()}>{sev.label}</Badge>;
                    })()}
                  </div>
                  <span style={{ fontSize: 12, color: '#e8734a' }}>View →</span>
                </div>
              </PatientCard>
            );
          })}
        </Grid>
      )}
    </Wrapper>
  );
}
