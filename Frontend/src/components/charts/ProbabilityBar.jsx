import styled from 'styled-components';

const Track = styled.div`
  width: 100%;
  height: 10px;
  border-radius: 999px;
  background: #f0f0f0;
  overflow: hidden;
`;

const Fill = styled.div`
  height: 100%;
  width: ${({ $value }) => `${Math.max(0, Math.min(100, $value))}%`};
  background: ${({ $color, theme }) => $color || theme.colors.brand[500]};
  transition: width ${({ theme }) => theme.transitions.default};
`;

const Row = styled.div`
  display: grid;
  gap: 8px;
`;

const Heading = styled.div`
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: 12px;
`;

export const ProbabilityBar = ({ label, value, color }) => (
  <Row>
    <Heading>
      <span>{label}</span>
      <span>{value.toFixed(2)}%</span>
    </Heading>
    <Track>
      <Fill $value={value} $color={color} />
    </Track>
  </Row>
);

export default ProbabilityBar;
