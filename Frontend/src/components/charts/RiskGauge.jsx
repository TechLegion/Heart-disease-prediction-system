import styled from 'styled-components';

const Wrap = styled.div`
  width: 100%;
  max-width: 280px;
  aspect-ratio: 1;
  margin: 0 auto;
  position: relative;
`;

const Center = styled.div`
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  text-align: center;
`;

const Percent = styled.div`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 32px;
  line-height: 1;
`;

const Label = styled.div`
  color: ${({ theme }) => theme.colors.textSecondary};
  margin-top: 8px;
  font-size: 12px;
`;

export const RiskGauge = ({ value = 0 }) => {
  const size = 240;
  const stroke = 16;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const normalized = Math.max(0, Math.min(100, value));
  const offset = circumference - (normalized / 100) * circumference;

  return (
    <Wrap>
      <svg width="100%" height="100%" viewBox={`0 0 ${size} ${size}`}>
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="#f0f0f0" strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#e8734a"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <Center>
        <div>
          <Percent>{normalized.toFixed(1)}%</Percent>
          <Label>Disease probability</Label>
        </div>
      </Center>
    </Wrap>
  );
};

export default RiskGauge;
