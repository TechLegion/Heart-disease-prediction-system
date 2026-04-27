import styled from 'styled-components';
import { Area, AreaChart, ResponsiveContainer } from 'recharts';
import Card from '../common/Card';
import Badge from '../common/Badge';

const Wrap = styled(Card)`
  padding: 14px;
  display: grid;
  gap: 10px;
`;

const Top = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const IconBox = styled.div`
  width: 32px;
  height: 32px;
  border-radius: 8px;
  display: grid;
  place-items: center;
  color: ${({ $accent }) => $accent};
  background: ${({ $accentLight }) => $accentLight};
`;

const Label = styled.div`
  font-size: 11px;
  text-transform: uppercase;
  color: #999;
`;

const Value = styled.div`
  font-size: 22px;
  font-weight: 700;
  color: ${({ theme }) => theme.colors.textPrimary};

  small {
    font-size: 13px;
    color: ${({ theme }) => theme.colors.textMuted};
  }
`;

const Sparkline = styled.div`
  height: 40px;
`;

export const MetricCard = ({ icon, label, value, unit, badge, data, stroke, fill }) => (
  <Wrap>
    <Top>
      <IconBox $accent={stroke} $accentLight={fill}>{icon}</IconBox>
      {badge ? <Badge variant={badge.variant}>{badge.label}</Badge> : null}
    </Top>
    <Label>{label}</Label>
    <Value>{value}{unit ? <small> {unit}</small> : null}</Value>
    <Sparkline>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <Area type="monotone" dataKey="value" stroke={stroke} fill={fill} fillOpacity={0.08} strokeWidth={1.5} />
        </AreaChart>
      </ResponsiveContainer>
    </Sparkline>
  </Wrap>
);

export default MetricCard;
