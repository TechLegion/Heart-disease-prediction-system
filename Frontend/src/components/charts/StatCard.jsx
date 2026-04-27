import styled from 'styled-components';
import { ArrowDown, ArrowUp } from 'lucide-react';
import Card from '../common/Card';

const Wrap = styled(Card)`
  padding: 18px;
  display: grid;
  gap: 14px;
`;

const Label = styled.div`
  color: ${({ theme }) => theme.colors.text.secondary};
  font-size: 11px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
`;

const Value = styled.div`
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 28px;
  line-height: 1;
`;

const Meta = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  color: ${({ theme }) => theme.colors.text.secondary};
  font-size: 12px;
`;

export const StatCard = ({ label, value, description, trend = 'up', trendText = 'Static trend' }) => (
  <Wrap>
    <Label>{label}</Label>
    <Value>{value}</Value>
    <Meta>
      {trend === 'up' ? <ArrowUp size={14} color="#3ECF8E" /> : <ArrowDown size={14} color="#E74C3C" />}
      <span>{trendText}</span>
      <span>·</span>
      <span>{description}</span>
    </Meta>
  </Wrap>
);

export default StatCard;
