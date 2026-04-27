import styled, { css, keyframes } from 'styled-components';
import { Check } from 'lucide-react';

const pulse = keyframes`
  0% { box-shadow: 0 0 0 0 rgba(232,115,74,0.3); }
  100% { box-shadow: 0 0 0 8px rgba(232,115,74,0); }
`;

const Wrapper = styled.div`
  display: grid;
  grid-template-columns: repeat(${({ steps }) => steps.length}, 1fr);
  gap: 0;
`;

const Item = styled.div`
  display: flex;
  align-items: center;
`;

const Node = styled.div`
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  font-size: 12px;
  font-weight: 600;
  margin-right: 10px;
  border: 2px solid;

  ${({ $state, theme }) => {
    if ($state === 'complete') {
      return css`
        background: ${theme.colors.orange};
        border-color: ${theme.colors.orange};
        color: #fff;
      `;
    }
    if ($state === 'active') {
      return css`
        background: #fff;
        border-color: ${theme.colors.orange};
        color: ${theme.colors.orange};
        animation: ${pulse} 1.4s infinite;
      `;
    }
    return css`
      background: #fff;
      border-color: #ddd;
      color: #aaa;
    `;
  }}
`;

const Label = styled.div`
  font-size: 12px;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const Line = styled.div`
  height: 2px;
  flex: 1;
  margin-right: 12px;
  background: ${({ $active, theme }) => ($active ? theme.colors.orange : '#ddd')};
`;

export const StepIndicator = ({ steps, currentStep }) => (
  <Wrapper steps={steps}>
    {steps.map((step, index) => {
      const number = index + 1;
      const state = number < currentStep ? 'complete' : number === currentStep ? 'active' : 'future';
      return (
        <Item key={step.title}>
          <Node $state={state}>{state === 'complete' ? <Check size={14} /> : number}</Node>
          <Label>{step.title}</Label>
          {index !== steps.length - 1 ? <Line $active={number < currentStep} /> : null}
        </Item>
      );
    })}
  </Wrapper>
);

export default StepIndicator;
