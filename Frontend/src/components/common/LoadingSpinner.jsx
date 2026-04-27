import styled, { keyframes } from 'styled-components';

const spin = keyframes`
  to { transform: rotate(360deg); }
`;

const Ring = styled.div`
  width: ${({ size = 18 }) => `${size}px`};
  height: ${({ size = 18 }) => `${size}px`};
  border-radius: 999px;
  border: 2px solid rgba(0, 0, 0, 0.12);
  border-top-color: ${({ theme }) => theme.colors.orange};
  animation: ${spin} 0.8s linear infinite;
`;

export const LoadingSpinner = (props) => <Ring {...props} />;

export default LoadingSpinner;
