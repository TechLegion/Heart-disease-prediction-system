import styled, { keyframes } from 'styled-components';

const shimmer = keyframes`
  0% { background-position: 100% 0; }
  100% { background-position: -100% 0; }
`;

export const Skeleton = styled.div`
  border-radius: ${({ theme }) => theme.radius.sm};
  background: linear-gradient(90deg, #f0f0f0 25%, #fafafa 50%, #f0f0f0 75%);
  background-size: 200% 100%;
  animation: ${shimmer} 1.5s infinite;
  min-height: ${({ height = '16px' }) => height};
  width: ${({ width = '100%' }) => width};
`;

export default Skeleton;
