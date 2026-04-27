import styled, { css, keyframes } from 'styled-components';

const spin = keyframes`
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
`;

const buttonVariants = {
  primary: css`
    background: ${({ theme }) => theme.colors.orange};
    color: ${({ theme }) => theme.colors.cardBg};
    border-color: ${({ theme }) => theme.colors.orange};

    &:hover:not(:disabled) {
      background: ${({ theme }) => theme.colors.orangeDark};
      border-color: ${({ theme }) => theme.colors.orangeDark};
    }
  `,
  ghost: css`
    background: transparent;
    color: ${({ theme }) => theme.colors.orange};
    border-color: ${({ theme }) => theme.colors.orange};

    &:hover:not(:disabled) {
      background: ${({ theme }) => theme.colors.orangeLight};
    }
  `,
  danger: css`
    background: transparent;
    color: ${({ theme }) => theme.colors.danger};
    border-color: ${({ theme }) => theme.colors.danger};

    &:hover:not(:disabled) {
      background: ${({ theme }) => theme.colors.dangerBg};
    }
  `,
  icon: css`
    width: 36px;
    height: 36px;
    padding: 0;
    background: ${({ theme }) => theme.colors.cardBg};
    border-color: transparent;
    color: ${({ theme }) => theme.colors.textPrimary};
    box-shadow: ${({ theme }) => theme.shadows.card};

    &:hover:not(:disabled) {
      box-shadow: ${({ theme }) => theme.shadows.cardHover};
      transform: translateY(-1px);
    }
  `,
};

const buttonSizes = {
  sm: css`
    height: 36px;
    padding: 0 12px;
    font-size: 12px;
  `,
  md: css`
    height: 44px;
    padding: 0 16px;
    font-size: 13px;
  `,
  lg: css`
    height: 44px;
    padding: 0 18px;
    font-size: 14px;
  `,
};

const Spinner = styled.span`
  width: 14px;
  height: 14px;
  border-radius: 50%;
  border: 2px solid rgba(255, 255, 255, 0.2);
  border-top-color: currentColor;
  animation: ${spin} 0.8s linear infinite;
`;

const StyledButton = styled.button`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border: 1px solid;
  border-radius: ${({ theme }) => theme.radius.sm};
  font-weight: 600;
  transition: ${({ theme }) => theme.transitions.default};
  cursor: pointer;

  ${({ variant }) => buttonVariants[variant || 'primary']}
  ${({ size }) => buttonSizes[size || 'md']}

  &:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }
`;

export const Button = ({ children, loading, disabled, variant = 'primary', size = 'md', ...props }) => (
  <StyledButton variant={variant} size={size} disabled={disabled || loading} {...props}>
    {loading ? <Spinner /> : null}
    <span>{children}</span>
  </StyledButton>
);

export default Button;
