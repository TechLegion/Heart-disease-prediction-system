import styled, { css } from 'styled-components';

const badgeVariants = {
  detected: css`
    background: ${({ theme }) => theme.colors.orangeLight};
    color: ${({ theme }) => theme.colors.orange};
    border-color: rgba(232,115,74,0.3);
  `,
  healthy: css`
    background: ${({ theme }) => theme.colors.tealLight};
    color: ${({ theme }) => theme.colors.teal};
    border-color: rgba(58,181,160,0.3);
  `,
  risk: css`
    background: ${({ theme }) => theme.colors.pinkLight};
    color: ${({ theme }) => theme.colors.pink};
    border-color: rgba(232,83,110,0.3);
  `,
  normal: css`
    background: #eef7ee;
    color: #3a8a3a;
    border-color: rgba(58,138,58,0.3);
  `,
  warning: css`
    background: ${({ theme }) => theme.colors.orangeLight};
    color: ${({ theme }) => theme.colors.orange};
    border-color: rgba(232,115,74,0.3);
  `,
  neutral: css`
    background: #f4f4f4;
    color: ${({ theme }) => theme.colors.textSecondary};
    border-color: #e6e6e6;
  `,
};

const StyledBadge = styled.span`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: ${({ theme }) => theme.radius.pill};
  border: 1px solid;
  padding: ${({ size }) => (size === 'lg' ? '5px 12px' : '3px 10px')};
  font-size: ${({ size }) => (size === 'lg' ? '12px' : '11px')};
  font-weight: 500;
  white-space: nowrap;
  ${({ variant }) => badgeVariants[variant || 'neutral']}
`;

export const Badge = ({ children, variant = 'neutral', size = 'sm', ...props }) => (
  <StyledBadge variant={variant} size={size} {...props}>
    {children}
  </StyledBadge>
);

export default Badge;
