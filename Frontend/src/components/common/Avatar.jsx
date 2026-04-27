import styled from 'styled-components';

const StyledAvatar = styled.div`
  width: ${({ size = 44 }) => `${size}px`};
  height: ${({ size = 44 }) => `${size}px`};
  border-radius: 50%;
  display: grid;
  place-items: center;
  font-weight: 600;
  color: #fff;
  background: ${({ variant, theme }) => {
    if (variant === 'detected') return theme.colors.orange;
    if (variant === 'healthy') return theme.colors.teal;
    return '#b7bcc7';
  }};
`;

export const Avatar = ({ initials, size = 44, variant = 'default', ...props }) => (
  <StyledAvatar size={size} variant={variant} {...props}>{initials}</StyledAvatar>
);

export default Avatar;
