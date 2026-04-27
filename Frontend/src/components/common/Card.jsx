import styled from 'styled-components';

export const Card = styled.div`
  background: ${({ theme }) => theme.colors.cardBg};
  border-radius: ${({ theme }) => theme.radius.lg};
  box-shadow: ${({ theme }) => theme.shadows.card};
  transition: ${({ theme }) => theme.transitions.default};

  ${({ $clickable, theme }) => $clickable && `
    cursor: pointer;
    &:hover {
      box-shadow: ${theme.shadows.cardHover};
      transform: translateY(-2px);
    }
  `}
`;

export default Card;
