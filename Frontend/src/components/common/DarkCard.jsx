import styled from 'styled-components';

const DarkCard = styled.div`
  background: ${({ theme }) => theme.colors.darkCard};
  border-radius: ${({ theme }) => theme.radius.md};
  padding: 12px 14px;
`;

export default DarkCard;
