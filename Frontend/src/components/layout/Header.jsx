import styled from 'styled-components';
import { Menu } from 'lucide-react';
import { Button } from '../common/Button';
import { usePatients } from '../../hooks/usePatients';

const Bar = styled.header`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 18px 24px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.default};
  background: ${({ theme }) => theme.colors.bg.surface};
`;

const TitleGroup = styled.div`
  display: grid;
  gap: 4px;
`;

const Eyebrow = styled.div`
  color: ${({ theme }) => theme.colors.text.secondary};
  font-size: 11px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
`;

const Title = styled.h1`
  margin: 0;
  font-size: 24px;
`;

const MenuButton = styled(Button)`
  display: none;

  @media (max-width: 767px) {
    display: inline-flex;
  }
`;

export const Header = ({ title, subtitle }) => {
  const { dispatch } = usePatients();

  return (
    <Bar>
      <TitleGroup>
        <Eyebrow>{subtitle || 'Clinical dashboard'}</Eyebrow>
        <Title>{title}</Title>
      </TitleGroup>
      <MenuButton variant="ghost" size="sm" onClick={() => dispatch({ type: 'SET_SIDEBAR_OPEN', payload: true })}>
        <Menu size={16} />
      </MenuButton>
    </Bar>
  );
};

export default Header;
