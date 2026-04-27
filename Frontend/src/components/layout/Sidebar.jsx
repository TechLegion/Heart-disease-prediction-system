import { useMemo } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import styled from 'styled-components';
import { BarChart3, Heart, LayoutGrid, PlusCircle, Settings, User, Users } from 'lucide-react';
import { getInitials } from '../../utils/formatters';
import { useAuth } from '../../hooks/useAuth';
import { usePatients } from '../../hooks/usePatients';
import Avatar from '../common/Avatar';

const Shell = styled.aside`
  width: 52px;
  min-width: 52px;
  background: ${({ theme }) => theme.colors.sidebarBg};
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px 0;
  gap: 8px;

  @media (max-width: 767px) {
    display: none;
  }
`;

const Brand = styled(Link)`
  width: 32px;
  height: 32px;
  border-radius: 10px;
  display: grid;
  place-items: center;
  color: ${({ theme }) => theme.colors.orange};
  margin-bottom: 10px;
`;

const RailIcon = styled(NavLink)`
  width: 36px;
  height: 36px;
  display: grid;
  place-items: center;
  border-radius: 10px;
  color: rgba(255, 255, 255, 0.4);
  transition: ${({ theme }) => theme.transitions.default};

  &:hover {
    color: rgba(255, 255, 255, 0.7);
  }

  &.active {
    color: #fff;
    background: ${({ theme }) => theme.colors.orange};
  }
`;

const Group = styled.div`
  display: grid;
  gap: 8px;
`;

const Bottom = styled.div`
  margin-top: auto;
  display: grid;
  gap: 8px;
  align-items: center;
`;

const items = [
  { to: '/dashboard', icon: LayoutGrid },
  { to: '/patients', icon: Users },
  { to: '/predict', icon: PlusCircle },
  { to: '/dashboard', icon: BarChart3 },
];

export const Sidebar = () => {
  const { user, logout } = useAuth();
  const location = useLocation();

  const name = user?.displayName || user?.email || 'CardioSense User';
  const initials = useMemo(() => getInitials(name), [name]);

  return (
    <Shell>
      <Brand to="/dashboard" aria-label="CardioSense home">
        <Heart size={18} />
      </Brand>

      <Group>
        {items.map(({ to, icon: Icon }, index) => (
          <RailIcon key={`${to}-${index}`} to={to} state={{ from: location }}>
            <Icon size={16} />
          </RailIcon>
        ))}
      </Group>

      <Bottom>
        <RailIcon to="/dashboard" aria-label="profile"><User size={16} /></RailIcon>
        <RailIcon to="/dashboard" aria-label="settings"><Settings size={16} /></RailIcon>
        <button
          type="button"
          onClick={logout}
          aria-label="Sign out"
          style={{ background: 'transparent', border: 0, cursor: 'pointer' }}
        >
          <Avatar size={28} initials={initials} variant="detected" />
        </button>
      </Bottom>
    </Shell>
  );
};

export default Sidebar;
