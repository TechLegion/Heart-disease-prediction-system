import { X } from 'lucide-react';
import styled from 'styled-components';
import { Button } from './Button';

const Overlay = styled.div`
  min-height: 400px;
  background: rgba(0, 0, 0, 0.45);
  display: grid;
  place-items: center;
  border-radius: ${({ theme }) => theme.radius.xl};
  padding: 16px;
`;

const Dialog = styled.div`
  width: min(420px, 100%);
  background: ${({ theme }) => theme.colors.cardBg};
  border-radius: ${({ theme }) => theme.radius.lg};
  padding: 28px;
  box-shadow: ${({ theme }) => theme.shadows.shell};
`;

const Header = styled.div`
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
`;

const Title = styled.h3`
  margin: 0;
  font-size: 18px;
  color: ${({ theme }) => theme.colors.textPrimary};
`;

const Body = styled.div`
  color: ${({ theme }) => theme.colors.textSecondary};
  line-height: 1.6;
`;

const Actions = styled.div`
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 24px;
`;

export const Modal = ({ open, title, children, onClose, onConfirm, confirmLabel = 'Confirm', danger = false }) => {
  if (!open) return null;

  return (
    <Overlay onClick={onClose}>
      <Dialog onClick={(event) => event.stopPropagation()}>
        <Header>
          <Title>{title}</Title>
          <Button variant="icon" onClick={onClose} aria-label="Close dialog">
            <X size={16} />
          </Button>
        </Header>
        <Body>{children}</Body>
        <Actions>
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button variant={danger ? 'danger' : 'primary'} onClick={onConfirm}>{confirmLabel}</Button>
        </Actions>
      </Dialog>
    </Overlay>
  );
};

export default Modal;
