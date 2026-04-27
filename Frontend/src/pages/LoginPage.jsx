import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import styled from 'styled-components';
import { Heart } from 'lucide-react';
import Card from '../components/common/Card';
import { Input } from '../components/common/Input';
import { Button } from '../components/common/Button';
import { useAuth } from '../hooks/useAuth';

const Page = styled.div`
  min-height: 100vh;
  background: ${({ theme }) => theme.colors.pageBg};
  display: grid;
  place-items: center;
  padding: 20px;
`;

const AuthCard = styled(Card)`
  width: min(420px, 100%);
  border-radius: ${({ theme }) => theme.radius.xl};
  box-shadow: 0 8px 48px rgba(0,0,0,0.1);
  padding: 36px;
`;

const Center = styled.div`
  text-align: center;
`;

const Title = styled.h1`
  margin-top: 10px;
  font-size: 20px;
  color: ${({ theme }) => theme.colors.textPrimary};
`;

const Sub = styled.p`
  margin-top: 4px;
  font-size: 12px;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const Tabs = styled.div`
  margin-top: 22px;
  border-radius: ${({ theme }) => theme.radius.pill};
  background: #f4f4f4;
  padding: 4px;
  display: grid;
  grid-template-columns: 1fr 1fr;
`;

const Tab = styled.button`
  height: 34px;
  border: 0;
  border-radius: ${({ theme }) => theme.radius.pill};
  background: ${({ $active, theme }) => ($active ? theme.colors.orange : 'transparent')};
  color: ${({ $active }) => ($active ? '#fff' : '#888')};
  font-size: 12px;
  font-weight: 600;
`;

const Form = styled.form`
  margin-top: 18px;
  display: grid;
  gap: 12px;
`;

const Alert = styled.div`
  border-radius: ${({ theme }) => theme.radius.pill};
  background: rgba(232,83,110,0.1);
  color: ${({ theme }) => theme.colors.danger};
  font-size: 12px;
  padding: 8px 12px;
`;

const Forgot = styled.button`
  border: 0;
  background: transparent;
  color: ${({ theme }) => theme.colors.orange};
  font-size: 12px;
  text-align: right;
`;

export default function LoginPage() {
  const [mode, setMode] = useState('signin');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [form, setForm] = useState({ name: '', email: '', password: '', confirmPassword: '' });
  const { login, signup } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const onChange = (event) => {
    const { name, value } = event.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const onSubmit = async (event) => {
    event.preventDefault();
    setError('');

    if (mode === 'signup' && form.password !== form.confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    setLoading(true);
    try {
      if (mode === 'signin') {
        await login(form.email, form.password);
      } else {
        await signup(form.name, form.email, form.password);
      }
      navigate(location.state?.from?.pathname || '/dashboard', { replace: true });
    } catch (authError) {
      setError(authError.message || 'Authentication failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Page>
      <AuthCard>
        <Center>
          <Heart size={32} color="#e8734a" />
          <Title>CardioSense</Title>
          <Sub>Heart disease prediction platform</Sub>
        </Center>
        <Tabs>
          <Tab type="button" $active={mode === 'signin'} onClick={() => setMode('signin')}>Sign in</Tab>
          <Tab type="button" $active={mode === 'signup'} onClick={() => setMode('signup')}>Sign up</Tab>
        </Tabs>
        <Form onSubmit={onSubmit}>
          {mode === 'signup' ? <Input label="Name" name="name" value={form.name} onChange={onChange} /> : null}
          <Input label="Email" type="email" name="email" value={form.email} onChange={onChange} />
          <Input label="Password" type="password" name="password" value={form.password} onChange={onChange} />
          {mode === 'signup' ? <Input label="Confirm Password" type="password" name="confirmPassword" value={form.confirmPassword} onChange={onChange} /> : null}
          {mode === 'signin' ? <Forgot type="button">Forgot password?</Forgot> : null}
          {error ? <Alert>{error}</Alert> : null}
          <Button type="submit" loading={loading} size="lg">{mode === 'signin' ? 'Sign in' : 'Create account'}</Button>
        </Form>
      </AuthCard>
    </Page>
  );
}
