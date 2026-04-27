import styled from 'styled-components';

const Wrapper = styled.label`
  display: grid;
  gap: 6px;
`;

const Label = styled.span`
  color: #999;
  font-size: 11px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
`;

const Field = styled.select`
  width: 100%;
  height: 44px;
  border-radius: ${({ theme }) => theme.radius.sm};
  border: 1px solid #ebebeb;
  background: ${({ theme }) => theme.colors.cardBg};
  color: ${({ theme }) => theme.colors.textPrimary};
  padding: 0 14px;
  outline: none;
  transition: ${({ theme }) => theme.transitions.default};

  &:focus {
    border-color: ${({ theme }) => theme.colors.orange};
    box-shadow: 0 0 0 3px rgba(232,115,74,0.12);
  }
`;

const Helper = styled.span`
  color: #bbb;
  font-size: 10px;
  line-height: 1.4;
`;

export const Select = ({ label, helperText, options = [], placeholder = 'Select option', ...props }) => (
  <Wrapper>
    {label ? <Label>{label}</Label> : null}
    <Field {...props}>
      {placeholder ? <option value="">{placeholder}</option> : null}
      {options.map((option) => (
        <option key={option.value ?? option.label} value={option.value}>
          {option.label}
        </option>
      ))}
    </Field>
    {helperText ? <Helper>{helperText}</Helper> : null}
  </Wrapper>
);

export default Select;
