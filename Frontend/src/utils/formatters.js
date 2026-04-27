export const formatDateTime = (value) => {
  if (!value) return '--';
  const date = typeof value?.toDate === 'function' ? value.toDate() : new Date(value);
  return new Intl.DateTimeFormat('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
};

export const formatShortDate = (value) => {
  if (!value) return '--';
  const date = typeof value?.toDate === 'function' ? value.toDate() : new Date(value);
  return new Intl.DateTimeFormat('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  }).format(date);
};

export const getInitials = (name = '') => {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return 'CS';
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
};

export const percent = (value, digits = 1) => `${(Number(value || 0) * 100).toFixed(digits)}%`;

export const buildPatientAgeSex = (patient) => `${patient.age ?? '--'} · ${patient.sex ?? '--'}`;
