import { createGlobalStyle } from 'styled-components';

export const GlobalStyles = createGlobalStyle`
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  html, body, #root { min-height: 100%; }

  body {
    font-family: ${({ theme }) => theme.fonts.sans};
    background: ${({ theme }) => theme.colors.pageBg};
    color: ${({ theme }) => theme.colors.textPrimary};
    -webkit-font-smoothing: antialiased;
  }

  a { color: inherit; text-decoration: none; }
  button, input, select, textarea { font: inherit; }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #ddd; border-radius: 4px; }
`;
