import { createContext, useContext, useEffect, useMemo, useReducer } from 'react';
import { useAuthContext } from './AuthContext';
import { deletePatientById, fetchPatients } from '../services/patients';

const AppContext = createContext(null);

const initialState = {
  patients: [],
  loadingPatients: true,
  sidebarOpen: false,
  searchQuery: '',
  filter: 'all',
};

function reducer(state, action) {
  switch (action.type) {
    case 'SET_PATIENTS':
      return { ...state, patients: action.payload, loadingPatients: false };
    case 'SET_LOADING':
      return { ...state, loadingPatients: action.payload };
    case 'SET_SIDEBAR_OPEN':
      return { ...state, sidebarOpen: action.payload };
    case 'SET_SEARCH_QUERY':
      return { ...state, searchQuery: action.payload };
    case 'SET_FILTER':
      return { ...state, filter: action.payload };
    case 'DELETE_PATIENT':
      return { ...state, patients: state.patients.filter((patient) => patient.id !== action.payload) };
    default:
      return state;
  }
}

export const AppProvider = ({ children }) => {
  const { user } = useAuthContext();
  const [state, dispatch] = useReducer(reducer, initialState);

  useEffect(() => {
    let active = true;

    const loadPatients = async () => {
      if (!user) {
        dispatch({ type: 'SET_PATIENTS', payload: [] });
        dispatch({ type: 'SET_LOADING', payload: false });
        return;
      }

      dispatch({ type: 'SET_LOADING', payload: true });
      const records = await fetchPatients(user.uid);
      if (active) {
        dispatch({ type: 'SET_PATIENTS', payload: records });
      }
    };

    loadPatients().catch(() => {
      if (active) {
        dispatch({ type: 'SET_PATIENTS', payload: [] });
        dispatch({ type: 'SET_LOADING', payload: false });
      }
    });

    return () => {
      active = false;
    };
  }, [user]);

  const value = useMemo(() => ({
    ...state,
    dispatch,
    refreshPatients: async () => {
      if (!user) return;
      dispatch({ type: 'SET_LOADING', payload: true });
      const records = await fetchPatients(user.uid);
      dispatch({ type: 'SET_PATIENTS', payload: records });
    },
    removePatient: async (id) => {
      await deletePatientById(id);
      dispatch({ type: 'DELETE_PATIENT', payload: id });
    },
  }), [state, user]);

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export const useAppContext = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within AppProvider');
  }
  return context;
};
