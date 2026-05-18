import { StoreProvider, useStore } from './store';
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import Toast from './components/Toast';

function AppRoutes() {
  const { user, loading } = useStore();
  
  if (loading) return (
    <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div className="typing"><span /><span /><span /></div>
    </div>
  );

  const isAnon = sessionStorage.getItem('lexshield_anon') === '1';
  if (!user && !isAnon) return <Landing />;
  
  return <Dashboard />;
}

export default function App() {
  return (
    <StoreProvider>
      <AppRoutes />
      <Toast />
    </StoreProvider>
  );
}
