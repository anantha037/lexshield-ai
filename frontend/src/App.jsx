import { useEffect } from 'react';
import { StoreProvider, useStore } from './store';
import Landing from './pages/Landing';
import ResetPassword from './pages/ResetPassword';

import Dashboard from './pages/Dashboard';
import Toast from './components/Toast';

function AppRoutes() {
  const { user, loading, toast } = useStore();
  
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const verified = params.get('verified');
    if (verified) {
      if (verified === 'success') {
        toast("Email verified successfully! You can now use all features.", 'success');
      } else if (verified === 'already') {
        toast("Your email was already verified.", 'success');
      } else if (verified === 'error') {
        toast("Verification link is invalid or expired. Please request a new one.", 'error');
      }
      
      const url = new URL(window.location.href);
      url.searchParams.delete('verified');
      window.history.replaceState({}, '', url);
    }
  }, [toast]);

  if (loading) return (
    <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div className="typing"><span /><span /><span /></div>
    </div>
  );

  const path = window.location.pathname;
  if (path === '/reset-password') {
    return <ResetPassword />;
  }

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
