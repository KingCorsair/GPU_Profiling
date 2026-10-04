import { useEffect, useRef, useState } from 'react';

type TurnstileApi = {
  render: (element: HTMLElement, options: { sitekey: string; action: string; theme: string; size: string;
    callback: (token: string) => void; 'expired-callback': () => void; 'error-callback': () => void; 'timeout-callback': () => void }) => string;
  remove: (id: string) => void;
};
declare global { interface Window { turnstile?: TurnstileApi } }

let loading: Promise<TurnstileApi> | null = null;
function loadTurnstile(): Promise<TurnstileApi> {
  if (window.turnstile) return Promise.resolve(window.turnstile);
  if (loading) return loading;
  loading = new Promise<TurnstileApi>((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';
    script.async = true;
    const timeout = window.setTimeout(fail, 20_000);
    function fail() { clearTimeout(timeout); script.remove(); reject(new Error('Verification could not load.')); }
    script.onerror = fail;
    script.onload = () => { clearTimeout(timeout); if (window.turnstile) resolve(window.turnstile); else fail(); };
    document.head.appendChild(script);
  }).catch((failure) => { loading = null; throw failure; });
  return loading;
}

/** A fresh widget after every action: Turnstile tokens are single use. */
export default function Turnstile({ siteKey, resetKey, onToken }: { siteKey: string; resetKey: number; onToken: (token: string | null) => void }) {
  const container = useRef<HTMLDivElement>(null);
  const callback = useRef(onToken);
  callback.current = onToken;
  const [status, setStatus] = useState('Loading visitor verification…');
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let active = true;
    let widget: string | undefined;
    callback.current(null);
    setStatus('Loading visitor verification…');
    setFailed(false);
    const invalid = (message: string) => { if (active) { callback.current(null); setStatus(message); setFailed(true); } };
    void loadTurnstile().then((api) => {
      if (!active || !container.current) return;
      setStatus('Complete the visitor verification to continue.');
      widget = api.render(container.current, {
        sitekey: siteKey, action: 'live-demo', theme: 'auto', size: 'flexible',
        callback: (token) => { if (active) { callback.current(token); setStatus('Visitor verified.'); setFailed(false); } },
        'expired-callback': () => invalid('Verification expired. Verify again to continue.'),
        'error-callback': () => invalid('Verification could not complete. Check your connection and try again.'),
        'timeout-callback': () => invalid('Verification timed out. Verify again to continue.'),
      });
    }).catch(() => invalid('Verification could not load. Check your connection and try again.'));
    return () => { active = false; if (widget !== undefined) window.turnstile?.remove(widget); };
  }, [siteKey, resetKey, attempt]);
  return <div className="live-verification"><div ref={container} /><p role="status">{status}</p>
    {failed && <button type="button" className="play-secondary-button" onClick={() => setAttempt((value) => value + 1)}>Verify again</button>}
  </div>;
}
