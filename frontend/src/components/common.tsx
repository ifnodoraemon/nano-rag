import type { ReactNode } from 'react';

interface PanelProps {
  title: string;
  subtitle?: string;
  children: ReactNode;
  actions?: ReactNode;
}

export function Panel({ title, subtitle, children, actions }: PanelProps) {
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2>{title}</h2>
          {subtitle && <div className="subtle">{subtitle}</div>}
        </div>
        {actions && <div className="panel-actions">{actions}</div>}
      </div>
      <div className="panel-body">{children}</div>
    </section>
  );
}

interface StatusLineProps {
  message?: string;
  isError?: boolean;
}

export function StatusLine({ message, isError }: StatusLineProps) {
  if (!message) return null;
  return <div className={`status-line${isError ? ' error' : ''}`}>{message}</div>;
}

interface JsonOutputProps {
  data: unknown;
  placeholder?: string;
}

export function JsonOutput({ data, placeholder = 'No data' }: JsonOutputProps) {
  if (!data) return <div className="output"><pre>{placeholder}</pre></div>;
  return (
    <div className="output">
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}

interface LoadingButtonProps {
  loading: boolean;
  onClick?: () => void;
  type?: 'button' | 'submit';
  children: ReactNode;
  variant?: 'primary' | 'secondary';
}

export function LoadingButton({
  loading,
  onClick,
  type = 'button',
  children,
  variant = 'primary',
}: LoadingButtonProps) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={loading}
      className={variant}
    >
      {loading ? 'Loading...' : children}
    </button>
  );
}

interface ChipProps {
  label: string;
  status: 'ok' | 'warn' | 'err' | 'neutral';
}

export function Chip({ label, status }: ChipProps) {
  return (
    <div className="chip">
      <span className={`dot ${status === 'neutral' ? '' : status}`} />
      {label}
    </div>
  );
}

interface CardProps {
  title: string;
  children: ReactNode;
}

export function Card({ title, children }: CardProps) {
  return (
    <div className="card">
      <strong>{title}</strong>
      <div>{children}</div>
    </div>
  );
}
