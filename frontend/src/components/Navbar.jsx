import { Activity, Menu, X } from 'lucide-react'
import { useState } from 'react'
import { NavLink } from 'react-router-dom'

const links = [
  { to: '/', label: 'Dashboard', end: true },
  { to: '/eeg', label: 'EEG Analysis' },
  { to: '/ecg', label: 'ECG Analysis' },
  { to: '/about', label: 'About Project' },
]

function linkClassName({ isActive }) {
  return [
    'rounded-md px-3 py-2 text-sm font-medium',
    isActive
      ? 'bg-clinical-50 text-clinical-800'
      : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900',
  ].join(' ')
}

function Navbar() {
  const [open, setOpen] = useState(false)

  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3">
        <NavLink to="/" className="flex items-center gap-2 text-slate-900">
          <Activity className="h-6 w-6 text-clinical-700" aria-hidden="true" />
          <span className="text-base font-semibold tracking-tight">
            NeuroCardio AI
          </span>
        </NavLink>

        <nav className="hidden items-center gap-1 md:flex" aria-label="Main">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.end}
              className={linkClassName}
            >
              {link.label}
            </NavLink>
          ))}
        </nav>

        <button
          type="button"
          className="rounded-md p-2 text-slate-700 hover:bg-slate-100 md:hidden"
          aria-expanded={open}
          aria-controls="mobile-nav"
          onClick={() => setOpen((current) => !current)}
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          <span className="sr-only">Toggle navigation</span>
        </button>
      </div>

      {open ? (
        <nav
          id="mobile-nav"
          className="space-y-1 border-t border-slate-200 px-4 py-3 md:hidden"
          aria-label="Mobile"
        >
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.end}
              className={({ isActive }) => `${linkClassName({ isActive })} block`}
              onClick={() => setOpen(false)}
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
      ) : null}

      <p className="border-t border-amber-200 bg-amber-50 px-4 py-2 text-center text-xs text-amber-950 md:text-sm">
        Research prototype. This tool does not provide a confirmed diagnosis.
        Findings must be reviewed by a qualified healthcare professional.
      </p>
    </header>
  )
}

export default Navbar
