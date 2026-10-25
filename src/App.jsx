import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import Navbar from './components/Navbar'
import About from './pages/About'
import Dashboard from './pages/Dashboard'
import ECGAnalysis from './pages/ECGAnalysis'
import EEGAnalysis from './pages/EEGAnalysis'

function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen flex-col bg-slate-50">
        <Navbar />
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/eeg" element={<EEGAnalysis />} />
            <Route path="/ecg" element={<ECGAnalysis />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
