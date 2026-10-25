import { Activity, Brain, Heart } from 'lucide-react'
import { Link } from 'react-router-dom'

function Dashboard() {
  return (
    <div className="space-y-8">
      <section className="rounded-lg border border-slate-200 bg-white p-6 sm:p-8">
        <p className="text-sm font-medium text-clinical-700">Clinical review assistant</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-slate-900">
          Review EEG and ECG recordings with a clear, careful workflow
        </h1>
        <p className="mt-4 max-w-3xl text-slate-600">
          NeuroCardio AI helps doctors inspect brain and heart electrical
          signals more efficiently. Patients can also use these pages to learn
          what EEG and ECG recordings are. The application is a research
          prototype. It does not replace clinical judgment and does not provide
          a confirmed diagnosis.
        </p>
        <div className="mt-6 flex flex-col gap-3 sm:flex-row">
          <Link
            to="/eeg"
            className="inline-flex items-center justify-center rounded-md bg-clinical-700 px-4 py-2.5 text-sm font-medium text-white hover:bg-clinical-800"
          >
            Open EEG analysis
          </Link>
          <Link
            to="/ecg"
            className="inline-flex items-center justify-center rounded-md border border-slate-300 bg-white px-4 py-2.5 text-sm font-medium text-slate-800 hover:bg-slate-50"
          >
            Open ECG analysis
          </Link>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-semibold text-slate-900">How the workflow works</h2>
        <ol className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            {
              step: '1',
              title: 'Upload',
              text: 'Choose an EEG or ECG file from your computer.',
            },
            {
              step: '2',
              title: 'Validate',
              text: 'The app checks the file type and size before anything else runs.',
            },
            {
              step: '3',
              title: 'Preview',
              text: 'CSV and text files can be charted in the browser. Other formats wait for backend processing.',
            },
            {
              step: '4',
              title: 'Analyze',
              text: 'Results appear on the same page when a real analysis service is connected.',
            },
          ].map((item) => (
            <li
              key={item.step}
              className="rounded-lg border border-slate-200 bg-white p-4"
            >
              <p className="text-xs font-semibold tracking-wide text-clinical-700">
                STEP {item.step}
              </p>
              <h3 className="mt-1 font-semibold text-slate-900">{item.title}</h3>
              <p className="mt-2 text-sm text-slate-600">{item.text}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <article className="rounded-lg border border-slate-200 bg-white p-5">
          <Brain className="h-6 w-6 text-clinical-700" aria-hidden="true" />
          <h2 className="mt-3 text-lg font-semibold text-slate-900">EEG</h2>
          <p className="mt-2 text-sm text-slate-600">
            An electroencephalogram records electrical activity from the brain,
            usually with sensors on the scalp. Clinicians use it when they need
            to understand seizures, sleep patterns, or other brain-rhythm
            questions.
          </p>
          <Link to="/eeg" className="mt-4 inline-block text-sm font-medium text-accent-700 hover:underline">
            Go to EEG analysis
          </Link>
        </article>
        <article className="rounded-lg border border-slate-200 bg-white p-5">
          <Heart className="h-6 w-6 text-clinical-700" aria-hidden="true" />
          <h2 className="mt-3 text-lg font-semibold text-slate-900">ECG</h2>
          <p className="mt-2 text-sm text-slate-600">
            An electrocardiogram records electrical activity from the heart.
            Clinicians use it to review heart rhythm and conduction. This app
            can help organize a recording for review; it does not diagnose
            cardiac conditions.
          </p>
          <Link to="/ecg" className="mt-4 inline-block text-sm font-medium text-accent-700 hover:underline">
            Go to ECG analysis
          </Link>
        </article>
      </section>

      <section className="flex gap-3 rounded-lg border border-slate-200 bg-white p-5">
        <Activity className="mt-0.5 h-5 w-5 shrink-0 text-slate-500" aria-hidden="true" />
        <p className="text-sm text-slate-600">
          Trained models and the FastAPI backend will be added later. Until
          then, analysis results stay empty or show that the service is not
          connected. Do not treat any chart on this site as a medical
          conclusion.
        </p>
      </section>
    </div>
  )
}

export default Dashboard
