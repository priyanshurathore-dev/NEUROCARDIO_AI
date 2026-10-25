import { AlertCircle, CheckCircle2, Info, LoaderCircle } from 'lucide-react'
import { formatFileSize } from '../utils/signalFile'

function FileFacts({ file }) {
  if (!file) {
    return null
  }

  return (
    <dl className="mt-4 grid gap-2 text-sm sm:grid-cols-3">
      <div>
        <dt className="text-slate-500">File</dt>
        <dd className="font-medium text-slate-800">{file.name}</dd>
      </div>
      <div>
        <dt className="text-slate-500">Size</dt>
        <dd className="font-medium text-slate-800">{formatFileSize(file.size)}</dd>
      </div>
      <div>
        <dt className="text-slate-500">Type</dt>
        <dd className="font-medium text-slate-800">{file.type}</dd>
      </div>
    </dl>
  )
}

function ResultsPanel({ result }) {
  const status = result?.status || 'idle'

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5">
      <h2 className="text-lg font-semibold text-slate-900">Analysis results</h2>

      {status === 'idle' ? (
        <p className="mt-3 flex items-start gap-2 text-sm text-slate-600">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" aria-hidden="true" />
          Upload a recording and choose Analyze to request results. No prediction
          is shown until a real analysis service responds.
        </p>
      ) : null}

      {status === 'loading' ? (
        <p className="mt-3 flex items-center gap-2 text-sm text-slate-700">
          <LoaderCircle className="h-4 w-4 animate-spin text-clinical-700" aria-hidden="true" />
          Sending the recording for analysis…
        </p>
      ) : null}

      {status === 'unavailable' ? (
        <div className="mt-3">
          <p className="flex items-start gap-2 text-sm text-slate-800">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-amber-700" aria-hidden="true" />
            {result.message}
          </p>
          <FileFacts file={result.file} />
          <p className="mt-4 rounded-md border border-slate-200 bg-slate-50 p-3 text-sm text-slate-600">
            Trained EEG and ECG models will be connected through FastAPI later.
            Until then, this panel reports file details only and does not invent
            clinical findings.
          </p>
        </div>
      ) : null}

      {status === 'error' ? (
        <div className="mt-3" role="alert">
          <p className="flex items-start gap-2 text-sm text-red-800">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            {result.message}
          </p>
          <FileFacts file={result.file} />
        </div>
      ) : null}

      {status === 'success' ? (
        <div className="mt-3">
          <p className="flex items-start gap-2 text-sm text-slate-800">
            <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-clinical-700" aria-hidden="true" />
            {result.data?.message || 'The analysis service returned a response.'}
          </p>
          <FileFacts file={result.file} />
          {result.data?.status ? (
            <p className="mt-3 text-sm text-slate-700">
              Status: <span className="font-medium">{result.data.status}</span>
            </p>
          ) : null}
          {Array.isArray(result.data?.notes) && result.data.notes.length > 0 ? (
            <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-slate-700">
              {result.data.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          ) : null}
          {Array.isArray(result.data?.series) ? (
            <p className="mt-3 text-sm text-slate-600">
              Returned series length: {result.data.series.length} samples.
            </p>
          ) : null}
          <p className="mt-4 text-xs text-slate-500">
            Any returned values still require review by a qualified healthcare
            professional. This application does not confirm a diagnosis.
          </p>
        </div>
      ) : null}
    </section>
  )
}

export default ResultsPanel
