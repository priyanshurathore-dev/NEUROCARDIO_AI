import { AlertCircle, CheckCircle2, Info, LoaderCircle } from 'lucide-react'
import { formatFileSize } from '../utils/signalFile'

function FileFacts({ file }) {
  if (!file) return null

  return (
    <dl className="mt-4 grid gap-2 text-sm sm:grid-cols-3">
      <div>
        <dt className="text-slate-500">File</dt>
        <dd className="font-medium text-slate-800 break-all">{file.name}</dd>
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

function ResultsPanel({ result, modality = 'eeg' }) {
  const status = result?.status || 'idle'
  const data = result?.data

    const heartbeatCounts = data?.heartbeat_class_counts || {}

  const nonNormalEntries = Object.entries(heartbeatCounts).filter(
    ([name, count]) => name !== 'Normal' && count > 0
  )

  const nonNormalTotal = nonNormalEntries.reduce(
    (total, [, count]) => total + count,
    0
  )

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5">
      <h2 className="text-lg font-semibold text-slate-900">Analysis results</h2>

      {status === 'idle' && (
        <p className="mt-3 flex gap-2 text-sm text-slate-600">
          <Info className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          Upload a {modality.toUpperCase()} recording and choose Analyze to request a model prediction.
        </p>
      )}

      {status === 'loading' && (
        <p className="mt-3 flex items-center gap-2 text-sm text-slate-700">
          <LoaderCircle className="h-4 w-4 animate-spin text-clinical-700" aria-hidden="true" />
          Processing {modality.toUpperCase()} recording…
        </p>
      )}

      {(status === 'unavailable' || status === 'error') && (
        <div className="mt-3" role={status === 'error' ? 'alert' : undefined}>
          <p className="flex gap-2 text-sm text-slate-800">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-amber-700" aria-hidden="true" />
            {result.message}
          </p>
          <FileFacts file={result.file} />
        </div>
      )}

      {status === 'success' && (
  <div className="mt-4">
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
      <p className="text-sm text-slate-500">
        {data?.input_type === 'raw_ecg_file'
          ? 'Predominant heartbeat class'
          : 'Predicted EEG class'}
      </p>

      <p className="mt-1 text-xl font-semibold text-slate-900">
        {data?.predominant_heartbeat_class ||
          data?.predicted_label ||
          data?.message ||
          'Response received'}
      </p>

      {data?.model_name && (
        <p className="mt-1 text-sm text-slate-600">
          Model: {data.model_name}
        </p>
      )}

      {data?.input_type === 'raw_ecg_file' && (
        <>
          <p className="mt-1 text-sm text-slate-600">
            Detected heartbeats: {data.detected_heartbeats}
          </p>
          <p className="mt-1 text-sm text-slate-600">
            Analyzed heartbeats: {data.analyzed_heartbeats}
          </p>
          <p className="mt-1 text-sm text-slate-600">
            Sampling frequency: {data.sampling_frequency_hz} Hz
          </p>
        </>
      )}

      {data?.features_extracted !== undefined && (
        <p className="mt-1 text-sm text-slate-600">
          Features extracted: {data.features_extracted}
        </p>
      )}
    </div>

    {data?.heartbeat_class_counts && (
      <div className="mt-4">
        <h3 className="text-sm font-semibold text-slate-800">
          Heartbeat classification
        </h3>

        <ul className="mt-2 space-y-2">
          {Object.entries(data.heartbeat_class_counts).map(
            ([name, count]) => (
              <li
                key={name}
                className="flex justify-between rounded-md bg-slate-50 px-3 py-2 text-sm"
              >
                <span className="text-slate-700">{name}</span>
                <span className="font-semibold text-slate-900">
                  {count}
                </span>
              </li>
            )
          )}
        </ul>
      </div>
    )}

    {data?.heartbeat_class_counts && (
  <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-4">
    <h3 className="font-semibold text-slate-800">
      Signal summary
    </h3>

    {nonNormalTotal === 0 ? (
      <p className="mt-2 text-sm text-slate-700">
        All analyzed heartbeats were classified as Normal by the model.
      </p>
    ) : (
      <>
        <p className="mt-2 text-sm text-slate-700">
          The model classified {nonNormalTotal} heartbeats into
          non-normal categories.
        </p>

        <ul className="mt-2 list-disc pl-5 text-sm text-slate-700">
          {nonNormalEntries.map(([name, count]) => (
            <li key={name}>
              {name}: {count} beats
            </li>
          ))}
        </ul>
      </>
    )}

    <p className="mt-3 text-xs text-slate-600">
      These automated classifications do not establish a heart
      condition. A qualified healthcare professional should
      review the complete ECG recording.
    </p>
  </div>
)}

    {data?.average_model_scores && (
      <div className="mt-4">
        <h3 className="text-sm font-semibold text-slate-800">
          Average model scores (not calibrated probabilities)
        </h3>

        <ul className="mt-2 space-y-3">
          {Object.entries(data.average_model_scores).map(
            ([name, score]) => (
              <li key={name} className="text-sm text-slate-700">
                <div className="mb-1 flex justify-between gap-3">
                  <span>{name}</span>
                  <span>{(Number(score) * 100).toFixed(1)}%</span>
                </div>

                <div className="h-2 overflow-hidden rounded bg-slate-100">
                  <div
                    className="h-full rounded bg-clinical-700"
                    style={{
                      width: `${Math.max(
                        0,
                        Math.min(100, Number(score) * 100)
                      )}%`,
                    }}
                  />
                </div>
              </li>
            )
          )}
        </ul>
      </div>
    )}

    {data?.class_scores && (
      <div className="mt-4">
        <h3 className="text-sm font-semibold text-slate-800">
          Model class scores (not calibrated probabilities)
        </h3>
        <ul className="mt-2 space-y-2">
          {Object.entries(data.class_scores).map(([id, score]) => (
            <li key={id} className="text-sm text-slate-700">
              <div className="mb-1 flex justify-between gap-3">
                <span>
                  {({ 0: 'Normal EEG', 1: 'Seizure-free', 2: 'Active seizure' })[id] || `Class ${id}`}
                </span>
                <span>{(Number(score) * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 overflow-hidden rounded bg-slate-100">
                <div
                  className="h-full rounded bg-clinical-700"
                  style={{
                    width: `${Math.max(0, Math.min(100, Number(score) * 100))}%`,
                  }}
                />
              </div>
            </li>
          ))}
        </ul>
      </div>
    )}

    <FileFacts file={result.file} />

    <p className="mt-4 rounded-md border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900">
      {data?.clinical_notice ||
        'Research demonstration only. This prediction is not a medical diagnosis and is not a substitute for professional assessment.'}
    </p>
  </div>
)}
    </section>
  )
}

export default ResultsPanel
