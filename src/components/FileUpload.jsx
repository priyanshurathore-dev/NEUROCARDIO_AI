import { Upload } from 'lucide-react'
import { useState } from 'react'
import { formatFileSize } from '../utils/signalFile'

function FileUpload({ accept, onFileChosen, selectedFile, error }) {
  const [isDragging, setIsDragging] = useState(false)

  function handleFiles(fileList) {
    const file = fileList?.[0]
    if (file) {
      onFileChosen(file)
    }
  }

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5">
      <h2 className="text-lg font-semibold text-slate-900">Upload recording</h2>
      <p className="mt-1 text-sm text-slate-600">
        Choose a local signal file. The file stays in your browser until you run
        analysis.
      </p>

      <label
        className={[
          'mt-4 flex cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed px-4 py-8 text-center',
          isDragging
            ? 'border-clinical-600 bg-clinical-50'
            : 'border-slate-300 bg-slate-50 hover:border-clinical-600 hover:bg-clinical-50',
        ].join(' ')}
        onDragOver={(event) => {
          event.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault()
          setIsDragging(false)
          handleFiles(event.dataTransfer.files)
        }}
      >
        <Upload className="h-8 w-8 text-clinical-700" aria-hidden="true" />
        <span className="mt-2 text-sm font-medium text-slate-800">
          Drop a file here or click to browse
        </span>
        <span className="mt-1 text-xs text-slate-500">Accepted: {accept}</span>
        <input
          type="file"
          accept={accept}
          className="sr-only"
          onChange={(event) => {
            handleFiles(event.target.files)
            event.target.value = ''
          }}
        />
      </label>

      {selectedFile ? (
        <p className="mt-3 text-sm text-slate-700">
          Selected: <span className="font-medium">{selectedFile.name}</span>
          {' · '}
          {formatFileSize(selectedFile.size)}
        </p>
      ) : (
        <p className="mt-3 text-sm text-slate-500">No file selected yet.</p>
      )}

      {error ? (
        <p className="mt-2 text-sm text-red-700" role="alert">
          {error}
        </p>
      ) : null}
    </section>
  )
}

export default FileUpload
