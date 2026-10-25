export const MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024

export const EEG_EXTENSIONS = ['.csv', '.txt', '.edf', '.fif', '.set']
export const ECG_EXTENSIONS = ['.csv', '.txt', '.dat', '.hea', '.mat']

const PARSEABLE_EXTENSIONS = ['.csv', '.txt']
const MAX_CHART_POINTS = 1200

export function getFileExtension(filename) {
  const name = filename.toLowerCase()
  const dot = name.lastIndexOf('.')
  if (dot < 0) {
    return ''
  }
  return name.slice(dot)
}

export function formatFileSize(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function allowedExtensionsFor(signalType) {
  return signalType === 'ecg' ? ECG_EXTENSIONS : EEG_EXTENSIONS
}

export function validateSignalFile(file, signalType) {
  if (!file) {
    return { ok: false, message: 'Please choose a signal file.' }
  }

  if (file.size === 0) {
    return { ok: false, message: 'The selected file is empty.' }
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    return {
      ok: false,
      message: `This file is larger than ${formatFileSize(MAX_FILE_SIZE_BYTES)}. Please choose a smaller file.`,
    }
  }

  const extension = getFileExtension(file.name)
  const allowed = allowedExtensionsFor(signalType)

  if (!allowed.includes(extension)) {
    return {
      ok: false,
      message: `This file type is not accepted. Use ${allowed.join(', ')}.`,
    }
  }

  return { ok: true }
}

function parseNumber(value) {
  const cleaned = String(value).trim().replace(/,$/, '')
  if (cleaned === '') {
    return null
  }
  const number = Number(cleaned)
  return Number.isFinite(number) ? number : null
}

export function parseCsvOrTxt(text) {
  const lines = text.split(/\r?\n/).filter((line) => line.trim() !== '')
  if (lines.length === 0) {
    return { ok: false, message: 'The file does not contain any rows.' }
  }

  const series = []
  let skippedHeader = false

  for (let i = 0; i < lines.length; i += 1) {
    const parts = lines[i].split(/[,;\t ]+/).filter((part) => part !== '')
    if (parts.length === 0) {
      continue
    }

    if (parts.length === 1) {
      const value = parseNumber(parts[0])
      if (value === null) {
        if (i === 0 && !skippedHeader) {
          skippedHeader = true
          continue
        }
        continue
      }
      series.push({ t: series.length, value })
      continue
    }

    const first = parseNumber(parts[0])
    const second = parseNumber(parts[1])

    if (first === null || second === null) {
      if (i === 0 && !skippedHeader) {
        skippedHeader = true
        continue
      }
      continue
    }

    series.push({ t: first, value: second })
  }

  if (series.length < 8) {
    return {
      ok: false,
      message: 'Not enough numeric samples were found to draw a preview.',
    }
  }

  return { ok: true, series: downsampleSeries(series, MAX_CHART_POINTS) }
}

export function downsampleSeries(series, maxPoints = MAX_CHART_POINTS) {
  if (series.length <= maxPoints) {
    return series
  }

  const step = Math.ceil(series.length / maxPoints)
  const sampled = []

  for (let i = 0; i < series.length; i += step) {
    sampled.push(series[i])
  }

  const last = series[series.length - 1]
  if (sampled[sampled.length - 1] !== last) {
    sampled.push(last)
  }

  return sampled
}

export async function preparePreview(file) {
  const extension = getFileExtension(file.name)

  if (!PARSEABLE_EXTENSIONS.includes(extension)) {
    return {
      canParse: false,
      series: null,
      note: 'This format cannot be previewed in the browser. Preview will be available after backend processing.',
    }
  }

  try {
    const text = await file.text()
    const parsed = parseCsvOrTxt(text)
    if (!parsed.ok) {
      return {
        canParse: false,
        series: null,
        note: `${parsed.message} Preview will be available after backend processing.`,
      }
    }

    return {
      canParse: true,
      series: parsed.series,
      note: 'Preview of samples from the uploaded file. Long recordings are downsampled for display.',
    }
  } catch {
    return {
      canParse: false,
      series: null,
      note: 'The file could not be read in the browser. Preview will be available after backend processing.',
    }
  }
}
