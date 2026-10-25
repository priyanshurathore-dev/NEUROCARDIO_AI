const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'

function fileSummary(file) {
  return {
    name: file.name,
    size: file.size,
    type: file.type || 'unknown',
  }
}

function unavailableResult(file) {
  return {
    status: 'unavailable',
    message:
      'The analysis service is not connected yet. No model output, diagnosis, or confidence score is available.',
    file: fileSummary(file),
  }
}

async function postAnalyze(path, file) {
  const formData = new FormData()
  formData.append('file', file)

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      body: formData,
    })

    if (response.status === 404 || response.status === 502 || response.status === 503) {
      return unavailableResult(file)
    }

    const text = await response.text()
    let data = null
    if (text) {
      try {
        data = JSON.parse(text)
      } catch {
        data = { message: text }
      }
    }

    if (!response.ok) {
      return {
        status: 'error',
        message:
          data?.message ||
          `The analysis request failed with status ${response.status}.`,
        file: fileSummary(file),
      }
    }

    return {
      status: 'success',
      data: data || {},
      file: fileSummary(file),
    }
  } catch {
    return unavailableResult(file)
  }
}

export function analyzeEeg(file) {
  return postAnalyze('/eeg/analyze', file)
}

export function analyzeEcg(file) {
  return postAnalyze('/ecg/analyze', file)
}
