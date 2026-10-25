function round(value) {
  return Math.round(value * 1000) / 1000
}

export function createDemoWaveform(signalType) {
  const pointCount = 400
  const series = []

  if (signalType === 'ecg') {
    for (let i = 0; i < pointCount; i += 1) {
      const phase = (i % 50) / 50
      let value = 0.05 * Math.sin(i / 8)

      if (phase > 0.18 && phase < 0.28) {
        value += 0.12 * Math.sin(((phase - 0.18) / 0.1) * Math.PI)
      }
      if (phase > 0.36 && phase < 0.42) {
        value -= 0.18
      }
      if (phase > 0.42 && phase < 0.5) {
        value += 1.15 * Math.exp(-((phase - 0.45) ** 2) / 0.00035)
      }
      if (phase > 0.5 && phase < 0.56) {
        value -= 0.22
      }
      if (phase > 0.68 && phase < 0.88) {
        value += 0.22 * Math.sin(((phase - 0.68) / 0.2) * Math.PI)
      }

      series.push({ t: i, value: round(value) })
    }

    return series
  }

  for (let i = 0; i < pointCount; i += 1) {
    const slow = Math.sin(i / 18) * 0.45
    const alpha = Math.sin(i / 4.2) * 0.22
    const jitter = ((i * 17) % 11) / 80 - 0.06
    series.push({ t: i, value: round(slow + alpha + jitter) })
  }

  return series
}
