import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

function SignalChart({ title, series, isDemo, note }) {
  const hasSeries = Array.isArray(series) && series.length > 0

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
        {isDemo && hasSeries ? (
          <span className="rounded-full border border-amber-300 bg-amber-50 px-2.5 py-0.5 text-xs font-medium text-amber-950">
            Demonstration data
          </span>
        ) : null}
      </div>

      {note ? <p className="mt-2 text-sm text-slate-600">{note}</p> : null}

      <div className="mt-4 h-64 w-full">
        {hasSeries ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={series} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
              <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
              <XAxis
                dataKey="t"
                tick={{ fill: '#475569', fontSize: 12 }}
                label={{
                  value: 'Sample / time',
                  position: 'insideBottom',
                  offset: -2,
                  fill: '#475569',
                  fontSize: 12,
                }}
              />
              <YAxis
                tick={{ fill: '#475569', fontSize: 12 }}
                label={{
                  value: 'Amplitude',
                  angle: -90,
                  position: 'insideLeft',
                  fill: '#475569',
                  fontSize: 12,
                }}
              />
              <Tooltip
                contentStyle={{
                  borderRadius: 8,
                  borderColor: '#cbd5e1',
                  fontSize: 12,
                }}
              />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#0f766e"
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center rounded-md border border-slate-200 bg-slate-50 px-4 text-center text-sm text-slate-500">
            Upload a file to see a signal preview.
          </div>
        )}
      </div>
    </section>
  )
}

export default SignalChart
