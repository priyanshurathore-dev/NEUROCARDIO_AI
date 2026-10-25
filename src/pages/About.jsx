function About() {
  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-slate-200 bg-white p-6">
        <h1 className="text-2xl font-semibold text-slate-900">About this project</h1>
        <p className="mt-3 text-sm leading-6 text-slate-600">
          NeuroCardio AI is a research and teaching prototype for reviewing EEG
          (brain) and ECG (heart) electrical recordings. The goal is a calm,
          clinical interface that doctors can use to organize files and that
          patients can read to understand what these signals are.
        </p>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-6">
        <h2 className="text-lg font-semibold text-slate-900">Purpose</h2>
        <ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-slate-600">
          <li>Give clinicians a simple place to upload and preview recordings.</li>
          <li>Keep analysis results on the same page as the file being reviewed.</li>
          <li>Explain EEG and ECG in plain language without overstating what software can do.</li>
        </ul>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-6">
        <h2 className="text-lg font-semibold text-slate-900">Technology</h2>
        <p className="mt-3 text-sm leading-6 text-slate-600">
          This frontend is built with React, Vite, Tailwind CSS, React Router,
          and Recharts. A FastAPI backend and trained EEG/ECG models are planned
          for a later phase. They are not part of this interface yet.
        </p>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-6">
        <h2 className="text-lg font-semibold text-slate-900">Limitations</h2>
        <ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-slate-600">
          <li>
            Browser preview only works for simple CSV and text numeric files.
          </li>
          <li>
            Formats such as EDF, FIF, SET, DAT, HEA, and MAT are accepted for
            upload but are not parsed here.
          </li>
          <li>
            Charts are a review aid, not a full clinical signal workstation.
          </li>
          <li>
            There is no published clinical validation, accuracy claim, or
            regulatory clearance for this prototype.
          </li>
          <li>
            Demonstration waveforms are synthetic examples. They are labeled and
            are not the patient recording.
          </li>
        </ul>
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-6">
        <h2 className="text-lg font-semibold text-slate-900">Responsible use</h2>
        <p className="mt-3 text-sm leading-6 text-slate-600">
          Do not use this application as a substitute for emergency care,
          clinical examination, or a confirmed diagnosis. Any future model
          output must be reviewed by a qualified healthcare professional. If you
          are a patient and have symptoms, contact a licensed clinician or local
          emergency services.
        </p>
      </section>
    </div>
  )
}

export default About
