function EducationalSection({ title, children }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-5">
      <details>
        <summary className="cursor-pointer text-lg font-semibold text-slate-900">
          {title}
        </summary>
        <div className="mt-3 space-y-3 text-sm leading-6 text-slate-600">
          {children}
        </div>
      </details>
    </section>
  )
}

export default EducationalSection
