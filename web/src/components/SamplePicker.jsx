export default function SamplePicker({ samples, selectedName, onPick, disabled }) {
  if (samples.length === 0) return null;
  return (
    <div className="field">
      <span className="field-label">or try a sample</span>
      <div className="samples">
        {samples.map((s) => (
          <button
            key={s.url}
            type="button"
            className={`sample ${selectedName === s.name ? "selected" : ""}`}
            onClick={() => onPick(s)}
            disabled={disabled}
            title={s.name}
          >
            <img src={s.url} alt={s.name} loading="lazy" crossOrigin="anonymous" />
            <span className={`tag ${s.kind}`}>{s.kind === "good" ? "normal" : "defective"}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
