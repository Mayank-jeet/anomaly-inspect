export default function CategorySelect({ categories, value, onChange, disabled }) {
  return (
    <label className="field">
      <span className="field-label">1. Product category</span>
      <select value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
        {categories.map((c) => (
          <option key={c} value={c}>
            {c.replace("_", " ")}
          </option>
        ))}
      </select>
      <span className="hint">
        Each category has its own model. Upload an image of the <b>selected</b> product type;
        images from another category give confusing scores.
      </span>
    </label>
  );
}
