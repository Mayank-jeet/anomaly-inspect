import { useState } from "react";

export default function ResultView({ result }) {
  const [opacity, setOpacity] = useState(0.6);
  const { score, threshold, is_defective, heatmap_png_base64, input_png_base64, latency_ms } = result;

  const input = `data:image/png;base64,${input_png_base64}`;
  const heatmap = `data:image/png;base64,${heatmap_png_base64}`;

  // Score bar: scale is 0 .. 2x threshold, so the threshold marker always sits in the middle.
  const fill = Math.min(100, (score / (threshold * 2)) * 100);

  return (
    <section className="result">
      <div className="verdict-row">
        <span className={`badge ${is_defective ? "bad" : "ok"}`}>
          {is_defective ? "Defective" : "Normal"}
        </span>
        <span className="latency">{latency_ms} ms</span>
      </div>

      <div className="scorebar" aria-label="anomaly score versus threshold">
        <div className={`scorefill ${is_defective ? "bad" : "ok"}`} style={{ width: `${fill}%` }} />
        <div className="threshold-marker" />
      </div>
      <p className="score-text">
        score <b>{score.toFixed(3)}</b> {is_defective ? ">" : "≤"} threshold <b>{threshold.toFixed(3)}</b>
      </p>

      <div className="compare">
        <figure>
          <img src={input} alt="Model input" />
          <figcaption>Input (the 224×224 center crop the model saw)</figcaption>
        </figure>
        <figure>
          <div className="overlay-wrap">
            <img src={input} alt="Model input under heatmap" />
            <img src={heatmap} alt="Anomaly heatmap" className="overlay" style={{ opacity }} />
          </div>
          <figcaption>Anomaly heatmap (red = most anomalous)</figcaption>
        </figure>
      </div>

      <label className="slider">
        Heatmap opacity
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={opacity}
          onChange={(e) => setOpacity(parseFloat(e.target.value))}
        />
      </label>
    </section>
  );
}
