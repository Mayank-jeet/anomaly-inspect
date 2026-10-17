import { useState } from "react";

export default function ResultView({ result }) {
  const [opacity, setOpacity] = useState(0.6);
  const {
    category, out_of_distribution, score, threshold, is_defective,
    heatmap_png_base64, input_png_base64, latency_ms,
  } = result;

  const input = `data:image/png;base64,${input_png_base64}`;

  // The image does not match the setup this category's model was trained on, so any verdict
  // or heatmap would be meaningless. Say so instead of showing a confident but wrong answer.
  if (out_of_distribution) {
    return (
      <section className="result">
        <div className="verdict-row">
          <span className="badge warn">Unsupported image</span>
          <span className="latency">{latency_ms} ms</span>
        </div>
        <p className="ood-text">
          This image doesn&apos;t look like the <b>{category.replace("_", " ")}</b> images this model was
          trained on (camera, background, lighting or object differ), so a defect verdict would not be
          reliable. Try one of the samples, or an image taken in the same setup as MVTec AD.
        </p>
        <figure className="ood-figure">
          <img src={input} alt="Model input" />
          <figcaption>Input (the 224×224 center crop the model saw)</figcaption>
        </figure>
      </section>
    );
  }

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
