import { useEffect, useState } from "react";
import { getCategories, getHealth, getSamples, predict, sampleToFile, validateFile } from "./api.js";
import CategorySelect from "./components/CategorySelect.jsx";
import UploadBox from "./components/UploadBox.jsx";
import SamplePicker from "./components/SamplePicker.jsx";
import ResultView from "./components/ResultView.jsx";

export default function App() {
  // "checking" -> "waking" (slow) -> "ready" | "down". Free hosts sleep when idle, so the first
  // request can take ~a minute; the user should see a message instead of a frozen page.
  const [server, setServer] = useState("checking");
  const [categories, setCategories] = useState([]);
  const [category, setCategory] = useState("");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [samples, setSamples] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // On page load: ping /health (retrying while the server wakes up), then fill the dropdown.
  useEffect(() => {
    let cancelled = false;
    const wakingTimer = setTimeout(() => {
      if (!cancelled) setServer((s) => (s === "checking" ? "waking" : s));
    }, 2500);

    (async () => {
      for (let attempt = 0; attempt < 30 && !cancelled; attempt++) {
        try {
          await getHealth();
          const cats = await getCategories();
          if (cancelled) return;
          setCategories(cats);
          setCategory(cats[0] || "");
          setServer("ready");
          return;
        } catch {
          await new Promise((r) => setTimeout(r, 3000));
        }
      }
      if (!cancelled) setServer("down");
    })();

    return () => {
      cancelled = true;
      clearTimeout(wakingTimer);
    };
  }, []);

  // Load the sample images whenever the category changes; an old result no longer applies.
  useEffect(() => {
    if (!category) return;
    setResult(null);
    setSamples([]);
    let cancelled = false;
    getSamples(category)
      .then((s) => !cancelled && setSamples(s))
      .catch(() => !cancelled && setSamples([]));
    return () => {
      cancelled = true;
    };
  }, [category]);

  // Local preview of the chosen file. Revoke the object URL afterwards to avoid leaking memory.
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const chooseFile = (f) => {
    const problem = validateFile(f);
    if (problem) {
      setError(problem);
      return;
    }
    setError("");
    setResult(null);
    setFile(f);
  };

  const pickSample = async (sample) => {
    try {
      chooseFile(await sampleToFile(sample));
    } catch (e) {
      setError(e.message);
    }
  };

  const inspect = async () => {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await predict(file, category));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const ready = server === "ready";

  return (
    <main className="app">
      <header>
        <h1>anomaly-inspect</h1>
        <p className="subtitle">
          Unsupervised defect detection on MVTec AD. The model was trained on defect-free images only.
        </p>
      </header>

      {(server === "checking" || server === "waking") && (
        <div className="banner info">
          {server === "waking"
            ? "Waking up the server… the first load can take up to a minute."
            : "Connecting to the server…"}
        </div>
      )}
      {server === "down" && (
        <div className="banner error">The server is not responding. Please try again later.</div>
      )}

      <div className="panel">
        <CategorySelect categories={categories} value={category} onChange={setCategory} disabled={!ready} />
        <UploadBox previewUrl={previewUrl} onFile={chooseFile} />
        <SamplePicker
          samples={samples}
          selectedName={file?.name}
          onPick={pickSample}
          disabled={!ready || loading}
        />

        <button className="inspect" onClick={inspect} disabled={!ready || !file || !category || loading}>
          {loading ? "Inspecting…" : "Inspect"}
        </button>

        {error && <div className="banner error">{error}</div>}
      </div>

      {result && <ResultView result={result} />}

      <footer>
        Dataset: MVTec AD (Bergmann et al.), licensed CC BY-NC-SA 4.0, non-commercial use only.
      </footer>
    </main>
  );
}
