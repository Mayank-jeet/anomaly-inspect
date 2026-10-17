// All communication with the FastAPI backend lives here, so components never build URLs
// or parse errors themselves. The base URL comes from an environment variable so the same
// build works locally (127.0.0.1:8000) and on Vercel (the Render URL).
const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

// Same limits as the backend (api/main.py), checked here first so users get instant feedback
// instead of uploading 20 MB just to receive a 413.
export const MAX_UPLOAD_MB = 5;
export const ALLOWED_TYPES = ["image/png", "image/jpeg"];

export function validateFile(file) {
  if (!ALLOWED_TYPES.includes(file.type)) return "Please choose a PNG or JPEG image.";
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `File is too large (max ${MAX_UPLOAD_MB} MB).`;
  return null;
}

async function request(path, options) {
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, options);
  } catch {
    throw new Error("Cannot reach the server. It may be starting up, please try again in a moment.");
  }
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body.detail) message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* response had no JSON body; keep the generic message */
    }
    throw new Error(message);
  }
  return res.json();
}

export const getHealth = () => request("/health");

export const getCategories = () => request("/categories").then((d) => d.categories);

export const getSamples = (category) =>
  request(`/samples/${encodeURIComponent(category)}`).then((d) =>
    d.samples.map((s) => ({ ...s, url: `${API_URL}${s.url}` }))
  );

export function predict(file, category) {
  const form = new FormData();
  form.append("file", file);
  form.append("category", category);
  // Do NOT set Content-Type manually: the browser must add the multipart boundary itself.
  return request("/predict", { method: "POST", body: form });
}

// Turn a sample image URL into a File, so samples go through exactly the same
// upload path (and the same /predict call) as user uploads.
export async function sampleToFile(sample) {
  // cache: "reload" skips any cached copy of this URL. The thumbnail <img> loaded it earlier
  // without an Origin header, so the cached response has no CORS headers and would make
  // this fetch() fail with a CORS error.
  const res = await fetch(sample.url, { cache: "reload" });
  if (!res.ok) throw new Error("Could not load the sample image.");
  const blob = await res.blob();
  return new File([blob], sample.name, { type: blob.type || "image/jpeg" });
}
