import { useRef, useState } from "react";

export default function UploadBox({ previewUrl, onFile }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = (files) => {
    if (files && files[0]) onFile(files[0]);
  };

  return (
    <div className="field">
      <span className="field-label">2. Upload an image</span>
      <div
        className={`dropzone ${dragging ? "dragging" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
      >
        {previewUrl ? (
          <img src={previewUrl} alt="Selected" className="preview" />
        ) : (
          <p>Drag and drop a PNG or JPEG here, or click to browse</p>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg"
          hidden
          onChange={(e) => {
            handleFiles(e.target.files);
            e.target.value = ""; // allow picking the same file twice in a row
          }}
        />
      </div>
    </div>
  );
}
