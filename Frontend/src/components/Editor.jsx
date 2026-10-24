import { useEffect, useState } from "react";
import "./Editor.css";

const API_URL = "http://127.0.0.1:8000";

const Editor = () => {
  const [originalUrl, setOriginalUrl] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [resultUrl, setResultUrl] = useState("");
  const [fileName, setFileName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [denoiseEnabled, setDenoiseEnabled] = useState(false);

  useEffect(() => {
    return () => {
      if (originalUrl) URL.revokeObjectURL(originalUrl);
    };
  }, [originalUrl]);

  useEffect(() => {
    return () => {
      if (resultUrl) URL.revokeObjectURL(resultUrl);
    };
  }, [resultUrl]);

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];

    setError("");
    setNotice("");
    setResultUrl("");
    setOriginalUrl("");
    setSelectedFile(null);
    setFileName("");

    if (!file) return;

    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setError("Please choose a JPG, PNG, or WebP image.");
      event.target.value = "";
      return;
    }

    setSelectedFile(file);
    setOriginalUrl(URL.createObjectURL(file));
    setFileName(file.name);
  };

  const handleProcess = async () => {
    if (!selectedFile || isLoading) return;

    setIsLoading(true);
    setError("");
    setNotice("");
    setResultUrl("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("denoise", String(denoiseEnabled));

      const response = await fetch(`${API_URL}/restore`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let message = "Image processing failed.";

        try {
          const data = await response.json();
          message = data.detail || message;
        } catch {
          // Keep the default message if the response isn't JSON.
        }

        throw new Error(message);
      }

      const imageBlob = await response.blob();

      if (!imageBlob.type.startsWith("image/")) {
        throw new Error("The server did not return a valid image.");
      }

      const newResultUrl = URL.createObjectURL(imageBlob);
      setResultUrl(newResultUrl);
      setNotice("Processing completed! Compare the original and result.");
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the backend. Check that FastAPI is running."
          : err.message || "Something went wrong while processing the image.",
      );
    } finally {
      setIsLoading(false);
    }
  };

  const downloadImage = (url, name) => {
    if (!url) return;

    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  return (
    <section className="editor-section" aria-labelledby="editor-title">
      <div className="editor-heading">
        <p className="editor-eyebrow">Photo workspace</p>
        <h2 id="editor-title">Upload an image to get started</h2>
        <p>
          Upload an old photo, process it with the backend, and compare the
          original with the restored result.
        </p>
      </div>

      <div className="editor-upload">
        <label className="upload-label" htmlFor="photo-upload">
          <span className="upload-icon" aria-hidden="true">
            ↑
          </span>
          <span>
            <strong>Choose a photo</strong>
            <small>JPG, PNG, or WebP</small>
          </span>
        </label>

        <input
          id="photo-upload"
          className="file-input"
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleFileChange}
          disabled={isLoading}
        />

        {fileName && (
          <span className="selected-file">Selected: {fileName}</span>
        )}
      </div>

      {error && (
        <p className="editor-message editor-error" role="alert">
          {error}
        </p>
      )}

      {notice && (
        <p className="editor-message editor-notice" role="status">
          {notice}
        </p>
      )}

      <div className="editor-workspace">
        <section className="preview-panel" aria-label="Original image preview">
          <div className="panel-heading">
            <h3>Original image</h3>
            <span>Input</span>
          </div>

          <div className={`image-stage${originalUrl ? " has-image" : ""}`}>
            {originalUrl ? (
              <img src={originalUrl} alt={`Original: ${fileName}`} />
            ) : (
              <p className="stage-message">
                Your selected photo will appear here.
              </p>
            )}
          </div>

          <button
            className="editor-button download-button"
            type="button"
            onClick={() =>
              downloadImage(originalUrl, fileName || "original.png")
            }
            disabled={!originalUrl || isLoading}
          >
            Download original
          </button>
        </section>

        <section className="controls-panel" aria-labelledby="controls-title">
          <div className="panel-heading">
            <h3 id="controls-title">Processing option</h3>
            <span>Optional</span>
          </div>

          <div className="editor-controls">
            <label className="noise-option" htmlFor="remove-noise">
              <input
                id="remove-noise"
                name="denoise"
                type="checkbox"
                checked={denoiseEnabled}
                onChange={(event) => setDenoiseEnabled(event.target.checked)}
                disabled={!selectedFile || isLoading}
              />
              <span>
                <strong>Remove image noise</strong>
                <small>Light denoising can reduce grain but may soften fine details.</small>
              </span>
            </label>
          </div>

          <button
            className="editor-button process-button"
            type="button"
            onClick={handleProcess}
            disabled={!selectedFile || isLoading}
          >
            {isLoading ? "Processing image…" : "Process image"}
          </button>

          <p className="control-note">
            Noise removal is off by default. Select it only when the photo has
            visible grain or noise.
          </p>
        </section>

        <section
          className="preview-panel result-panel"
          aria-label="Restored result preview"
        >
          <div className="panel-heading">
            <h3>Restored result</h3>
            <span>Output</span>
          </div>

          <div
            className={`image-stage result-stage${resultUrl ? " has-image" : ""}`}
          >
            {resultUrl ? (
              <img src={resultUrl} alt="Processed photo" />
            ) : (
              <p className="stage-message">
                Your processed photo will appear here.
              </p>
            )}
          </div>

          <button
            className="editor-button download-button"
            type="button"
            onClick={() =>
              downloadImage(
                resultUrl,
                `${fileName.replace(/\.[^.]+$/, "")}-restored.png`,
              )
            }
            disabled={!resultUrl || isLoading}
          >
            Download image
          </button>
        </section>
      </div>
    </section>
  );
};

export default Editor;
