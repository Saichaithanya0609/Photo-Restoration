import { useEffect, useState } from "react";
import "./Editor.css";

const Editor = () => {
  const [originalUrl, setOriginalUrl] = useState("");
  const [fileName, setFileName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [settings, setSettings] = useState({
    contrast: 50,
    noiseReduction: 0,
    sharpening: 0,
  });

  useEffect(() => {
    if (!originalUrl) return undefined;
    return () => URL.revokeObjectURL(originalUrl);
  }, [originalUrl]);

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    setError("");
    setNotice("");
    setOriginalUrl("");
    setFileName("");
    setIsLoading(false);
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please choose a valid image file.");
      event.target.value = "";
      return;
    }
    setOriginalUrl(URL.createObjectURL(file));
    setFileName(file.name);
    setIsLoading(true);
  };

  const handleDownloadOriginal = () => {
    if (!originalUrl) return;
    const link = document.createElement("a");
    link.href = originalUrl;
    link.download = fileName || "original-image";
    link.click();
  };

  const handleProcess = () => {
    setNotice(
      "Processing is not connected yet. Your original image has not been changed.",
    );
  };

  const handleSettingChange = (event) => {
    const { name, value } = event.target;
    setSettings((current) => ({ ...current, [name]: Number(value) }));
  };

  return (
    <section className="editor-section" aria-labelledby="editor-title">
      <div className="editor-heading">
        <p className="editor-eyebrow">Photo workspace</p>
        <h2 id="editor-title">Upload an image to get started</h2>
        <p>
          Preview your original and choose enhancement settings. Processing will
          be available when the image service is connected.
        </p>
      </div>
      <div className="editor-upload">
        <label className="upload-label" htmlFor="photo-upload">
          <span className="upload-icon" aria-hidden="true">
            ↑
          </span>
          <span>
            <strong>Choose a photo</strong>
            <small>JPG, PNG, WebP or another browser-supported image</small>
          </span>
        </label>
        <input
          id="photo-upload"
          className="file-input"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
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
              <>
                <img
                  src={originalUrl}
                  alt={`Original: ${fileName}`}
                  onLoad={() => setIsLoading(false)}
                  onError={() => {
                    setIsLoading(false);
                    setError(
                      "This image could not be loaded. Please choose another file.",
                    );
                  }}
                />
                {isLoading && (
                  <p className="stage-message">Loading image preview…</p>
                )}
              </>
            ) : (
              <p className="stage-message">
                Your selected photo will appear here.
              </p>
            )}
          </div>
          <button
            className="editor-button download-button"
            type="button"
            onClick={handleDownloadOriginal}
            disabled={!originalUrl || isLoading}
          >
            Download original
          </button>
        </section>

        <section className="controls-panel" aria-labelledby="controls-title">
          <div className="panel-heading">
            <h3 id="controls-title">Enhancement controls</h3>
            <span>Settings</span>
          </div>
          <div className="editor-controls">
            <label className="range-control" htmlFor="contrast">
              <span>
                <strong>Contrast enhancement</strong>
                <output>{settings.contrast}%</output>
              </span>
              <input
                id="contrast"
                name="contrast"
                type="range"
                min="0"
                max="100"
                value={settings.contrast}
                onChange={handleSettingChange}
                disabled={!originalUrl}
              />
            </label>
            <label className="range-control" htmlFor="noiseReduction">
              <span>
                <strong>Noise reduction</strong>
                <output>{settings.noiseReduction}%</output>
              </span>
              <input
                id="noiseReduction"
                name="noiseReduction"
                type="range"
                min="0"
                max="100"
                value={settings.noiseReduction}
                onChange={handleSettingChange}
                disabled={!originalUrl}
              />
            </label>
            <label className="range-control" htmlFor="sharpening">
              <span>
                <strong>Sharpening</strong>
                <output>{settings.sharpening}%</output>
              </span>
              <input
                id="sharpening"
                name="sharpening"
                type="range"
                min="0"
                max="100"
                value={settings.sharpening}
                onChange={handleSettingChange}
                disabled={!originalUrl}
              />
            </label>
          </div>
          <button
            className="editor-button process-button"
            type="button"
            onClick={handleProcess}
            disabled={!originalUrl || isLoading}
            title="Image processing will be enabled when the backend is connected"
          >
            {isLoading ? "Loading image…" : "Process image"}
          </button>
          <p className="control-note">
            Processing is not connected yet. Your selected settings are not
            applied to the image.
          </p>
        </section>

        <section
          className="preview-panel result-panel"
          aria-label="Enhanced result preview"
        >
          <div className="panel-heading">
            <h3>Enhanced result</h3>
            <span>Output</span>
          </div>
          <div className="image-stage result-stage">
            <p className="stage-message">
              No processed result yet. The preview will appear here after
              processing is connected.
            </p>
          </div>
          <button
            className="editor-button download-button"
            type="button"
            disabled
            title="Download will be available after an enhanced result exists"
          >
            Download image
          </button>
        </section>
      </div>
    </section>
  );
};

export default Editor;
