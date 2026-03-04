const PREDICT_URL = 'http://127.0.0.1:8000/predict';

const uploadTrigger = document.getElementById('uploadTrigger');
const fileInput = document.getElementById('fileInput');
const fileNameEl = document.getElementById('fileName');
const preview = document.getElementById('preview');
const analyzeBtn = document.getElementById('analyzeBtn');
const btnText = document.getElementById('btnText');
const spinner = document.getElementById('spinner');
const resultCard = document.getElementById('resultCard');
const resultValue = document.getElementById('resultValue');
const progressWrap = document.getElementById('progressWrap');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
let selectedImageFile = null;

function setProgress(percent) {
  const clamped = Math.max(0, Math.min(100, Math.round(percent)));
  if (progressBar) progressBar.style.width = `${clamped}%`;
  if (progressText) progressText.textContent = `${clamped}%`;
}

function postWithProgress(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url, true);

    xhr.upload.onprogress = (evt) => {
      if (!evt.lengthComputable) return;
      const pct = (evt.loaded / evt.total) * 100;
      onProgress(pct);
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch (e) {
          reject(new Error('Invalid JSON response from server'));
        }
      } else {
        reject(new Error(xhr.responseText || `Request failed with status ${xhr.status}`));
      }
    };

    xhr.onerror = () => reject(new Error('Network error while uploading image. Ensure API is running at http://127.0.0.1:8000'));
    xhr.send(formData);
  });
}

if (uploadTrigger && fileInput) {
  uploadTrigger.addEventListener('click', (event) => {
    event.preventDefault();
    fileInput.click();
  });
  fileInput.addEventListener('change', () => {
    const f = fileInput.files && fileInput.files[0];
    if (!f) return;
    selectedImageFile = f;
    if (fileNameEl) fileNameEl.textContent = f.name;
    const url = URL.createObjectURL(f);
    if (preview) {
      preview.src = url;
      preview.style.display = 'block';
    }
  });
}

if (analyzeBtn) {
  analyzeBtn.addEventListener('click', async (event) => {
    event.preventDefault();
    const age = document.getElementById('ageInput')?.value;
    const sex = document.getElementById('sexInput')?.value;
    const files = fileInput?.files;
    const activeFile = selectedImageFile || (files && files[0] ? files[0] : null);

    if (!activeFile) {
      alert('Please upload a photo first.');
      return;
    }

    const form = new FormData();
    form.append('image', activeFile);
    if (age) form.append('age', age);
    if (sex) form.append('sex', sex);

    if (btnText) btnText.textContent = 'Submitting...';
    if (spinner) spinner.classList.add('spinning');
    if (progressWrap) progressWrap.style.display = 'block';
    setProgress(0);

    try {
      const data = await postWithProgress(PREDICT_URL, form, (pct) => setProgress(pct));
      setProgress(100);
      const pred = data.prediction ?? data.result ?? JSON.stringify(data);
      if (resultValue) resultValue.textContent = pred;
      if (resultCard) resultCard.classList.remove('hidden');
    } catch (err) {
      alert('Error: ' + (err.message || err));
    } finally {
      if (btnText) btnText.textContent = 'Submit for Analysis';
      if (spinner) spinner.classList.remove('spinning');
      setTimeout(() => {
        if (progressWrap) progressWrap.style.display = 'none';
      }, 500);
    }
  });
}