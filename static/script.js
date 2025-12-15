// static/script.js

const fileInput = document.getElementById('fileInput');
const previewImg = document.getElementById('preview-img');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const predictBtn = document.getElementById('predict-btn');
const loadingDiv = document.getElementById('loading');
const resultCard = document.getElementById('result-card');

// 1. Handle Image Preview
fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImg.src = e.target.result;
            previewImg.classList.remove('hidden');
            uploadPlaceholder.classList.add('hidden');
            predictBtn.classList.remove('hidden');
            
            // Reset result if any
            resultCard.classList.add('hidden');
        }
        reader.readAsDataURL(file);
    }
});

// 2. Handle Switch Model (Blue/Green)
async function switchModel(color) {
    const btnBlue = document.getElementById('btn-blue');
    const btnGreen = document.getElementById('btn-green');

    // Visual Update
    if (color === 'blue') {
        setActiveStyle(btnBlue);
        setInactiveStyle(btnGreen);
    } else {
        setActiveStyle(btnGreen);
        setInactiveStyle(btnBlue);
    }

    // Call API
    try {
        const response = await fetch(`/admin/switch-model?color=${color}`, { method: 'POST' });
        const data = await response.json();
        console.log("Model Switched:", data);
    } catch (error) {
        alert("Gagal mengganti model. Pastikan server berjalan.");
    }
}

function setActiveStyle(btn) {
    btn.className = "px-4 py-1.5 rounded-full text-sm font-semibold transition-all shadow-sm bg-white text-indigo-600";
}

function setInactiveStyle(btn) {
    btn.className = "px-4 py-1.5 rounded-full text-sm font-semibold transition-all text-slate-500 hover:text-indigo-600";
}

// 3. Handle Prediction Request
async function predict() {
    const file = fileInput.files[0];
    if (!file) return;

    // Show Loading
    predictBtn.disabled = true;
    loadingDiv.classList.remove('hidden');
    resultCard.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Prediksi Gagal');

        const data = await response.json();
        displayResult(data);

    } catch (error) {
        alert("Terjadi kesalahan: " + error.message);
    } finally {
        predictBtn.disabled = false;
        loadingDiv.classList.add('hidden');
    }
}

// 4. Display Result
function displayResult(data) {
    const pred = data.prediction;
    
    document.getElementById('res-common-name').innerText = pred.common_name;
    document.getElementById('res-latin-name').innerText = pred.latin_name;
    document.getElementById('res-confidence').innerText = pred.confidence;
    document.getElementById('res-desc').innerText = pred.description;
    document.getElementById('res-model-used').innerText = `Model: ${data.model_used.toUpperCase()}`;

    resultCard.classList.remove('hidden');
    
    // Smooth scroll to result
    resultCard.scrollIntoView({ behavior: 'smooth' });
}