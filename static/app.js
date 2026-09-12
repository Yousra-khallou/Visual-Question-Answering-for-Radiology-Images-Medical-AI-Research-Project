const API_URL = window.location.origin;

let currentFile = null;
let analysisHistory = JSON.parse(localStorage.getItem('medvqa_history') || '[]');
let loadingTimerInterval = null;
let loadingStartTime = null;

/* ============================================================
   THEME MANAGEMENT
   ============================================================ */
function initTheme() {
    const saved = localStorage.getItem('medvqa_theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = saved || (prefersDark ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', theme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('medvqa_theme', next);
}

/* ============================================================
   TOAST NOTIFICATIONS
   ============================================================ */
function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.setAttribute('role', 'alert');

    const icons = {
        success: '✓',
        error: '✕',
        info: 'ℹ'
    };

    toast.innerHTML = `
        <span class="toast-icon ${type}">${icons[type]}</span>
        <span>${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-out');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

/* ============================================================
   DRAG & DROP
   ============================================================ */
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

['dragover', 'dragenter'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
});

['dragleave', 'dragend'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
        dropZone.classList.remove('dragover');
    });
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length) handleFile(files[0]);
});

dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

/* ============================================================
   FILE HANDLING
   ============================================================ */
function handleFile(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/tiff', 'image/gif', 'application/dicom'];
    const validExts = ['.jpg', '.jpeg', '.png', '.webp', '.tiff', '.tif', '.gif', '.dcm'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    if (!validTypes.includes(file.type) && !validExts.includes(ext)) {
        showToast(`Format non supporté : ${ext}`, 'error');
        return;
    }

    if (file.size > 20 * 1024 * 1024) {
        showToast('Image trop volumineuse (max 20MB)', 'error');
        return;
    }

    currentFile = file;

    // Simulate upload progress
    const progressBar = document.querySelector('.dz-progress-bar');
    const progressContainer = document.getElementById('dzProgress');
    progressContainer.classList.add('active');

    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 30;
        if (progress >= 100) {
            progress = 100;
            clearInterval(interval);
            setTimeout(() => {
                progressContainer.classList.remove('active');
                progressBar.style.width = '0%';
            }, 300);
        }
        progressBar.style.width = progress + '%';
    }, 100);

    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImg').src = e.target.result;
        document.getElementById('preview').style.display = 'block';
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('previewFilename').textContent = file.name;

        // Get image dimensions
        const img = new Image();
        img.onload = () => {
            document.getElementById('previewDimensions').textContent = `${img.naturalWidth} × ${img.naturalHeight}`;
        };
        img.src = e.target.result;

        showToast(`Image chargée : ${file.name}`, 'success', 2000);
    };
    reader.readAsDataURL(file);
}

// Remove image
document.getElementById('removeImageBtn').addEventListener('click', () => {
    currentFile = null;
    document.getElementById('preview').style.display = 'none';
    document.getElementById('previewImg').src = '';
    document.getElementById('predictBtn').disabled = true;
    document.getElementById('fileInput').value = '';
    document.getElementById('gradcamControls').style.display = 'none';
    document.getElementById('previewOverlay').style.display = 'none';
    showToast('Image supprimée', 'info', 2000);
});

/* ============================================================
   QUESTION INPUT
   ============================================================ */
function setQuestion(q) {
    const input = document.getElementById('question');
    input.value = q;
    input.focus();
    // Visual feedback
    input.parentElement.style.animation = 'none';
    input.parentElement.offsetHeight; // trigger reflow
    input.parentElement.style.animation = 'pulse-border 0.4s ease';
}

// Add pulse-border animation via JS since we can't modify CSS
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse-border {
        0%, 100% { box-shadow: 0 0 0 0px rgba(45,156,219,0.2); }
        50% { box-shadow: 0 0 0 6px rgba(45,156,219,0.1); }
    }
`;
document.head.appendChild(style);

/* ============================================================
   PREDICTION
   ============================================================ */
async function predict() {
    if (!currentFile) return;

    const question = document.getElementById('question').value;
    if (!question.trim()) {
        showToast('Veuillez entrer une question clinique', 'error');
        document.getElementById('question').focus();
        return;
    }

    // Hide results/placeholder, show loading
    document.getElementById('results').style.display = 'none';
    document.getElementById('placeholder').style.display = 'none';
    document.getElementById('loading').style.display = 'flex';
    document.getElementById('predictBtn').disabled = true;

    // Reset pipeline steps
    document.querySelectorAll('.step').forEach(s => {
        s.classList.remove('active', 'completed');
    });
    document.getElementById('loadingProgressBar').style.width = '0%';

    // Start timer
    loadingStartTime = Date.now();
    const timerEl = document.getElementById('loadingTimer');
    loadingTimerInterval = setInterval(() => {
        const elapsed = ((Date.now() - loadingStartTime) / 1000).toFixed(1);
        timerEl.textContent = elapsed + 's';
    }, 100);

    // Simulate pipeline steps
    const steps = ['preprocess', 'vit', 'bert', 'fusion'];
    const stepDelays = [300, 800, 1200, 1800];

    steps.forEach((step, i) => {
        setTimeout(() => {
            const stepEl = document.querySelector(`[data-step="${step}"]`);
            if (stepEl) {
                // Mark previous as completed
                if (i > 0) {
                    const prev = document.querySelector(`[data-step="${steps[i-1]}"]`);
                    if (prev) prev.classList.add('completed');
                }
                stepEl.classList.add('active');
            }
            document.getElementById('loadingProgressBar').style.width = ((i + 1) / steps.length * 80) + '%';
        }, stepDelays[i]);
    });

    const formData = new FormData();
    formData.append('image', currentFile);
    formData.append('question', question);
    formData.append('explain', 'true');

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        const data = await response.json();

        // Complete all steps
        clearInterval(loadingTimerInterval);
        steps.forEach(s => {
            const el = document.querySelector(`[data-step="${s}"]`);
            el.classList.remove('active');
            el.classList.add('completed');
        });
        document.getElementById('loadingProgressBar').style.width = '100%';

        // Small delay for visual completion
        await new Promise(r => setTimeout(r, 400));

        // Hide loading, show results
        document.getElementById('loading').style.display = 'none';
        document.getElementById('results').style.display = 'block';

        // Populate results
        const answerText = document.getElementById('answerText');
        answerText.textContent = data.answer ? data.answer.toUpperCase() : '—';

        // Animate confidence
        // app.py renvoie confidence en 0-100 (déjà en pourcentage)
        const rawConf = data.confidence || 0;
        const confidence = rawConf > 1 ? rawConf / 100 : rawConf;  // normalise en 0-1
        const confidencePct = (confidence * 100).toFixed(1);

        document.getElementById('confidenceText').textContent = `Confiance : ${confidencePct}%`;
        if (document.getElementById('confidence'))
            document.getElementById('confidence').textContent = `Confiance : ${confidencePct}%`;

        // Animate bar
        const barFill = document.getElementById('answerBarFill');
        barFill.style.width = '0%';
        requestAnimationFrame(() => {
            barFill.style.width = confidencePct + '%';
            if (confidence > 0.5) barFill.classList.add('animate');
        });

        // Animate ring
        const ringFill = document.getElementById('confidenceRingFill');
        const circumference = 2 * Math.PI * 42; // r=42
        const offset = circumference - (confidence * circumference);
        ringFill.style.strokeDashoffset = circumference;
        requestAnimationFrame(() => {
            setTimeout(() => {
                ringFill.style.strokeDashoffset = offset;
            }, 100);
        });

        // Animate ring value
        const ringValue = document.getElementById('ringValue');
        let ringCounter = 0;
        const ringInterval = setInterval(() => {
            ringCounter += 2;
            if (ringCounter >= parseFloat(confidencePct)) {
                ringCounter = parseFloat(confidencePct);
                clearInterval(ringInterval);
            }
            ringValue.textContent = ringCounter.toFixed(0) + '%';
        }, 20);

        // Meta
        document.getElementById('metaType').textContent = data.answer_type || '—';
        document.getElementById('metaTime').textContent = (data.processing_time_ms || 0) + 'ms';
        document.getElementById('metaDate').textContent = new Date().toLocaleString('fr-FR', {
            day: '2-digit', month: '2-digit', year: 'numeric',
            hour: '2-digit', minute: '2-digit'
        });

        // Badge — app.py renvoie "Closed (oui/non/binaire)" ou "Open (médical)"
        const badge = document.getElementById('answerBadge');
        const atype = data.answer_type || '';
        badge.textContent = atype.includes('Closed') ? 'Binaire' :
                           atype.includes('Open')   ? 'Ouverte' : atype;

        // Grad-CAM — app.py renvoie 'cam_base64'
        const camData = data.cam_base64 || data.gradcam_base64;
        if (camData) {
            const gradcamSrc = 'data:image/jpeg;base64,' + camData;
            document.getElementById('gradcamImg').src = gradcamSrc;

            // Setup overlay
            const overlayImg = document.getElementById('gradcamOverlayImg');
            overlayImg.src = gradcamSrc;
            document.getElementById('gradcamControls').style.display = 'flex';
            document.getElementById('previewOverlay').style.display = 'block';

            // Default to overlay view
            switchGradcamView('overlay');
        } // end if camData

        // Add to history
        addToHistory({
            answer: data.answer,
            confidence: confidencePct,
            question: question,
            type: data.answer_type,
            time: data.processing_time_ms,
            timestamp: new Date().toISOString()
        });

        // Store for export
        storeResult({
            answer: data.answer,
            confidence: confidence,
            question: question,
            answer_type: data.answer_type,
            processing_time_ms: data.processing_time_ms,
            timestamp: new Date().toISOString()
        }, camData ? 'data:image/jpeg;base64,' + camData : null);

        showToast('Analyse terminée avec succès', 'success');

    } catch (err) {
        clearInterval(loadingTimerInterval);
        document.getElementById('loading').style.display = 'none';
        document.getElementById('placeholder').style.display = 'block';
        showToast('Erreur : ' + err.message, 'error', 6000);
        console.error(err);
    } finally {
        document.getElementById('predictBtn').disabled = false;
    }
}

/* ============================================================
   GRAD-CAM VIEW SWITCHING
   ============================================================ */
function switchGradcamView(view) {
    const tabs = document.querySelectorAll('.gradcam-tab');
    const panel = document.getElementById('gradcamPanel');
    const overlay = document.getElementById('previewOverlay');

    tabs.forEach(t => {
        t.classList.remove('active');
        t.setAttribute('aria-selected', 'false');
    });

    document.getElementById('tab' + view.charAt(0).toUpperCase() + view.slice(1)).classList.add('active');
    document.getElementById('tab' + view.charAt(0).toUpperCase() + view.slice(1)).setAttribute('aria-selected', 'true');

    if (view === 'overlay') {
        overlay.style.display = 'block';
        panel.style.opacity = '0';
        setTimeout(() => {
            panel.innerHTML = `
                <img id="gradcamImg" src="${document.getElementById('gradcamImg').src}" alt="Carte de chaleur Grad-CAM">
                <div class="gradcam-legend">
                    <span>Faible</span>
                    <div class="legend-bar"></div>
                    <span>Élevé</span>
                </div>
            `;
            panel.style.opacity = '1';
        }, 200);
    } else {
        overlay.style.display = 'none';
        panel.style.opacity = '0';
        setTimeout(() => {
            panel.innerHTML = `
                <img id="gradcamImg" src="${document.getElementById('gradcamImg').src}" alt="Carte de chaleur Grad-CAM isolée" style="background: var(--bg-section); padding: 20px;">
                <div class="gradcam-legend">
                    <span>Faible</span>
                    <div class="legend-bar"></div>
                    <span>Élevé</span>
                </div>
            `;
            panel.style.opacity = '1';
        }, 200);
    }
}

/* ============================================================
   GRAD-CAM OPACITY SLIDER
   ============================================================ */
document.getElementById('opacitySlider').addEventListener('input', (e) => {
    const val = e.target.value;
    document.getElementById('opacityValue').textContent = val + '%';
    document.getElementById('previewOverlay').style.opacity = val / 100;
});

/* ============================================================
   HISTORY MANAGEMENT
   ============================================================ */
function addToHistory(item) {
    analysisHistory.unshift(item);
    if (analysisHistory.length > 50) analysisHistory.pop();
    localStorage.setItem('medvqa_history', JSON.stringify(analysisHistory));
    updateHistoryUI();
}

function updateHistoryUI() {
    const count = document.getElementById('historyCount');
    const list = document.getElementById('historyList');

    count.textContent = analysisHistory.length;
    count.style.display = analysisHistory.length > 0 ? 'flex' : 'none';

    if (analysisHistory.length === 0) {
        list.innerHTML = '<div class="history-empty">Aucune analyse effectuée</div>';
        return;
    }

    list.innerHTML = analysisHistory.map((item, i) => `
        <div class="history-item" onclick="loadHistoryItem(${i})" role="button" tabindex="0">
            <div class="history-item-header">
                <span class="history-item-answer">${(item.answer || '—').toUpperCase()}</span>
                <span class="history-item-conf">${item.confidence}%</span>
            </div>
            <div class="history-item-question">${escapeHtml(item.question)}</div>
            <div class="history-item-meta">
                <span>${item.type || '—'}</span>
                <span>·</span>
                <span>${item.time}ms</span>
                <span>·</span>
                <span>${new Date(item.timestamp).toLocaleDateString('fr-FR')}</span>
            </div>
        </div>
    `).join('');
}

function loadHistoryItem(index) {
    const item = analysisHistory[index];
    if (!item) return;

    document.getElementById('question').value = item.question;
    showToast('Question chargée depuis l\'historique', 'info', 2000);
    closeHistory();
}

function clearHistory() {
    analysisHistory = [];
    localStorage.removeItem('medvqa_history');
    updateHistoryUI();
    showToast('Historique effacé', 'info', 2000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// History panel toggle
const historyPanel = document.getElementById('historyPanel');
const historyOverlay = document.getElementById('historyOverlay');

function openHistory() {
    historyPanel.classList.add('open');
    historyOverlay.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeHistory() {
    historyPanel.classList.remove('open');
    historyOverlay.classList.remove('active');
    document.body.style.overflow = '';
}

document.getElementById('historyBtn').addEventListener('click', openHistory);
document.getElementById('historyClose').addEventListener('click', closeHistory);
historyOverlay.addEventListener('click', closeHistory);
document.getElementById('clearHistoryBtn').addEventListener('click', clearHistory);

// Keyboard shortcut for history
document.addEventListener('keydown', (e) => {
    if (e.key === 'h' && !e.ctrlKey && !e.metaKey && !e.altKey) {
        if (historyPanel.classList.contains('open')) {
            closeHistory();
        } else {
            openHistory();
        }
    }
    if (e.key === 'Escape' && historyPanel.classList.contains('open')) {
        closeHistory();
    }
});

/* ============================================================
   EXPORT / TÉLÉCHARGEMENT DES RÉSULTATS
   ============================================================ */
let lastResultData = null;
let lastGradcamSrc = null;

function storeResult(data, gradcamSrc) {
    lastResultData = data;
    lastGradcamSrc = gradcamSrc;
}

function downloadJSON() {
    if (!lastResultData) {
        showToast('Aucun résultat à exporter', 'error');
        return;
    }
    const exportData = {
        ...lastResultData,
        exported_at: new Date().toISOString(),
        application: 'MedVQA N6',
        version: '3.0'
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `medvqa-result-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('Résultats exportés en JSON', 'success', 2000);
}

function downloadGradCAM() {
    if (!lastGradcamSrc) {
        showToast('Aucune image Grad-CAM à exporter', 'error');
        return;
    }
    const a = document.createElement('a');
    a.href = lastGradcamSrc;
    a.download = `medvqa-gradcam-${Date.now()}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    showToast('Grad-CAM téléchargée', 'success', 2000);
}

function downloadReport() {
    if (!lastResultData) {
        showToast('Aucun résultat à exporter', 'error');
        return;
    }

    // Charger jsPDF dynamiquement si pas encore chargé
    function generatePDF() {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
        const d = lastResultData;
        const date = new Date().toLocaleString('fr-FR');
        const W = 210; // largeur A4
        const margin = 20;
        const contentW = W - margin * 2;
        let y = 0;

        // ── Fond header ──
        doc.setFillColor(26, 122, 107);   // teal-dark
        doc.rect(0, 0, W, 42, 'F');

        // Titre
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(22);
        doc.setTextColor(255, 255, 255);
        doc.text('MedVQA N6', margin, 18);

        doc.setFont('helvetica', 'normal');
        doc.setFontSize(10);
        doc.setTextColor(180, 230, 225);
        doc.text('Rapport de Diagnostic Médical par IA', margin, 26);
        doc.text(`Généré le ${date}`, margin, 33);

        // Badge version
        doc.setFillColor(255, 255, 255, 0.15);
        doc.setDrawColor(255, 255, 255);
        doc.setLineWidth(0.4);
        doc.roundedRect(W - 50, 14, 30, 10, 2, 2, 'D');
        doc.setFontSize(8);
        doc.setTextColor(255, 255, 255);
        doc.text('v 3.0', W - 35, 20.5, { align: 'center' });

        y = 52;

        // ── Fonction helpers ──
        function sectionTitle(title, icon) {
            doc.setFillColor(232, 244, 243);
            doc.rect(margin, y, contentW, 9, 'F');
            doc.setFillColor(45, 156, 219);
            doc.rect(margin, y, 3, 9, 'F');
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(9);
            doc.setTextColor(26, 46, 53);
            doc.text(title.toUpperCase(), margin + 7, y + 6);
            y += 14;
        }

        function row(label, value, highlight) {
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9);
    doc.setTextColor(90, 122, 125);
    doc.text(label, margin + 4, y);
    doc.setFont('helvetica', highlight ? 'bold' : 'normal');
    doc.setFontSize(highlight ? 11 : 9);

    // ✅ CORRECTION : séparer le ternaire du setTextColor
    if (highlight) {
        doc.setTextColor(26, 122, 107);   // teal vert
    } else {
        doc.setTextColor(26, 46, 53);     // texte sombre
    }

    doc.text(String(value), margin + 55, y);
    y += 7;
}

        function divider() {
            doc.setDrawColor(213, 232, 230);
            doc.setLineWidth(0.3);
            doc.line(margin, y - 2, W - margin, y - 2);
        }

        // ── Section 1 : Question clinique ──
        sectionTitle('Question clinique');
        doc.setFont('helvetica', 'italic');
        doc.setFontSize(10);
        doc.setTextColor(26, 46, 53);
        const questionLines = doc.splitTextToSize(`"${d.question || 'Non spécifiée'}"`, contentW - 8);
        doc.text(questionLines, margin + 4, y);
        y += questionLines.length * 6 + 8;

        // ── Section 2 : Diagnostic ──
        sectionTitle('Diagnostic');

        // Grande réponse encadrée
        const confPct = (d.confidence * 100).toFixed(1);
        doc.setFillColor(240, 247, 247);
        doc.setDrawColor(45, 156, 219);
        doc.setLineWidth(0.5);
        doc.roundedRect(margin, y, contentW, 22, 3, 3, 'FD');

        doc.setFont('helvetica', 'bold');
        doc.setFontSize(18);
        doc.setTextColor(26, 122, 107);
        doc.text((d.answer || '—').toUpperCase(), margin + 6, y + 13);

        // Badge confiance
        const conf = parseFloat(confPct);
        const confColor = conf >= 70 ? [39, 174, 96] : conf >= 40 ? [243, 156, 18] : [231, 76, 60];
        doc.setFillColor(...confColor);
        doc.roundedRect(W - margin - 38, y + 5, 36, 12, 2, 2, 'F');
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(9);
        doc.setTextColor(255, 255, 255);
        doc.text(`${confPct}%`, W - margin - 20, y + 12.5, { align: 'center' });
        doc.setFontSize(7);
        doc.text('confiance', W - margin - 20, y + 15.5, { align: 'center' });

        y += 28;
        row('Type de réponse', d.answer_type || '—');
        y += 3; divider(); y += 5;

        // ── Section 3 : Infos techniques ──
        sectionTitle('Informations techniques');
        row('Temps de traitement', `${d.processing_time_ms || 0} ms`);
        row('Modèle visuel', 'ViT-B/16 (ImageNet-21k)');
        row('Modèle textuel', 'BiomedBERT (PubMed abstracts)');
        row('Fusion', 'Projection + Concat + MLP');
        row('Explainability', 'Grad-CAM (ViT norm layer)');
        row('Datasets', 'VQA-RAD + SLAKE (~7 000 samples)');
        row('Val Combined', '73.7% (epoch 29)');
        y += 3; divider(); y += 5;

        // ── Grad-CAM image ──
        if (lastGradcamSrc) {
            sectionTitle('Carte d\'activation Grad-CAM');
            try {
                const imgW = 70;
                const imgH = 70;
                const imgX = W / 2 - imgW / 2;
                doc.addImage(lastGradcamSrc, 'JPEG', imgX, y, imgW, imgH);
                doc.setFont('helvetica', 'normal');
                doc.setFontSize(7.5);
                doc.setTextColor(143, 168, 170);
                doc.text('Rouge = forte activation · Bleu = faible activation', W / 2, y + imgH + 5, { align: 'center' });
                y += imgH + 14;
            } catch(e) { y += 4; }
        }

        // ── Footer disclaimer ──
        const footerY = 275;
        doc.setFillColor(240, 247, 247);
        doc.rect(0, footerY, W, 22, 'F');
        doc.setDrawColor(213, 232, 230);
        doc.setLineWidth(0.3);
        doc.line(0, footerY, W, footerY);
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(7.5);
        doc.setTextColor(143, 168, 170);
        doc.text(
            'Ce rapport a été généré automatiquement par MedVQA N6 · ENSAH 2025–2026',
            W / 2, footerY + 7, { align: 'center' }
        );
        doc.text(
            'Les résultats sont fournis à titre indicatif et ne remplacent pas l\'avis d\'un professionnel de santé.',
            W / 2, footerY + 13, { align: 'center' }
        );

        // Numéro de page
        doc.setFontSize(7);
        doc.setTextColor(200, 215, 214);
        doc.text('Page 1 / 1', W - margin, footerY + 10, { align: 'right' });

        doc.save(`medvqa-rapport-${Date.now()}.pdf`);
        showToast('Rapport PDF téléchargé', 'success', 2000);
    }

    // Charger jsPDF depuis CDN si nécessaire
    if (window.jspdf) {
        generatePDF();
    } else {
        showToast('Chargement de jsPDF...', 'info', 2000);
        const script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js';
        script.onload = generatePDF;
        script.onerror = () => showToast('Erreur chargement jsPDF', 'error');
        document.head.appendChild(script);
    }
}

/* ============================================================
   KEYBOARD SHORTCUTS
   ============================================================ */
document.addEventListener('keydown', (e) => {
    // Ctrl+Enter to predict
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (!document.getElementById('predictBtn').disabled) {
            predict();
        }
    }
    // Escape to clear
    if (e.key === 'Escape' && document.activeElement === document.getElementById('question')) {
        document.getElementById('question').blur();
    }
});

/* ============================================================
   INITIALIZATION
   ============================================================ */
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    updateHistoryUI();

    // Theme toggle
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);

    // Add ring gradient for dark mode compatibility
    const svgNS = 'http://www.w3.org/2000/svg';
    const defs = document.createElementNS(svgNS, 'defs');
    const grad = document.createElementNS(svgNS, 'linearGradient');
    grad.setAttribute('id', 'ringGradient');
    grad.setAttribute('x1', '0%');
    grad.setAttribute('y1', '0%');
    grad.setAttribute('x2', '100%');
    grad.setAttribute('y2', '100%');

    const stop1 = document.createElementNS(svgNS, 'stop');
    stop1.setAttribute('offset', '0%');
    stop1.setAttribute('style', 'stop-color:var(--accent-1);stop-opacity:1');

    const stop2 = document.createElementNS(svgNS, 'stop');
    stop2.setAttribute('offset', '100%');
    stop2.setAttribute('style', 'stop-color:var(--accent-2);stop-opacity:1');

    grad.appendChild(stop1);
    grad.appendChild(stop2);
    defs.appendChild(grad);

    const ringSvg = document.querySelector('.answer-confidence-ring svg');
    if (ringSvg) ringSvg.prepend(defs);

    // Auto-focus question on load if no image
    if (!currentFile) {
        setTimeout(() => document.getElementById('question').focus(), 500);
    }
});