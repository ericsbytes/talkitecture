const uploadInput = document.getElementById('upload');
const analyzeButton = document.getElementById('analyze');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const factsDiv = document.getElementById('facts');

let analysisData = null;
let currentFrame = 0;

analyzeButton.addEventListener('click', async () => {
    const file = uploadInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:8000/analyze-video', {
            method: 'POST',
            body: formData
        });
        analysisData = await response.json();
        displayFacts();
        setupVideo(file);
    } catch (error) {
        console.error('Error:', error);
    }
});

function displayFacts() {
    factsDiv.innerHTML = `<h2>${analysisData.landmark.name}</h2>
    <ul>${analysisData.landmark.facts.map(fact => `<li>${fact}</li>`).join('')}</ul>`;
    
    // Speak the facts
    const text = `${analysisData.landmark.name}. ${analysisData.landmark.facts.join('. ')}`;
    speakText(text);
}

let isSpeaking = false;

function speakText(text) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.onstart = () => { isSpeaking = true; };
        utterance.onend = () => { isSpeaking = false; };
        window.speechSynthesis.speak(utterance);
    }
}

function setupVideo(file) {
    const url = URL.createObjectURL(file);
    video.src = url;
    video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        video.currentTime = 0;
        video.play();
    };
}

video.addEventListener('timeupdate', () => {
    drawFrame();
});

let blinkTimer = 0;
let isBlinking = false;

function drawFrame() {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    if (analysisData && analysisData.tracking) {
        const frameIndex = Math.floor(video.currentTime * 30); // Assuming 30 fps
        const track = analysisData.tracking[frameIndex % analysisData.tracking.length];
        if (track) {
            // Draw eyes with blinking
            drawEye(track.x - 30, track.y - 20, 10, isBlinking);
            drawEye(track.x + 30, track.y - 20, 10, isBlinking);
            
            // Draw mouth
            drawMouth(track.x, track.y + 20, 20);
        }
    }
    
    // Handle blinking
    blinkTimer++;
    if (blinkTimer > 60) { // Blink every ~2 seconds at 30fps
        isBlinking = true;
        if (blinkTimer > 65) {
            isBlinking = false;
            blinkTimer = 0;
        }
    }
}

function drawEye(x, y, radius, blinking) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = 'white';
    ctx.fill();
    ctx.stroke();
    
    if (!blinking) {
        // Pupil
        ctx.beginPath();
        ctx.arc(x, y, radius / 2, 0, 2 * Math.PI);
        ctx.fillStyle = 'black';
        ctx.fill();
    } else {
        // Closed eye
        ctx.beginPath();
        ctx.moveTo(x - radius, y);
        ctx.lineTo(x + radius, y);
        ctx.stroke();
    }
}

function drawMouth(x, y, width) {
    if (isSpeaking) {
        // Animate mouth
        const time = Date.now() * 0.01;
        const openAmount = Math.sin(time) * 0.5 + 0.5; // 0 to 1
        ctx.beginPath();
        ctx.ellipse(x, y, width / 2, (width / 2) * openAmount, 0, 0, Math.PI);
        ctx.stroke();
    } else {
        ctx.beginPath();
        ctx.arc(x, y, width / 2, 0, Math.PI);
        ctx.stroke();
    }
}