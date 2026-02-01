const uploadInput = document.getElementById('upload');
const analyzeButton = document.getElementById('analyze');
const arButton = document.getElementById('arMode');
const video = document.getElementById('video');
const camera = document.getElementById('camera');
const canvas = document.getElementById('canvas');
const arCanvas = document.getElementById('arCanvas');
const ctx = canvas.getContext('2d');
const arCtx = arCanvas.getContext('2d');
const factsDiv = document.getElementById('facts');

let analysisData = null;
let currentFrame = 0;
let arMode = false;
let stream = null;
let watchId = null;
let orientation = { alpha: 0, beta: 0, gamma: 0 };
let position = { latitude: 0, longitude: 0 };

async function startAR() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' },
            audio: false
        });

        camera.srcObject = stream;
        await camera.play();

        arCanvas.width = camera.videoWidth;
        arCanvas.height = camera.videoHeight;
        arCanvas.style.display = 'block';

        if ('geolocation' in navigator) {
            watchId = navigator.geolocation.watchPosition(
                (pos) => {
                    position.latitude = pos.coords.latitude;
                    position.longitude = pos.coords.longitude;
                },
                (error) => console.error('GPS Error:', error),
                { enableHighAccuracy: true, maximumAge: 10000 }
            );
        }

        if ('DeviceOrientationEvent' in window) {
            window.addEventListener('deviceorientation', (event) => {
                orientation.alpha = event.alpha || 0;
                orientation.beta = event.beta || 0;
                orientation.gamma = event.gamma || 0;
            });
        }

        arMode = true;
        arButton.textContent = '❌ Stop AR';
        arButton.classList.add('active');

        processARFrame();

    } catch (error) {
        console.error('AR initialization failed:', error);
        alert('AR mode requires camera and location permissions');
    }
}

function stopAR() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    if (watchId) {
        navigator.geolocation.clearWatch(watchId);
        watchId = null;
    }

    camera.srcObject = null;
    arCanvas.style.display = 'none';
    arMode = false;
    arButton.textContent = '🎥 AR Mode';
    arButton.classList.remove('active');
}

async function processARFrame() {
    if (!arMode) return;

    arCtx.drawImage(camera, 0, 0, arCanvas.width, arCanvas.height);

    arCanvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');
        formData.append('latitude', position.latitude.toString());
        formData.append('longitude', position.longitude.toString());
        formData.append('heading', orientation.alpha.toString());
        formData.append('pitch', orientation.beta.toString());
        formData.append('roll', orientation.gamma.toString());

        try {
            const response = await fetch(`${window.location.protocol}//${window.location.hostname}:8000/analyze-ar`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const arData = await response.json();
                drawAROverlay(arData);
            }
        } catch (error) {
            console.error('AR analysis failed:', error);
        }
    });

    // Continue processing frames
    requestAnimationFrame(processARFrame);
}

function drawAROverlay(arData) {
    if (!arData || !arData.landmark) return;

    // Clear previous overlay
    arCtx.clearRect(0, 0, arCanvas.width, arCanvas.height);
    arCtx.drawImage(camera, 0, 0);

    // Draw face elements if we have tracking data
    if (arData.tracking && arData.tracking.length > 0) {
        const track = arData.tracking[0]; // Use first frame data

        // Draw eyes
        drawEye(track.x - 30, track.y - 20, 10, false);
        drawEye(track.x + 30, track.y - 20, 10, false);

        // Draw mouth
        drawMouth(track.x, track.y + 20, 20);
    }

    // Display landmark info
    arCtx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    arCtx.fillRect(10, 10, 300, 80);
    arCtx.fillStyle = 'white';
    arCtx.font = '16px Arial';
    arCtx.fillText(`🏛️ ${arData.landmark.name}`, 20, 30);
    arCtx.font = '12px Arial';
    arData.landmark.facts.slice(0, 2).forEach((fact, i) => {
        arCtx.fillText(`• ${fact}`, 20, 50 + i * 15);
    });
}

analyzeButton.addEventListener('click', async () => {
    const file = uploadInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${window.location.protocol}//${window.location.hostname}:8000/analyze-video`, {
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
    if (arMode) {
        // In AR mode, facts are displayed on the AR overlay
        return;
    }

    factsDiv.innerHTML = `<ul>${analysisData.landmark.facts.map(fact => `<li>${fact}</li>`).join('')}</ul>`;
    
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
    const context = arMode ? arCtx : ctx;
    context.beginPath();
    context.arc(x, y, radius, 0, 2 * Math.PI);
    context.fillStyle = 'white';
    context.fill();
    context.stroke();

    if (!blinking) {
        // Pupil
        context.beginPath();
        context.arc(x, y, radius / 2, 0, 2 * Math.PI);
        context.fillStyle = 'black';
        context.fill();
    } else {
        // Closed eye
        context.beginPath();
        context.moveTo(x - radius, y);
        context.lineTo(x + radius, y);
        context.stroke();
    }
}

function drawMouth(x, y, width) {
    const context = arMode ? arCtx : ctx;
    if (isSpeaking) {
        // Animate mouth
        const time = Date.now() * 0.01;
        const openAmount = Math.sin(time) * 0.5 + 0.5; // 0 to 1
        context.beginPath();
        context.ellipse(x, y, width / 2, (width / 2) * openAmount, 0, 0, Math.PI);
        context.stroke();
    } else {
        context.beginPath();
        context.arc(x, y, width / 2, 0, Math.PI);
        context.stroke();
    }
}