// AR Mode JavaScript for dedicated AR page
const camera = document.getElementById('camera');
const overlay = document.getElementById('overlay');
const landmarkInfo = document.getElementById('landmarkInfo');
const landmarkName = document.getElementById('landmarkName');
const landmarkFacts = document.getElementById('landmarkFacts');
const statusDiv = document.getElementById('status');
const sensorDataDiv = document.getElementById('sensorData');

let stream = null;
let watchId = null;
let orientation = { alpha: 0, beta: 0, gamma: 0 };
let gpsData = { latitude: 0, longitude: 0, accuracy: 0 };
let isInitialized = false;
let websocket = null;
let lastFrameTime = 0;
const FRAME_RATE_LIMIT = 3000 / 10; // 30 FPS

function updateSensorDisplay() {
    const gpsText = gpsData.latitude && gpsData.longitude
        ? `${gpsData.latitude.toFixed(4)}, ${gpsData.longitude.toFixed(4)} (±${gpsData.accuracy.toFixed(0)}m)`
        : '--';

    const orientationText = orientation.alpha || orientation.beta || orientation.gamma
        ? `α:${orientation.alpha.toFixed(0)}° β:${orientation.beta.toFixed(0)}° γ:${orientation.gamma.toFixed(0)}°`
        : '--';

    const connectionText = websocket && websocket.readyState === WebSocket.OPEN
        ? 'WebSocket'
        : 'HTTP';

    sensorDataDiv.innerHTML = `
        <div>GPS: ${gpsText}</div>
        <div>Orientation: ${orientationText}</div>
        <div>Connection: ${connectionText}</div>
    `;
}

async function initAR() {
    try {
        statusDiv.textContent = 'Requesting camera access...';

        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' },
            audio: false
        });

        camera.srcObject = stream;
        await camera.play();

        overlay.width = camera.videoWidth;
        overlay.height = camera.videoHeight;

        statusDiv.textContent = 'Initializing GPS...';

        if ('geolocation' in navigator) {
            watchId = navigator.geolocation.watchPosition(
                (pos) => {
                    gpsData.latitude = pos.coords.latitude;
                    gpsData.longitude = pos.coords.longitude;
                    gpsData.accuracy = pos.coords.accuracy;
                    statusDiv.textContent = `GPS: ${gpsData.latitude.toFixed(4)}, ${gpsData.longitude.toFixed(4)}`;
                    updateSensorDisplay();
                },
                (error) => {
                    console.error('GPS error:', error);
                    statusDiv.textContent = 'GPS unavailable';
                    updateSensorDisplay();
                },
                { enableHighAccuracy: true, maximumAge: 1000 }
            );
        }

        statusDiv.textContent = 'Initializing orientation...';

        if ('DeviceOrientationEvent' in window) {
            if (typeof DeviceOrientationEvent.requestPermission === 'function') {
                const permissionButton = document.createElement('button');
                permissionButton.textContent = 'Enable Device Orientation';
                permissionButton.style.cssText = `
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    padding: 15px 20px;
                    background: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-size: 16px;
                    cursor: pointer;
                    z-index: 1000;
                `;

                permissionButton.onclick = async () => {
                    try {
                        const permission = await DeviceOrientationEvent.requestPermission();
                        if (permission === 'granted') {
                            permissionButton.remove();
                            window.addEventListener('deviceorientation', (event) => {
                                orientation.alpha = event.alpha || 0;
                                orientation.beta = event.beta || 0;
                                orientation.gamma = event.gamma || 0;
                                updateSensorDisplay();
                            });
                            statusDiv.textContent = 'Orientation enabled - connecting to AR service...';
                            setupWebSocket();
                        } else {
                            permissionButton.textContent = 'Permission Denied - Orientation unavailable';
                            setTimeout(() => permissionButton.remove(), 3000);
                        }
                    } catch (error) {
                        permissionButton.textContent = 'Permission Failed';
                        setTimeout(() => permissionButton.remove(), 3000);
                    }
                };

                document.body.appendChild(permissionButton);
                statusDiv.textContent = 'Tap "Enable Device Orientation" to continue';
                return;
            }

            window.addEventListener('deviceorientation', (event) => {
                orientation.alpha = event.alpha || 0;
                orientation.beta = event.beta || 0;
                orientation.gamma = event.gamma || 0;
                updateSensorDisplay();
            });

            setupWebSocket();
        } else {
            statusDiv.textContent = 'Device orientation not supported';
        }

        isInitialized = true;
        updateSensorDisplay();
        statusDiv.textContent = 'Connecting to AR service...';
        setupWebSocket();

    } catch (error) {
        console.error('AR initialization failed:', error);
        statusDiv.textContent = `Error: ${error.message}`;
    }
}

function setupWebSocket() {
    if (!websocket || websocket.readyState === WebSocket.CLOSED) {
        const wsUrl = `ws://localhost:8000/ar-stream`;
        websocket = new WebSocket(wsUrl);

        websocket.onopen = () => {
            statusDiv.textContent = 'AR mode active - scanning for landmarks...';
            updateSensorDisplay();
            processARFrame();
        };

        websocket.onmessage = (event) => {
            try {
                const result = JSON.parse(event.data);
                updateAROverlay(result);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };

        websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            statusDiv.textContent = 'WebSocket error - falling back to HTTP mode';
            updateSensorDisplay();
            websocket = null;
            setTimeout(() => processARFrame(), 1000);
        };

        websocket.onclose = (event) => {
            statusDiv.textContent = 'Connection lost - attempting to reconnect...';
            updateSensorDisplay();
            websocket = null;
            setTimeout(() => {
                if (isInitialized) {
                    setupWebSocket();
                }
            }, 2000);
        };
    }
}

async function processARFrame() {
    if (!isInitialized) return;

    const now = Date.now();
    if (now - lastFrameTime < FRAME_RATE_LIMIT) {
        requestAnimationFrame(processARFrame);
        return;
    }
    lastFrameTime = now;

    if (!websocket || websocket.readyState !== WebSocket.OPEN) {
        overlay.toBlob(async (imageBlob) => {
            const formData = new FormData();
            formData.append('frame', imageBlob, 'frame.jpg');
            formData.append('latitude', gpsData.latitude.toString());
            formData.append('longitude', gpsData.longitude.toString());
            formData.append('accuracy', gpsData.accuracy.toString());
            formData.append('alpha', orientation.alpha.toString());
            formData.append('beta', orientation.beta.toString());
            formData.append('gamma', orientation.gamma.toString());

            try {
                const response = await fetch(`http://localhost:8000/analyze-ar-frame`, {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const result = await response.json();
                    updateAROverlay(result);
                } else {
                    console.error('Frame analysis failed:', response.status);
                }
            } catch (error) {
                console.error('Frame analysis failed:', error);
            }
        });
    } else {
        overlay.toBlob((imageBlob) => {
            const reader = new FileReader();
            reader.onload = () => {
                const base64Data = reader.result.split(',')[1];

                const frameData = {
                    frame: base64Data,
                    latitude: gpsData.latitude,
                    longitude: gpsData.longitude,
                    accuracy: gpsData.accuracy,
                    alpha: orientation.alpha,
                    beta: orientation.beta,
                    gamma: orientation.gamma
                };

                try {
                    websocket.send(JSON.stringify(frameData));
                } catch (error) {
                    console.error('Failed to send frame via WebSocket:', error);
                }
            };
            reader.readAsDataURL(imageBlob);
        });
    }

    requestAnimationFrame(processARFrame);
}

function updateAROverlay(result) {
    if (result.error) {
        landmarkInfo.style.display = 'none';
        return;
    }

    // Update landmark info
    landmarkName.textContent = result.landmark.name;
    landmarkFacts.innerHTML = result.landmark.facts.map(fact => `<div>• ${fact}</div>`).join('');

    // Show landmark info
    landmarkInfo.style.display = 'block';

    // Clear previous overlays
    overlay.innerHTML = '';

    // Add face overlay if face detected
    if (result.face_region) {
        const [x, y, w, h] = [
            result.face_region.x,
            result.face_region.y,
            result.face_region.width,
            result.face_region.height
        ];

        // Scale coordinates to screen size
        const scaleX = overlay.width / camera.videoWidth;
        const scaleY = overlay.height / camera.videoHeight;

        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        const scaledW = w * scaleX;
        const scaledH = h * scaleY;

        // Create face bounding box
        const faceDiv = document.createElement('div');
        faceDiv.className = 'face-overlay';
        faceDiv.style.left = `${scaledX}px`;
        faceDiv.style.top = `${scaledY}px`;
        faceDiv.style.width = `${scaledW}px`;
        faceDiv.style.height = `${scaledH}px`;
        overlay.appendChild(faceDiv);
    }
}

// Initialize AR when page loads
window.addEventListener('load', () => {
    updateSensorDisplay(); // Initialize display
    initAR();
});

// Cleanup when leaving page
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    if (watchId) {
        navigator.geolocation.clearWatch(watchId);
    }
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.close();
    }
});