// AR Mode JavaScript for dedicated AR page
const camera = document.getElementById('camera');
const overlay = document.getElementById('overlay');
const landmarkInfo = document.getElementById('landmarkInfo');
const landmarkName = document.getElementById('landmarkName');
const landmarkFacts = document.getElementById('landmarkFacts');
const statusDiv = document.getElementById('status');
const sensorDataDiv = document.getElementById('sensorData');
const debugLogDiv = document.getElementById('debugLog');

let stream = null;
let watchId = null;
let deviceOrientation = { alpha: 0, beta: 0, gamma: 0 };
let gpsData = { latitude: 0, longitude: 0, accuracy: 0 };
let gpsSamples = [];
const GPS_MAX_ACCURACY_METERS = 50;
const GPS_SMOOTHING_WINDOW = 5;
let isInitialized = false;
let websocket = null;
let lastFrameTime = 0;
let sendIntervalId = null;
let useApi = false; // API integration toggle - start disabled for testing local landmarks
const FRAME_RATE_LIMIT = 3000 / 10; // 30 FPS

function logDebug(message) {
    if (!debugLogDiv) return;
    const entry = document.createElement('div');
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    debugLogDiv.appendChild(entry);
    debugLogDiv.scrollTop = debugLogDiv.scrollHeight;
}

function sendSensorData() {
    if (!websocket || websocket.readyState !== WebSocket.OPEN) return;

    const frameData = {
        latitude: gpsData.latitude,
        longitude: gpsData.longitude,
        accuracy: gpsData.accuracy,
        alpha: deviceOrientation.alpha,
        beta: deviceOrientation.beta,
        gamma: deviceOrientation.gamma,
        use_api: useApi
    };

    logDebug(`Sending sensor data via WebSocket: lat=${frameData.latitude.toFixed(4)}, lon=${frameData.longitude.toFixed(4)}, alpha=${frameData.alpha.toFixed(0)}`);
    try {
        websocket.send(JSON.stringify(frameData));
    } catch (error) {
        console.error('Failed to send data via WebSocket:', error);
        logDebug('Failed to send data via WebSocket');
    }
}

function updateSensorDisplay() {
    const gpsText = gpsData.latitude && gpsData.longitude
        ? `${gpsData.latitude.toFixed(4)}, ${gpsData.longitude.toFixed(4)} (±${gpsData.accuracy.toFixed(0)}m)`
        : '--';

    const orientationText = deviceOrientation.alpha || deviceOrientation.beta || deviceOrientation.gamma
        ? `α:${deviceOrientation.alpha.toFixed(0)}° β:${deviceOrientation.beta.toFixed(0)}° γ:${deviceOrientation.gamma.toFixed(0)}°`
        : '--';

    const connectionText = websocket && websocket.readyState === WebSocket.OPEN
        ? 'WebSocket'
        : 'HTTP';

    const apiText = useApi ? 'API + Local' : 'Local Only';

    sensorDataDiv.innerHTML = `
        <div>GPS: ${gpsText}</div>
        <div>Orientation: ${orientationText}</div>
        <div>Connection: ${connectionText}</div>
        <div>Data Source: ${apiText}</div>
    `;
}

function toggleApi() {
    console.log('toggleApi called, current useApi:', useApi);
    useApi = !useApi;
    console.log('new useApi:', useApi);

    const apiToggle = document.getElementById('apiToggle');
    const apiIcon = document.getElementById('apiIcon');
    const apiText = document.getElementById('apiText');

    if (useApi) {
        apiToggle.classList.add('active');
        apiIcon.textContent = '🌐';
        apiText.textContent = 'API + Local';
    } else {
        apiToggle.classList.remove('active');
        apiIcon.textContent = '🏛️';
        apiText.textContent = 'Local Only';
    }

    updateSensorDisplay();
    console.log('Toggle completed');
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
            console.log('Requesting GPS location...');
            watchId = navigator.geolocation.watchPosition(
                (pos) => {
                    console.log(`GPS update: ${pos.coords.latitude}, ${pos.coords.longitude}`);
                    const accuracy = pos.coords.accuracy;
                    if (accuracy > GPS_MAX_ACCURACY_METERS) {
                        logDebug(`GPS accuracy too low (${accuracy.toFixed(0)}m), waiting for better fix...`);
                        return;
                    }

                    gpsSamples.push({
                        latitude: pos.coords.latitude,
                        longitude: pos.coords.longitude,
                        accuracy
                    });

                    if (gpsSamples.length > GPS_SMOOTHING_WINDOW) {
                        gpsSamples.shift();
                    }

                    // Weighted average by inverse accuracy
                    let weightSum = 0;
                    let latSum = 0;
                    let lonSum = 0;
                    for (const sample of gpsSamples) {
                        const weight = 1 / Math.max(sample.accuracy, 1);
                        weightSum += weight;
                        latSum += sample.latitude * weight;
                        lonSum += sample.longitude * weight;
                    }

                    gpsData.latitude = latSum / weightSum;
                    gpsData.longitude = lonSum / weightSum;
                    gpsData.accuracy = accuracy;
                    statusDiv.textContent = `GPS: ${gpsData.latitude.toFixed(4)}, ${gpsData.longitude.toFixed(4)}`;
                    updateSensorDisplay();
                },
                (error) => {
                    console.error('GPS error:', error);
                    statusDiv.textContent = 'GPS unavailable';
                    updateSensorDisplay();
                },
                { enableHighAccuracy: true, maximumAge: 0, timeout: 10000 }
            );
        }

        statusDiv.textContent = 'Initializing deviceOrientation...';

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
                                deviceOrientation.alpha = event.alpha || 0;
                                deviceOrientation.beta = event.beta || 0;
                                deviceOrientation.gamma = event.gamma || 0;
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
                deviceOrientation.alpha = event.alpha || 0;
                deviceOrientation.beta = event.beta || 0;
                deviceOrientation.gamma = event.gamma || 0;
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
        // Get server URL - use the same host as the current page
        let serverUrl = window.SERVER_URL || window.location.hostname;
        
        // For localhost, use localhost explicitly (not 127.0.0.1)
        if (serverUrl === '127.0.0.1' || serverUrl === '') {
            serverUrl = 'localhost';
        }
        
        console.log(`setupWebSocket: serverUrl="${serverUrl}", location.hostname="${window.location.hostname}"`);

        // Use WebSocket (ws for HTTP, wss for HTTPS)
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${serverUrl}:8000/ar-stream`;
        console.log(`Attempting WebSocket connection to: ${wsUrl}`);
        logDebug(`Attempting WebSocket connection to: ${wsUrl}`);
        websocket = new WebSocket(wsUrl);

        websocket.onopen = () => {
            console.log('WebSocket connected!');
            logDebug('WebSocket connected');
            statusDiv.textContent = 'AR mode active - scanning for landmarks...';
            updateSensorDisplay();
            sendSensorData();
            if (sendIntervalId) {
                clearInterval(sendIntervalId);
            }
            sendIntervalId = setInterval(() => {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    sendSensorData();
                }
            }, 500);
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
            logDebug('WebSocket error - falling back to HTTP');
            statusDiv.textContent = 'WebSocket error - falling back to HTTP mode';
            updateSensorDisplay();
            websocket = null;
            if (sendIntervalId) {
                clearInterval(sendIntervalId);
                sendIntervalId = null;
            }
            setTimeout(() => processARFrame(), 1000);
        };

        websocket.onclose = (event) => {
            statusDiv.textContent = 'Connection lost - attempting to reconnect...';
            updateSensorDisplay();
            websocket = null;
            if (sendIntervalId) {
                clearInterval(sendIntervalId);
                sendIntervalId = null;
            }
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

    console.log(`processARFrame: websocket=${websocket ? 'exists' : 'null'}, readyState=${websocket ? websocket.readyState : 'N/A'}, OPEN=${WebSocket.OPEN}`);

    if (!websocket || websocket.readyState !== WebSocket.OPEN) {
        console.log('Using HTTP fallback');
        logDebug('Using HTTP fallback');
        overlay.toBlob(async (imageBlob) => {
            const formData = new FormData();
            formData.append('frame', imageBlob, 'frame.jpg');
            formData.append('latitude', gpsData.latitude.toString());
            formData.append('longitude', gpsData.longitude.toString());
            formData.append('accuracy', gpsData.accuracy.toString());
            formData.append('alpha', deviceOrientation.alpha.toString());
            formData.append('beta', deviceOrientation.beta.toString());
            formData.append('gamma', deviceOrientation.gamma.toString());
            formData.append('use_api', useApi.toString());

            // Get server URL for HTTP fallback
            let serverUrl = window.SERVER_URL || window.location.hostname;
            if (serverUrl === '127.0.0.1' || serverUrl === '') {
                serverUrl = 'localhost';
            }

            try {
                const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
                const response = await fetch(`${protocol}//${serverUrl}:8000/analyze-ar-frame`, {
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
        console.log('Using WebSocket');
        logDebug('Using WebSocket');
        if (!isInitialized) {
            logDebug('AR not fully initialized yet, sending sensor data anyway');
        }
        sendSensorData();
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

    // Add API toggle event listener
    document.getElementById('apiToggle').addEventListener('click', toggleApi);
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