// AR Mode JavaScript for dedicated AR page
const camera = document.getElementById('camera');
const overlay = document.getElementById('overlay');
const landmarkInfo = document.getElementById('landmarkInfo');
const landmarkName = document.getElementById('landmarkName');
const landmarkScript = document.getElementById('landmarkScript');
const playButton = document.getElementById('playButton');
const audioStatus = document.getElementById('audioStatus');
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
let currentLandmarkName = null;
let currentLandmarkData = null; // Store full landmark data for audio_url
let audioElement = null;
let isPlayingAudio = false;
let currentAudioBlobUrl = null; // Store blob URL to prevent garbage collection

function logDebug(message) {
    if (!debugLogDiv) return;
    const entry = document.createElement('div');
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    debugLogDiv.appendChild(entry);
    debugLogDiv.scrollTop = debugLogDiv.scrollHeight;
}

async function playLandmarkNarration(landmarkName) {
    if (!landmarkName || isPlayingAudio) return;
    
    try {
        playButton.disabled = true;
        audioStatus.textContent = 'Loading audio...';

        // Stop any existing playback first
        if (audioElement && !audioElement.paused) {
            audioElement.pause();
            audioElement.currentTime = 0;
        }

        let audioUrl = null;
        
        // Check if landmark data has audio_url
        if (currentLandmarkData && currentLandmarkData.audio_url) {
            // Extract filename from audio_url (e.g., "/audio/sciences_library.mp3" -> "sciences_library.mp3")
            const filename = currentLandmarkData.audio_url.split('/').pop();
            
            // Use local audio file served from same origin (client directory)
            audioUrl = `/audio/${filename}`;
        } else {
            // Fallback to generating audio from TTS
            let serverUrl = window.SERVER_URL || window.location.hostname;
            if (serverUrl === '127.0.0.1' || serverUrl === '') {
                serverUrl = 'localhost';
            }

            const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
            const ttsUrl = `${protocol}//${serverUrl}:8000/landmark/voice?landmark_name=${encodeURIComponent(landmarkName)}`;

            try {
                const response = await fetch(ttsUrl);
                if (!response.ok) {
                    throw new Error(`TTS failed: ${response.status}`);
                }
                const audioBlob = await response.blob();
                audioUrl = URL.createObjectURL(audioBlob);
            } catch (ttsError) {
                throw ttsError;
            }
        }

        // Create or reuse audio element
        if (!audioElement) {
            audioElement = new Audio();
            audioElement.preload = 'auto';
            
            audioElement.addEventListener('ended', () => {
                isPlayingAudio = false;
                playButton.disabled = false;
                audioStatus.textContent = 'Finished';
                playButton.textContent = '🔊 Play Narration';
            });
            
            audioElement.addEventListener('error', (e) => {
                const errorCode = e.target.error ? e.target.error.code : 'unknown';
                const errorMsg = e.target.error ? e.target.error.message : 'unknown error';
                console.error('Audio playback error:', e, e.target.error);
                isPlayingAudio = false;
                playButton.disabled = false;
                audioStatus.textContent = `Error: ${errorMsg}`;
            });
        }

        audioElement.src = audioUrl;
        audioElement.volume = 1.0;
        audioElement.load(); // Explicitly start loading
        
        isPlayingAudio = true;
        audioStatus.textContent = 'Loading...';
        playButton.textContent = '⏸ Pause';
        
        // Wait for enough data before playing
        await new Promise((resolve, reject) => {
            const canplayHandler = () => {
                audioElement.removeEventListener('canplaythrough', canplayHandler);
                audioElement.removeEventListener('error', errorHandler);
                resolve();
            };
            
            const errorHandler = (e) => {
                audioElement.removeEventListener('canplaythrough', canplayHandler);
                audioElement.removeEventListener('error', errorHandler);
                reject(new Error('Failed to load audio'));
            };
            
            audioElement.addEventListener('canplaythrough', canplayHandler, { once: true });
            audioElement.addEventListener('error', errorHandler, { once: true });
        });
        
        audioStatus.textContent = 'Playing...';
        
        try {
            await audioElement.play();
        } catch (playError) {
            throw playError;
        }
    } catch (error) {
        console.error('Error playing narration:', error);
        isPlayingAudio = false;
        playButton.disabled = false;
        audioStatus.textContent = 'Error playing audio';
    }
}

function toggleAudioPlayback() {
    if (!currentLandmarkName) {
        audioStatus.textContent = 'No landmark selected';
        return;
    }

    if (isPlayingAudio && audioElement && !audioElement.paused) {
        // Pause the audio
        audioElement.pause();
        isPlayingAudio = false;
        playButton.textContent = '▶️ Resume';
        audioStatus.textContent = 'Paused';
        playButton.disabled = false;
    } else if (audioElement && audioElement.paused && audioElement.currentTime > 0 && audioElement.currentTime < audioElement.duration) {
        // Resume paused audio
        isPlayingAudio = true;
        audioElement.play().then(() => {
            playButton.textContent = '⏸ Pause';
            audioStatus.textContent = 'Playing...';
        }).catch(err => {
            isPlayingAudio = false;
            playButton.disabled = false;
        });
    } else {
        // Start playing from beginning
        playLandmarkNarration(currentLandmarkName);
    }
}

function stopAudio() {
    if (audioElement) {
        audioElement.pause();
        audioElement.currentTime = 0;
        isPlayingAudio = false;
        playButton.textContent = '🔊 Play Narration';
        audioStatus.textContent = 'Stopped';
        playButton.disabled = false;
    }
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
        use_api: useApi,
        screen_width: overlay.width || window.innerWidth,
        screen_height: overlay.height || window.innerHeight
    };

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

        // Wait a moment for video dimensions to be available
        await new Promise(resolve => {
            const checkDimensions = () => {
                if (camera.videoWidth > 0 && camera.videoHeight > 0) {
                    console.log(`Camera dimensions: ${camera.videoWidth}x${camera.videoHeight}`);
                    overlay.width = camera.videoWidth;
                    overlay.height = camera.videoHeight;
                    console.log(`Overlay canvas dimensions: ${overlay.width}x${overlay.height}`);
                    
                    // Initialize 2D context immediately to reserve it
                    const ctx = overlay.getContext('2d');
                    console.log('Overlay 2D context initialized:', !!ctx);
                    
                    resolve();
                } else {
                    setTimeout(checkDimensions, 100);
                }
            };
            checkDimensions();
        });

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

let glRenderer = null;
let faceCanvas = null;
let faceCtx = null;

function initFaceCanvas() {
    if (!faceCanvas) {
        console.log('Creating face canvas...');
        faceCanvas = document.createElement('canvas');
        faceCanvas.width = 200;
        faceCanvas.height = 200;
        faceCtx = faceCanvas.getContext('2d');
        console.log('Face canvas created:', faceCanvas.width, 'x', faceCanvas.height);
        
        // Draw face immediately
        drawFace(faceCanvas, faceCtx);
        console.log('Face drawn on canvas');
    } else {
        console.log('Face canvas already initialized');
    }
}

function initWebGL() {
    try {
        const gl = overlay.getContext('webgl2') || overlay.getContext('webgl');
        if (!gl) throw new Error('WebGL not supported');
        
        glRenderer = { gl };
        return glRenderer;
    } catch (error) {
        console.error('WebGL initialization failed:', error);
        return null;
    }
}

function drawFace(canvas, ctx) {
    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'rgba(255, 200, 100, 0.8)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw face outline
    ctx.strokeStyle = '#8B6F47';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(canvas.width/2, canvas.height/2, canvas.width/2 - 5, 0, Math.PI * 2);
    ctx.stroke();
    
    // Draw eyes
    ctx.fillStyle = '#000';
    ctx.beginPath();
    ctx.arc(canvas.width/3, canvas.height/3, 15, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.beginPath();
    ctx.arc(canvas.width * 2/3, canvas.height/3, 15, 0, Math.PI * 2);
    ctx.fill();
    
    // Draw mouth
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(canvas.width/2, canvas.height * 2/3, 40, 0, Math.PI);
    ctx.stroke();
}

function updateAROverlay(result) {
    if (result.error) {
        landmarkInfo.style.display = 'none';
        return;
    }

    // Update landmark info
    const landmarkNameText = result.landmark.name;
    landmarkName.textContent = landmarkNameText;
    // Check if landmark changed
    const landmarkChanged = currentLandmarkName !== landmarkNameText;
    
    currentLandmarkName = landmarkNameText;
    currentLandmarkData = result.landmark; // Store full landmark data
    
    console.log('updateAROverlay - landmark:', landmarkNameText, 'face_region:', result.face_region);
    
    // Reset audio status ONLY when landmark changes
    if (landmarkChanged) {
        if (audioElement && !audioElement.paused) {
            logDebug(`Landmark changed - stopping audio`);
            audioElement.pause();
        }
        isPlayingAudio = false;
        playButton.disabled = false;
        playButton.textContent = '🔊 Play Narration';
        audioStatus.textContent = '';
    }

    // Show landmark info
    landmarkInfo.style.display = 'block';

    // Initialize face canvas immediately
    initFaceCanvas();
    
    // Ensure canvas has proper dimensions FIRST
    if (!overlay.width || !overlay.height) {
        if (camera && camera.videoWidth > 0 && camera.videoHeight > 0) {
            overlay.width = camera.videoWidth;
            overlay.height = camera.videoHeight;
            console.log('Set overlay dimensions:', overlay.width, 'x', overlay.height);
        } else {
            // Fallback to window dimensions
            overlay.width = window.innerWidth;
            overlay.height = window.innerHeight;
            console.log('Set overlay to window dimensions:', overlay.width, 'x', overlay.height);
        }
    }
    
    // Get 2D context for drawing
    const ctx = overlay.getContext('2d');
    console.log('Overlay canvas:', overlay.width, 'x', overlay.height);
    console.log('Context obtained:', !!ctx);
    console.log('FaceCanvas status:', faceCanvas ? `${faceCanvas.width}x${faceCanvas.height}` : 'null');
    
    // Clear canvas
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Draw debug indicator to verify canvas works
    ctx.fillStyle = '#FF0000';
    ctx.fillRect(10, 10, 20, 20);

    // Render face with perspective transform
    if (result.face_region) {
        const face = result.face_region;
        console.log('Drawing face with region:', JSON.stringify(face));
        
        try {
            // Handle both array format (quad points) and object format (bounding box)
            if (Array.isArray(face) && face.length >= 4) {
                console.log('Applying quad perspective transform');
                console.log('Face region as quad:', face);
                drawFaceWithPerspective(faceCanvas, face, ctx);
            } else if (face.x !== undefined) {
                console.log('Drawing face at bounding box position with wall projection:', face);
                if (!faceCanvas) {
                    console.error('faceCanvas is null!');
                    return;
                }
                
                // Create 3D wall projection with perspective
                const x = face.x;
                const y = face.y;
                const w = face.width;
                const h = face.height;
                
                // Simulate a vertical wall with 3D perspective
                // Narrower at top AND sides (creates depth illusion)
                const perspectiveX = w * 0.12;  // Side perspective (12%)
                const perspectiveY = h * 0.08;  // Top perspective (8%)
                
                // Create quad that simulates a wall viewed from angle
                const quad = [
                    [x + perspectiveX, y + perspectiveY],              // top-left (inset both ways)
                    [x + w - perspectiveX, y + perspectiveY],          // top-right (inset both ways)
                    [x + w, y + h],                                    // bottom-right (full width at bottom)
                    [x, y + h]                                         // bottom-left (full width at bottom)
                ];
                
                console.log('Wall projection quad:', quad);
                drawFaceWithPerspective(faceCanvas, quad, ctx);
                console.log('Face projected onto wall surface');
            } else {
                console.log('Face region format not recognized, raw data:', face);
                // Fallback: draw at center of screen
                const centerX = overlay.width / 2 - 100;
                const centerY = overlay.height / 2 - 100;
                ctx.globalAlpha = 0.8;
                ctx.drawImage(faceCanvas, centerX, centerY, 200, 200);
                ctx.globalAlpha = 1.0;
                console.log('Face drawn at center (fallback)');
            }
        } catch (error) {
            console.error('Error drawing face:', error);
        }
    } else {
        console.log('No face_region in result');
        // Draw at center as fallback
        try {
            const centerX = overlay.width / 2 - 100;
            const centerY = overlay.height / 2 - 100;
            ctx.globalAlpha = 0.6;
            ctx.drawImage(faceCanvas, centerX, centerY, 200, 200);
            ctx.globalAlpha = 1.0;
            console.log('Face drawn at center (no region)');
        } catch (error) {
            console.error('Error drawing face fallback:', error);
        }
    }
}

function drawFaceWithPerspective(srcCanvas, quad, destCanvas) {
    /**
     * Draw source face canvas onto destination canvas with perspective
     * quad format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (TL, TR, BR, BL)
     */
    const ctx = destCanvas.getContext('2d');
    const w = srcCanvas.width;
    const h = srcCanvas.height;
    
    // Source corners in original face
    const srcCorners = [[0, 0], [w, 0], [w, h], [0, h]];
    
    // Use canvas 2D transform with quadrilateral mapping
    // We'll use simple triangulation - split quad into 2 triangles
    
    ctx.globalAlpha = 0.7;
    
    // Draw first triangle (TL, TR, BR)
    drawTriangle(ctx, srcCanvas,
        srcCorners[0], srcCorners[1], srcCorners[2],
        quad[0], quad[1], quad[2]
    );
    
    // Draw second triangle (TL, BR, BL)
    drawTriangle(ctx, srcCanvas,
        srcCorners[0], srcCorners[2], srcCorners[3],
        quad[0], quad[2], quad[3]
    );
    
    ctx.globalAlpha = 1.0;
}

function drawTriangle(ctx, srcCanvas, srcP1, srcP2, srcP3, dstP1, dstP2, dstP3) {
    /**
     * Draw a triangle from source canvas to destination with perspective transform
     */
    try {
        ctx.save();
        
        // Create clipping region for destination triangle
        ctx.beginPath();
        ctx.moveTo(dstP1[0], dstP1[1]);
        ctx.lineTo(dstP2[0], dstP2[1]);
        ctx.lineTo(dstP3[0], dstP3[1]);
        ctx.closePath();
        ctx.clip();
        
        // Compute affine transformation from source to destination
        const mat = getAffineTransform(srcP1, srcP2, srcP3, dstP1, dstP2, dstP3);
        console.log('Affine matrix:', mat);
        
        if (!mat || (mat.a === 1 && mat.b === 0 && mat.c === 0 && mat.d === 1 && mat.e === 0 && mat.f === 0)) {
            console.warn('Affine matrix is identity, might indicate collinear points');
        }
        
        // Apply transformation
        ctx.transform(mat.a, mat.b, mat.c, mat.d, mat.e, mat.f);
        
        // Draw the source canvas
        ctx.drawImage(srcCanvas, 0, 0);
        console.log('Triangle drawn successfully');
        
        ctx.restore();
    } catch (error) {
        console.error('Error in drawTriangle:', error);
        ctx.restore();
    }
}

function getAffineTransform(src1, src2, src3, dst1, dst2, dst3) {
    /**
     * Calculate affine transformation matrix that maps src points to dst points
     */
    const x1 = src1[0], y1 = src1[1];
    const x2 = src2[0], y2 = src2[1];
    const x3 = src3[0], y3 = src3[1];
    
    const u1 = dst1[0], v1 = dst1[1];
    const u2 = dst2[0], v2 = dst2[1];
    const u3 = dst3[0], v3 = dst3[1];
    
    const denom = x1 * (y2 - y3) - x2 * (y1 - y3) + x3 * (y1 - y2);
    
    if (Math.abs(denom) < 0.0001) return { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };
    
    const a = ((u1 * (y2 - y3) - u2 * (y1 - y3) + u3 * (y1 - y2)) / denom);
    const b = ((v1 * (y2 - y3) - v2 * (y1 - y3) + v3 * (y1 - y2)) / denom);
    const c = ((u1 * (x3 - x2) - u2 * (x3 - x1) + u3 * (x2 - x1)) / denom);
    const d = ((v1 * (x3 - x2) - v2 * (x3 - x1) + v3 * (x2 - x1)) / denom);
    const e = (u1 * (x2 * y3 - x3 * y2) - u2 * (x1 * y3 - x3 * y1) + u3 * (x1 * y2 - x2 * y1)) / denom;
    const f = (v1 * (x2 * y3 - x3 * y2) - v2 * (x1 * y3 - x3 * y1) + v3 * (x1 * y2 - x2 * y1)) / denom;
    
    return { a, b, c, d, e, f };
}


// Initialize AR when page loads
window.addEventListener('load', () => {
    updateSensorDisplay(); // Initialize display
    initAR();

    // Add API toggle event listener
    document.getElementById('apiToggle').addEventListener('click', toggleApi);
    
    // Add audio control button listeners
    playButton.addEventListener('click', toggleAudioPlayback);
    document.getElementById('stopButton').addEventListener('click', stopAudio);
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