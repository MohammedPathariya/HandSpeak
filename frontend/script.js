const videoElement = document.getElementById("video");
const predictionText = document.getElementById("prediction");
const BACKEND_URL = "https://pathariyamohammed--handspeak-backend.hf.space/predict";

// === Setup MediaPipe Hands ===
const hands = new Hands({
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7,
});

hands.onResults(onResults);

// === Start Webcam Feed ===
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({ image: videoElement });
  },
  width: 640,
  height: 480,
});
camera.start();

// === Handle Detection Results ===
async function onResults(results) {
  if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
    predictionText.textContent = "No hand detected";
    return;
  }

  const handLandmarks = results.multiHandLandmarks[0];
  const flatLandmarks = handLandmarks
    .map((lm) => [lm.x, lm.y, lm.z])
    .flat();

  try {
    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ landmarks: flatLandmarks }),
    });

    const data = await response.json();
    predictionText.textContent = data.prediction || "Error";
  } catch (err) {
    predictionText.textContent = "Error contacting server";
    console.error("Prediction Error:", err);
  }
}

let isCameraRunning = true;

const toggleBtn = document.getElementById("toggle-camera-btn");

toggleBtn.addEventListener("click", () => {
  if (isCameraRunning) {
    camera.stop();
    predictionText.textContent = "Camera Off";
    toggleBtn.textContent = "Start Camera";
    isCameraRunning = false;
  } else {
    camera.start().then(() => {
      console.log("✅ Camera restarted");
      predictionText.textContent = "Detecting...";
      toggleBtn.textContent = "Stop Camera";
      isCameraRunning = true;
    }).catch((err) => {
      console.error("❌ Failed to restart camera:", err);
    });
  }
});
