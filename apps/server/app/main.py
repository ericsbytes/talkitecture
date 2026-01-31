from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from vision import analyze_video

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze-video")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    with open(f"temp_{file.filename}", "wb") as buffer:
        buffer.write(await file.read())

    # Analyze the video
    result = analyze_video(f"temp_{file.filename}")

    # Clean up temp file
    import os
    os.remove(f"temp_{file.filename}")

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
