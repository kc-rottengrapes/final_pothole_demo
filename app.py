from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image

app = FastAPI(
    title="Pothole Detection API",
    description="Upload an image to detect potholes",
    version="1.0"
)

# Load YOLO model once
model = YOLO("best.pt")

@app.post("/detect")
async def detect_potholes(
    file: UploadFile = File(..., description="Upload an image (jpg/png)"),
    conf: float = Form(0.3, description="Confidence threshold (0.1 - 0.9)"),
    return_image: bool = Form(False, description="Return annotated image instead of JSON")
):
    # Read file into numpy
    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)))

    # Run YOLO prediction
    results = model(image, conf=conf, save=False, verbose=False)

    # Extract detections
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": r.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()
            })

    if return_image:
        # Annotated image
        annotated = results[0].plot()
        _, img_encoded = cv2.imencode(".jpg", annotated)
        return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

    return JSONResponse({"detections": detections})
