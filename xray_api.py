# xray_api.py - SIMPLIFIED WORKING VERSION
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import uvicorn
from datetime import datetime
import os
import base64

app = FastAPI(title="X-Ray Security Detection API", version="1.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model - try ONNX first, then PT
MODEL_PT = "runs/detect/train/weights/best.pt"
MODEL_ONNX = "runs/detect/train/weights/best.onnx"

model = None
if os.path.exists(MODEL_ONNX):
    try:
        model = YOLO(MODEL_ONNX)
        print(f"✅ X-Ray model loaded (ONNX): {MODEL_ONNX}")
    except Exception as e:
        print(f"ONNX load failed: {e}")

if model is None and os.path.exists(MODEL_PT):
    try:
        model = YOLO(MODEL_PT)
        print(f"✅ X-Ray model loaded (PyTorch): {MODEL_PT}")
    except Exception as e:
        print(f"PyTorch load failed: {e}")

if model is None:
    print("❌ No X-Ray model found!")

# Class names from STCray
CLASS_NAMES = {
    0: "Explosive", 1: "Gun", 2: "3D Gun", 3: "Knife", 4: "Cutter",
    5: "Blade", 6: "Shaving Razor", 7: "Lighter", 8: "Injection",
    9: "Battery", 10: "Nail Cutter", 11: "Other Sharp Item", 12: "Powerbank",
    13: "Scissors", 14: "Hammer", 15: "Pliers", 16: "Wrench",
    17: "Screwdriver", 18: "Handcuffs", 19: "Bullet", 20: "Multilabel Threat",
    21: "Non Threat", 22: "3D printed gun", 23: "Syringe"
}

HIGH_RISK = {0, 1, 2, 3, 19, 22}  # Explosive, Gun, 3D Gun, Knife, Bullet, 3D printed gun

@app.get("/health")
async def health_check():
    return {
        "status": "operational",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/classes")
async def get_classes():
    if model and hasattr(model, 'names') and model.names:
        return {"classes": model.names}
    return {"classes": CLASS_NAMES}

@app.post("/detect")
async def detect_threats(
    file: UploadFile = File(...),
    confidence: float = 0.5,
    return_image: bool = False
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        print(f"Processing X-Ray image: {file.filename}, shape: {image.shape}")
        
        # Run detection
        results = model(image, conf=confidence, verbose=False)
        
        # Parse results
        threats = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    
                    if model and hasattr(model, 'names') and model.names:
                        class_name = model.names.get(class_id, CLASS_NAMES.get(class_id, "Unknown"))
                    else:
                        class_name = CLASS_NAMES.get(class_id, "Unknown")
                    
                    threats.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(conf_score, 4),
                        "is_high_risk": class_id in HIGH_RISK
                    })
        
        threats.sort(key=lambda x: x["confidence"], reverse=True)
        threat_detected = any(t["is_high_risk"] for t in threats)
        
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "threat_analysis": {
                "threat_detected": threat_detected,
                "total_detections": len(threats),
                "high_risk_count": len([t for t in threats if t["is_high_risk"]]),
                "threats": threats
            },
            "security_assessment": {
                "status": "ALERT" if threat_detected else "CLEAR",
                "risk_level": "HIGH" if threat_detected else "LOW",
                "message": f"{len(threats)} threat(s) detected" if threats else "No threats detected",
                "action_required": threat_detected
            }
        }
        
        # Add annotated image if requested
        if return_image and results and len(results) > 0:
            annotated = results[0].plot(line_width=2)
            _, buffer = cv2.imencode('.jpg', annotated)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            response["annotated_image"] = f"data:image/jpeg;base64,{img_base64}"
        
        print(f"X-Ray detection complete: {len(threats)} threats found")
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"Error in X-Ray detection: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("=" * 50)
    print("📦 X-RAY API")
    print("=" * 50)
    print(f"Model loaded: {model is not None}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")