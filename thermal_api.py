# thermal_api.py - COMPLETE WORKING VERSION
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

app = FastAPI(title="Thermal Security Detection API", version="1.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = "runs/detect/thermal_weapons_model/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    model = None
else:
    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ Thermal model loaded from {MODEL_PATH}")
        print(f"   Classes: {model.names}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

# Class names (fallback if model.names is empty)
CLASS_NAMES = {
    0: "Gun", 1: "Knife", 2: "Scissor", 3: "Hammer",
    4: "Screwdriver", 5: "Wrench", 6: "Lighter",
    7: "Handsaw", 8: "Lock"
}

HIGH_RISK = {0, 1, 2}  # Gun, Knife, Scissor

@app.get("/health")
async def health_check():
    return {
        "status": "operational",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/classes")
async def get_classes():
    if model and hasattr(model, 'names') and model.names:
        return {"classes": model.names}
    return {"classes": CLASS_NAMES}

@app.post("/detect")
async def detect_weapons(
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
        
        print(f"Processing image: {file.filename}, shape: {image.shape}")
        
        # Run detection
        results = model(image, conf=confidence, verbose=False)
        
        # Parse results
        weapons = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    
                    # Get class name
                    if model and hasattr(model, 'names') and model.names:
                        class_name = model.names.get(class_id, CLASS_NAMES.get(class_id, "Unknown"))
                    else:
                        class_name = CLASS_NAMES.get(class_id, "Unknown")
                    
                    weapons.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(conf_score, 4),
                        "is_high_risk": class_id in HIGH_RISK
                    })
        
        weapons.sort(key=lambda x: x["confidence"], reverse=True)
        weapon_detected = len(weapons) > 0
        high_risk_detected = any(w["is_high_risk"] for w in weapons)
        
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "weapon_analysis": {
                "weapon_detected": high_risk_detected,
                "total_detections": len(weapons),
                "high_risk_count": len([w for w in weapons if w["is_high_risk"]]),
                "weapons": weapons
            },
            "security_assessment": {
                "status": "ALERT" if high_risk_detected else "CLEAR",
                "risk_level": "HIGH" if high_risk_detected else "LOW",
                "message": f"{len(weapons)} weapon(s) detected" if weapons else "No weapons detected",
                "action_required": high_risk_detected
            }
        }
        
        # Add annotated image if requested
        if return_image and results and len(results) > 0:
            annotated = results[0].plot(line_width=2)
            _, buffer = cv2.imencode('.jpg', annotated)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            response["annotated_image"] = f"data:image/jpeg;base64,{img_base64}"
        
        print(f"Detection complete: {len(weapons)} weapons found")
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"Error in thermal detection: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("=" * 50)
    print("🔥 THERMAL API")
    print("=" * 50)
    print(f"Model: {MODEL_PATH}")
    print(f"Exists: {os.path.exists(MODEL_PATH)}")
    print(f"Model loaded: {model is not None}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")