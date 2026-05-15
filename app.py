# main.py - Fusion UI for Thermal + X-Ray Detection + Queue Simulation
import streamlit as st
import requests
import json
from PIL import Image
import io
import base64
import time
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# API endpoints
THERMAL_API_URL = "http://127.0.0.1:8001/detect"
XRAY_API_URL = "http://127.0.0.1:8000/detect"
THERMAL_HEALTH_URL = "http://127.0.0.1:8001/health"
XRAY_HEALTH_URL = "http://127.0.0.1:8000/health"

# Load person detector (cached to avoid reloading)
@st.cache_resource
def load_person_detector():
    """Load YOLO model for person detection"""
    return YOLO("yolov8n.pt")

# Load weapon model (cached)
@st.cache_resource
def load_weapon_model():
    """Load your trained thermal weapon detection model"""
    return YOLO("runs/detect/thermal_weapons_model/weights/best.pt")

st.set_page_config(
    page_title="Multi-Modal Security Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert { font-size: 1.2rem; font-weight: bold; }
    .alert-red { background-color: #ff4b4b; color: white; padding: 1rem; border-radius: 10px; text-align: center; }
    .alert-green { background-color: #00cc66; color: white; padding: 1rem; border-radius: 10px; text-align: center; }
    .alert-yellow { background-color: #ffcc00; color: black; padding: 1rem; border-radius: 10px; text-align: center; }
    .person-box { border: 1px solid #ddd; border-radius: 10px; padding: 10px; margin: 5px; }
    .threat-box { border: 2px solid #ff4b4b; background-color: #ffeeee; }
    .bag-box { border: 1px solid #ddd; border-radius: 10px; padding: 10px; margin: 5px; background-color: #f9f9f9; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Multi-Modal Security Detection System")
st.markdown("*Thermal Imaging (Humans) + X-Ray (Baggage) + Queue Simulation*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    st.markdown("---")
    st.header("📊 System Status")
    
    # Check Thermal API
    thermal_online = False
    try:
        r = requests.get(THERMAL_HEALTH_URL, timeout=3)
        if r.status_code == 200:
            thermal_online = True
            st.success("✅ Thermal API (Port 8001) - Online")
        else:
            st.error("❌ Thermal API - Error")
    except requests.exceptions.ConnectionError:
        st.error("❌ Thermal API (Port 8001) - Not running")
    except Exception as e:
        st.error(f"❌ Thermal API - {str(e)[:50]}")
    
    # Check X-Ray API
    xray_online = False
    try:
        r = requests.get(XRAY_HEALTH_URL, timeout=3)
        if r.status_code == 200:
            xray_online = True
            st.success("✅ X-Ray API (Port 8000) - Online")
        else:
            st.error("❌ X-Ray API - Error")
    except requests.exceptions.ConnectionError:
        st.error("❌ X-Ray API (Port 8000) - Not running")
    except Exception as e:
        st.error(f"❌ X-Ray API - {str(e)[:50]}")
    
    st.markdown("---")
    st.caption("Start APIs with:\n`python thermal_api.py`\n`python xray_api.py`")

# ============ TABBED INTERFACE ============
tab1, tab2, tab3 = st.tabs(["🔥 Single Person Scan", "📦 X-Ray Baggage Scan", "👥 Queue Simulation (Multi-Person)"])

# Session state
if 'thermal_result' not in st.session_state:
    st.session_state.thermal_result = None
if 'xray_result' not in st.session_state:
    st.session_state.xray_result = None
if 'queue_results' not in st.session_state:
    st.session_state.queue_results = None

# ============ TAB 1: THERMAL (Single Person) ============
with tab1:
    st.header("🔥 Thermal Camera (Single Person)")
    st.caption("Upload thermal image of a single person to detect concealed weapons")
    
    thermal_image = st.file_uploader(
        "Upload Thermal Image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        key="thermal"
    )
    
    if thermal_image is not None:
        st.image(thermal_image, caption="Thermal Image", use_container_width=True)
        
        if st.button("🔍 Scan Person", key="scan_thermal", type="primary"):
            if not thermal_online:
                st.error("❌ Thermal API is not running. Please start it with: python thermal_api.py")
            else:
                with st.spinner("Analyzing thermal image..."):
                    files = {"file": thermal_image.getvalue()}
                    data = {"confidence": confidence_threshold, "return_image": True}
                    
                    try:
                        response = requests.post(THERMAL_API_URL, files=files, data=data, timeout=30)
                        
                        if response.status_code == 200:
                            try:
                                result = response.json()
                                st.session_state.thermal_result = result
                            except requests.exceptions.JSONDecodeError:
                                st.error(f"Thermal API returned invalid JSON. Response: {response.text[:200]}")
                        else:
                            st.error(f"Thermal API returned status {response.status_code}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Display thermal results
        if st.session_state.thermal_result:
            result = st.session_state.thermal_result
            if result.get("success"):
                weapons = result.get("weapon_analysis", {}).get("weapons", [])
                status = result.get("security_assessment", {}).get("status", "UNKNOWN")
                
                if status == "ALERT":
                    st.markdown('<div class="alert-red">🚨 ALERT: Weapons detected on person!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-green">✅ CLEAR: No weapons detected</div>', unsafe_allow_html=True)
                
                if weapons:
                    st.subheader("🔫 Detected Weapons:")
                    for w in weapons:
                        st.write(f"- **{w['class_name']}** (confidence: {w['confidence']:.2%})")
                
                if "annotated_image" in result:
                    img_data = base64.b64decode(result["annotated_image"].split(",")[1])
                    st.image(Image.open(io.BytesIO(img_data)), caption="Detection Result", use_container_width=True)

# ============ TAB 2: X-RAY (Baggage) ============
with tab2:
    st.header("📦 X-Ray Scanner (Baggage)")
    st.caption("Upload X-ray image of baggage to detect threats")
    
    xray_image = st.file_uploader(
        "Upload X-Ray Image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        key="xray"
    )
    
    if xray_image is not None:
        st.image(xray_image, caption="X-Ray Image", use_container_width=True)
        
        if st.button("🔍 Scan Baggage", key="scan_xray", type="primary"):
            if not xray_online:
                st.error("❌ X-Ray API is not running")
            else:
                with st.spinner("Analyzing X-ray image..."):
                    files = {"file": xray_image.getvalue()}
                    data = {"confidence": confidence_threshold, "return_image": True}
                    
                    try:
                        response = requests.post(XRAY_API_URL, files=files, data=data, timeout=30)
                        
                        if response.status_code == 200:
                            try:
                                result = response.json()
                                st.session_state.xray_result = result
                            except requests.exceptions.JSONDecodeError:
                                st.error(f"X-Ray API returned invalid JSON")
                        else:
                            st.error(f"X-Ray API returned status {response.status_code}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Display X-Ray results
        if st.session_state.xray_result:
            result = st.session_state.xray_result
            if result.get("success"):
                threats = result.get("threat_analysis", {}).get("threats", [])
                status = result.get("security_assessment", {}).get("status", "UNKNOWN")
                
                if status == "ALERT":
                    st.markdown('<div class="alert-red">🚨 ALERT: Threats detected in baggage!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-green">✅ CLEAR: No threats detected</div>', unsafe_allow_html=True)
                
                if threats:
                    st.subheader("⚠️ Detected Threats:")
                    for t in threats:
                        st.write(f"- **{t['class_name']}** (confidence: {t['confidence']:.2%})")
                
                if "annotated_image" in result:
                    img_data = base64.b64decode(result["annotated_image"].split(",")[1])
                    st.image(Image.open(io.BytesIO(img_data)), caption="Detection Result", use_container_width=True)

# ============ TAB 3: QUEUE SIMULATION (Multi-Person + Multi-Bag) ============
with tab3:
    st.header("👥 Queue Simulation (Multi-Person + Multi-Bag)")
    st.caption("Upload BOTH a stitched thermal image (people) and a stitched X-ray image (bags). The system will detect each person and each bag, run weapon detection on both, and merge alerts by index.")
    
    st.info("💡 **How to create stitched images:** Use any image editor to place multiple single-person thermal images side-by-side horizontally. Create a matching X-ray stitched image with bags in the same left-to-right order. Upload both here.")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("👥 Stitched Thermal Image (People)")
        queue_image = st.file_uploader(
            "Upload Stitched Thermal Image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="queue_thermal"
        )
        if queue_image is not None:
            st.image(queue_image, caption="Stitched People Image", use_container_width=True)
    
    with col_right:
        st.subheader("🎒 Stitched X-Ray Image (Bags)")
        xray_queue_image = st.file_uploader(
            "Upload Stitched X-Ray Image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="queue_xray"
        )
        if xray_queue_image is not None:
            st.image(xray_queue_image, caption="Stitched Bags Image", use_container_width=True)
    
    if queue_image is not None or xray_queue_image is not None:
        if st.button("👥 Scan Full Queue (People + Bags)", key="scan_full_queue", type="primary"):
            if not thermal_online:
                st.error("❌ Thermal API is not running. Please start it with: python thermal_api.py")
            elif not xray_online:
                st.error("❌ X-Ray API is not running. Please start it with: python xray_api.py")
            else:
                with st.spinner("Processing queue - detecting people, bags, and weapons..."):
                    try:
                        # Load models
                        person_model = load_person_detector()
                        weapon_model = load_weapon_model()
                        
                        # ========== PROCESS THERMAL IMAGE (PEOPLE) ==========
                        thermal_results_data = []
                        thermal_alert_count = 0
                        num_people = 0
                        
                        if queue_image is not None:
                            # Convert uploaded file to OpenCV image
                            file_bytes = np.asarray(bytearray(queue_image.read()), dtype=np.uint8)
                            thermal_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                            
                            if thermal_img is not None:
                                h, w = thermal_img.shape[:2]
                                st.info(f"📐 Thermal image size: {w} x {h} pixels")
                                
                                # Detect all people
                                person_results = person_model(thermal_img, verbose=False)
                                
                                person_boxes = []
                                for result in person_results:
                                    if result.boxes is not None:
                                        for box in result.boxes:
                                            class_id = int(box.cls[0])
                                            if person_model.names[class_id] == "person":
                                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                                conf = float(box.conf[0])
                                                person_boxes.append({
                                                    "bbox": [x1, y1, x2, y2],
                                                    "confidence": conf
                                                })
                                
                                # Sort left to right
                                person_boxes.sort(key=lambda p: p["bbox"][0])
                                num_people = len(person_boxes)
                                st.success(f"✅ Detected {num_people} people in thermal image")
                                
                                # Process each person
                                for idx, person in enumerate(person_boxes):
                                    person_num = idx + 1
                                    x1, y1, x2, y2 = person["bbox"]
                                    
                                    # Crop person
                                    person_img = thermal_img[y1:y2, x1:x2]
                                    
                                    if person_img.size == 0:
                                        continue
                                    
                                    # Resize for weapon model
                                    person_resized = cv2.resize(person_img, (640, 640))
                                    
                                    # Run weapon detection via API (to keep consistency)
                                    _, encoded = cv2.imencode('.jpg', person_resized)
                                    files = {"file": encoded.tobytes()}
                                    data = {"confidence": confidence_threshold}
                                    
                                    try:
                                        response = requests.post(THERMAL_API_URL, files=files, data=data, timeout=30)
                                        if response.status_code == 200:
                                            result = response.json()
                                            weapons = result.get("weapon_analysis", {}).get("weapons", [])
                                        else:
                                            weapons = []
                                    except:
                                        weapons = []
                                    
                                    has_weapon = len(weapons) > 0
                                    if has_weapon:
                                        thermal_alert_count += 1
                                    
                                    # Extract head (top 30%)
                                    head_h = int((y2 - y1) * 0.3)
                                    head_img = person_img[0:head_h, 0:person_img.shape[1]]
                                    
                                    thermal_results_data.append({
                                        "person_num": person_num,
                                        "has_weapon": has_weapon,
                                        "weapons": weapons,
                                        "head_image": head_img,
                                        "person_image": person_img
                                    })
                        
                        # ========== PROCESS X-RAY IMAGE (BAGS) ==========
                        xray_results_data = []
                        xray_alert_count = 0
                        num_bags = 0
                        
                        if xray_queue_image is not None:
                            # Convert uploaded file to OpenCV image
                            file_bytes = np.asarray(bytearray(xray_queue_image.read()), dtype=np.uint8)
                            xray_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                            
                            if xray_img is not None:
                                h, w = xray_img.shape[:2]
                                st.info(f"📐 X-Ray image size: {w} x {h} pixels")
                                
                                # Since we don't have a bag detector, assume fixed zones or use simple contour detection
                                # For now, we'll use the same number of people as reference
                                if num_people > 0:
                                    num_bags = num_people
                                    bag_width = w // num_bags
                                    
                                    for idx in range(num_bags):
                                        bag_num = idx + 1
                                        x1 = idx * bag_width
                                        x2 = (idx + 1) * bag_width
                                        
                                        # Crop bag region
                                        bag_img = xray_img[0:h, x1:x2]
                                        
                                        if bag_img.size == 0:
                                            continue
                                        
                                        # Encode and send to X-Ray API
                                        _, encoded = cv2.imencode('.jpg', bag_img)
                                        files = {"file": encoded.tobytes()}
                                        data = {"confidence": confidence_threshold}
                                        
                                        try:
                                            response = requests.post(XRAY_API_URL, files=files, data=data, timeout=30)
                                            if response.status_code == 200:
                                                result = response.json()
                                                threats = result.get("threat_analysis", {}).get("threats", [])
                                            else:
                                                threats = []
                                        except:
                                            threats = []
                                        
                                        has_threat = len(threats) > 0
                                        if has_threat:
                                            xray_alert_count += 1
                                        
                                        xray_results_data.append({
                                            "bag_num": bag_num,
                                            "has_threat": has_threat,
                                            "threats": threats
                                        })
                                    
                                    st.success(f"✅ Processed {num_bags} bag zones")
                        
                        # ========== MERGE ALERTS ==========
                        # Create unified results for all persons (up to max of people or bags)
                        max_items = max(len(thermal_results_data), len(xray_results_data))
                        unified_results = []
                        
                        for i in range(max_items):
                            person_num = i + 1
                            thermal_data = thermal_results_data[i] if i < len(thermal_results_data) else {"has_weapon": False, "weapons": [], "head_image": None, "person_image": None}
                            xray_data = xray_results_data[i] if i < len(xray_results_data) else {"has_threat": False, "threats": []}
                            
                            has_alert = thermal_data.get("has_weapon", False) or xray_data.get("has_threat", False)
                            
                            unified_results.append({
                                "person_num": person_num,
                                "has_weapon": thermal_data.get("has_weapon", False),
                                "weapons": thermal_data.get("weapons", []),
                                "has_threat": xray_data.get("has_threat", False),
                                "threats": xray_data.get("threats", []),
                                "has_alert": has_alert,
                                "head_image": thermal_data.get("head_image"),
                                "person_image": thermal_data.get("person_image")
                            })
                        
                        # ========== DISPLAY RESULTS ==========
                        st.markdown("---")
                        st.subheader("📋 Individual Scan Results")
                        
                        # Create columns for grid display
                        cols = st.columns(min(max_items, 4))
                        
                        for idx, result in enumerate(unified_results):
                            col_idx = idx % 4
                            with cols[col_idx]:
                                if result["has_alert"]:
                                    st.markdown(f'<div class="person-box threat-box">', unsafe_allow_html=True)
                                    st.markdown(f"### 👤 Person {result['person_num']} 🔴")
                                else:
                                    st.markdown(f'<div class="person-box">', unsafe_allow_html=True)
                                    st.markdown(f"### 👤 Person {result['person_num']} 🟢")
                                
                                if result["person_image"] is not None:
                                    st.image(result["person_image"], caption=f"Person {result['person_num']}", use_container_width=True)
                                
                                if result["has_weapon"]:
                                    weapons_str = ", ".join([w["class_name"] for w in result["weapons"]])
                                    st.markdown(f"**🔫 Person:** {weapons_str}")
                                
                                if result["has_threat"]:
                                    threats_str = ", ".join([t["class_name"] for t in result["threats"]])
                                    st.markdown(f"**🎒 Bag:** {threats_str}")
                                
                                if not result["has_weapon"] and not result["has_threat"]:
                                    st.markdown("✅ No threats")
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        # ========== FINAL FUSION ALERT SUMMARY ==========
                        st.markdown("---")
                        st.subheader("🚨 FUSION ALERT SUMMARY (People + Bags)")
                        
                        alert_persons = [r for r in unified_results if r["has_alert"]]
                        
                        if len(alert_persons) > 0:
                            st.markdown(f'<div class="alert-red">⚠️ ALERT: {len(alert_persons)} person(s) require manual inspection!</div>', unsafe_allow_html=True)
                            
                            # Detailed breakdown
                            for ap in alert_persons:
                                reasons = []
                                if ap["has_weapon"]:
                                    weapons_str = ", ".join([w["class_name"] for w in ap["weapons"]])
                                    reasons.append(f"Person has {weapons_str}")
                                if ap["has_threat"]:
                                    threats_str = ", ".join([t["class_name"] for t in ap["threats"]])
                                    reasons.append(f"Bag contains {threats_str}")
                                
                                # Position description with "last one to leave" for rightmost
                                if ap["person_num"] == 1:
                                    position = "leftmost (first to enter)"
                                elif ap["person_num"] == len(unified_results):
                                    position = "rightmost (last one to leave)"
                                else:
                                    position = f"{ap['person_num']}th from left"
                                
                                st.write(f"🔴 **Person {ap['person_num']}** ({position}): {' | '.join(reasons)}")
                            
                            # Display heads of all alerted persons
                            st.subheader("👤 Security Alert - Identify These Individuals")
                            head_cols = st.columns(min(len(alert_persons), 4))
                            head_idx = 0
                            for ap in alert_persons:
                                if ap["head_image"] is not None and head_idx < len(head_cols):
                                    with head_cols[head_idx]:
                                        st.image(ap["head_image"], caption=f"Person {ap['person_num']} (SUSPECT)", use_container_width=True)
                                    head_idx += 1
                        else:
                            st.markdown(f'<div class="alert-green">✅ CLEAR: No weapons on any person and no threats in any bags</div>', unsafe_allow_html=True)
                        
                        st.session_state.queue_results = unified_results
                        
                    except Exception as e:
                        st.error(f"Error processing queue: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

st.markdown("---")
st.caption("⚠️ Disclaimer: This system is for research purposes. Always follow official security protocols.")