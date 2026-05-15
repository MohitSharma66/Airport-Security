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
tab1, tab2, tab3, tab4 = st.tabs(["🔥 Single Person Scan", "📦 X-Ray Baggage Scan", "👥 Queue Simulation (Multi-Person)", "🎬 Video Queue Simulation"])
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

# ============ TAB 4: VIDEO QUEUE SIMULATION (Thermal Only) ============
with tab4:
    st.header("🎬 Video Queue Simulation (Thermal Only)")
    st.caption("Upload a stitched thermal video containing multiple people walking. The system extracts frames, detects people, and runs weapon detection on each person.")
    
    st.info("💡 **Video requirements:** Stitched thermal video with 2-4 people walking side-by-side.")
    
    # Video upload
    video_file = st.file_uploader(
        "Upload Stitched Thermal Video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        key="video_queue"
    )
    
    # Settings
    col_settings1, col_settings2, col_settings3, col_settings4, col_settings5 = st.columns(5)
    with col_settings1:
        num_frames_to_extract = st.slider(
            "Number of frames to extract", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="Extract this many equally spaced frames from the video"
        )
    with col_settings2:
        min_confirmation_frames = st.slider(
            "Minimum frames for confirmation",
            min_value=1,
            max_value=5,
            value=2,
            help="Number of frames weapon must appear to raise alert"
        )
    with col_settings3:
        confidence_threshold_video = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.5, 0.05,
            key="video_confidence"
        )
    with col_settings4:
        padding_percent = st.slider(
            "Person Crop Padding",
            min_value=0.0,
            max_value=0.8,
            value=0.3,
            step=0.05,
            help="Add padding around detected person (0.3 = 30% extra space). Helps capture guns in pockets."
        )
    with col_settings5:
        colormap_option = st.selectbox(
            "Color Mapping",
            options=["HOT", "INFERNO", "MAGMA", "PLASMA", "GRAY", "ORIGINAL"],
            index=0,
            help="Convert purple thermal videos to match training data appearance"
        )
    
    # Colormap mapping
    colormap_dict = {
        "HOT": cv2.COLORMAP_HOT,
        "INFERNO": cv2.COLORMAP_INFERNO,
        "MAGMA": cv2.COLORMAP_MAGMA,
        "PLASMA": cv2.COLORMAP_PLASMA,
        "GRAY": None,
        "ORIGINAL": "original"
    }
    
    def preprocess_thermal_frame(frame, colormap_choice):
        """Convert purple-ish thermal frames to match training data appearance."""
        if colormap_choice == "ORIGINAL":
            return frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if colormap_choice == "GRAY":
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        colormap_id = colormap_dict.get(colormap_choice)
        if colormap_id is not None:
            thermal_like = cv2.applyColorMap(gray, colormap_id)
            thermal_like = cv2.convertScaleAbs(thermal_like, alpha=1.1, beta=5)
            return thermal_like
        
        return frame
    
    def smart_crop_person(frame, bbox, padding):
        """
        Smart crop person from frame using YOLO bounding box + padding.
        This preserves the entire person including lower body (where guns are often hidden).
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Calculate padding
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = int(box_w * padding)
        pad_y = int(box_h * padding)
        
        # Expand bbox with padding (clamp to frame edges)
        new_x1 = max(0, x1 - pad_x)
        new_x2 = min(w, x2 + pad_x)
        new_y1 = max(0, y1 - pad_y)
        new_y2 = min(h, y2 + pad_y)
        
        # Crop and resize to 640x640 (what model expects)
        person_crop = frame[new_y1:new_y2, new_x1:new_x2]
        
        if person_crop.size == 0:
            return None
        
        person_resized = cv2.resize(person_crop, (640, 640))
        
        return person_resized, (new_x1, new_y1, new_x2, new_y2)
    
    if video_file is not None:
        # Display video preview
        st.video(video_file)
        
        if st.button("🎬 Process Video Queue", key="scan_video_queue", type="primary"):
            if not thermal_online:
                st.error("❌ Thermal API is not running. Please start it with: python thermal_api.py")
            else:
                with st.spinner("Processing video - extracting frames and detecting weapons..."):
                    try:
                        # Load person detector
                        person_model = load_person_detector()
                        
                        # Save uploaded video to temporary file
                        temp_video_path = Path("temp_uploaded_video.mp4")
                        with open(temp_video_path, "wb") as f:
                            f.write(video_file.read())
                        
                        # Open video with OpenCV
                        cap = cv2.VideoCapture(str(temp_video_path))
                        
                        if not cap.isOpened():
                            st.error("Could not open video file")
                        else:
                            # Get video info
                            fps = int(cap.get(cv2.CAP_PROP_FPS))
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            duration = total_frames / fps if fps > 0 else 0
                            
                            st.info(f"📹 Video info: {duration:.1f} seconds, {fps} fps, {total_frames} frames")
                            st.info(f"🔧 Using {padding_percent*100:.0f}% padding around each person to capture full body (including pockets)")
                            
                            # Determine frame indices to extract
                            if total_frames <= num_frames_to_extract:
                                frame_indices = list(range(total_frames))
                            else:
                                step = total_frames // num_frames_to_extract
                                frame_indices = [i * step for i in range(num_frames_to_extract)]
                            
                            st.info(f"📸 Extracting {len(frame_indices)} frames")
                            
                            # Store results per frame
                            all_frames_results = []
                            processed_frames = []
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for idx, frame_num in enumerate(frame_indices):
                                status_text.text(f"Processing frame {idx+1}/{len(frame_indices)}...")
                                
                                # Seek to frame
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                                ret, frame = cap.read()
                                
                                if not ret:
                                    continue
                                
                                # Calculate timestamp
                                timestamp = frame_num / fps if fps > 0 else 0
                                
                                # ========== PREPROCESS: Color mapping ==========
                                frame_processed = preprocess_thermal_frame(frame, colormap_option)
                                
                                # Detect people in this frame
                                person_results = person_model(frame_processed, verbose=False)
                                
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
                                
                                frame_results = []
                                for person_idx, person in enumerate(person_boxes):
                                    person_num = person_idx + 1
                                    
                                    # ========== SMART CROP with padding ==========
                                    crop_result = smart_crop_person(
                                        frame_processed, 
                                        person["bbox"], 
                                        padding_percent
                                    )
                                    
                                    if crop_result is None:
                                        continue
                                    
                                    person_resized, expanded_bbox = crop_result
                                    
                                    # Run weapon detection via API
                                    _, encoded = cv2.imencode('.jpg', person_resized)
                                    files = {"file": encoded.tobytes()}
                                    data = {"confidence": confidence_threshold_video}
                                    
                                    try:
                                        response = requests.post(THERMAL_API_URL, files=files, data=data, timeout=30)
                                        if response.status_code == 200:
                                            result = response.json()
                                            weapons = result.get("weapon_analysis", {}).get("weapons", []) if result.get("weapon_analysis") else []
                                        else:
                                            weapons = []
                                    except Exception as e:
                                        weapons = []
                                    
                                    has_weapon = len(weapons) > 0
                                    
                                    # Extract head (top 30% of original crop, not resized)
                                    head_h = int((person["bbox"][3] - person["bbox"][1]) * 0.3)
                                    head_img = frame_processed[person["bbox"][1]:person["bbox"][1]+head_h, 
                                                                person["bbox"][0]:person["bbox"][2]]
                                    
                                    frame_results.append({
                                        "person_num": person_num,
                                        "has_weapon": has_weapon,
                                        "weapons": weapons,
                                        "head_image": head_img,
                                        "person_image": person_resized
                                    })
                                
                                all_frames_results.append({
                                    "frame_index": frame_num,
                                    "timestamp": timestamp,
                                    "results": frame_results,
                                    "num_people": len(person_boxes)
                                })
                                
                                # Store processed frame for display (with bounding boxes drawn)
                                frame_with_boxes = frame_processed.copy()
                                for person in person_boxes:
                                    x1, y1, x2, y2 = person["bbox"]
                                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                processed_frames.append(frame_with_boxes)
                                
                                progress_bar.progress((idx + 1) / len(frame_indices))
                            
                            cap.release()
                            temp_video_path.unlink()
                            
                            status_text.text("Processing complete! Aggregating results...")
                            
                            # ========== AGGREGATE RESULTS ACROSS FRAMES ==========
                            max_people = max([f["num_people"] for f in all_frames_results]) if all_frames_results else 0
                            
                            # Track weapon detections per person across frames
                            person_tracking = {}
                            for person_num in range(1, max_people + 1):
                                person_tracking[person_num] = {
                                    "weapon_frames": [],
                                    "weapon_details": [],
                                    "head_image": None,
                                    "person_image": None,
                                    "first_weapon_timestamp": None
                                }
                            
                            for frame_data in all_frames_results:
                                timestamp = frame_data["timestamp"]
                                for person_result in frame_data["results"]:
                                    pn = person_result["person_num"]
                                    if person_result["has_weapon"]:
                                        person_tracking[pn]["weapon_frames"].append(timestamp)
                                        for w in person_result["weapons"]:
                                            person_tracking[pn]["weapon_details"].append({
                                                "timestamp": timestamp,
                                                "weapon": "Gun",
                                                "confidence": w["confidence"]
                                            })
                                        if person_tracking[pn]["head_image"] is None:
                                            person_tracking[pn]["head_image"] = person_result["head_image"]
                                        if person_tracking[pn]["first_weapon_timestamp"] is None:
                                            person_tracking[pn]["first_weapon_timestamp"] = timestamp
                            
                            # Determine alerts based on confirmation threshold
                            alert_persons = []
                            for person_num, data in person_tracking.items():
                                unique_frames = len(set(data["weapon_frames"]))
                                if unique_frames >= min_confirmation_frames:
                                    alert_persons.append({
                                        "person_num": person_num,
                                        "weapon_frames": data["weapon_frames"],
                                        "weapon_details": data["weapon_details"],
                                        "head_image": data["head_image"],
                                        "first_alert": data["first_weapon_timestamp"]
                                    })
                            
                            # ========== DISPLAY RESULTS ==========
                            st.markdown("---")
                            st.subheader("📊 Processed Frames Preview")
                            st.caption(f"Color mapping: **{colormap_option}** | Person padding: **{padding_percent*100:.0f}%**")
                            
                            # Show processed frames
                            frame_cols = st.columns(min(len(processed_frames), 4))
                            for idx, frame in enumerate(processed_frames):
                                col_idx = idx % 4
                                with frame_cols[col_idx]:
                                    st.image(frame, caption=f"Frame {idx+1} (t={all_frames_results[idx]['timestamp']:.1f}s)", use_container_width=True)
                            
                            # ========== ALERT SUMMARY ==========
                            st.markdown("---")
                            st.subheader("🚨 VIDEO QUEUE ALERT SUMMARY")
                            
                            if len(alert_persons) > 0:
                                st.markdown(f'<div class="alert-red">⚠️ ALERT: {len(alert_persons)} person(s) confirmed with weapons across {min_confirmation_frames}+ frames!</div>', unsafe_allow_html=True)
                                
                                for ap in alert_persons:
                                    weapons_list = list(set([w["weapon"] for w in ap["weapon_details"]]))
                                    weapons_str = ", ".join(weapons_list)
                                    timestamps_str = ", ".join([f"{t:.1f}s" for t in ap["weapon_frames"]])
                                    
                                    if ap["person_num"] == 1:
                                        position = "leftmost (first to enter)"
                                    elif ap["person_num"] == max_people:
                                        position = "rightmost (last to leave)"
                                    else:
                                        position = f"{ap['person_num']}th from left"
                                    
                                    st.write(f"🔴 **Person {ap['person_num']}** ({position}): {weapons_str} detected at {timestamps_str}")
                                
                                # Display heads of alerted persons
                                st.subheader("👤 Security Alert - Identify These Individuals")
                                head_cols = st.columns(min(len(alert_persons), 4))
                                for idx, ap in enumerate(alert_persons):
                                    if idx < len(head_cols) and ap["head_image"] is not None:
                                        with head_cols[idx]:
                                            st.image(ap["head_image"], caption=f"Person {ap['person_num']} (SUSPECT)", use_container_width=True)
                                
                                # Timeline visualization
                                st.subheader("📈 Detection Timeline")
                                timeline_data = []
                                for ap in alert_persons:
                                    for wd in ap["weapon_details"]:
                                        timeline_data.append({
                                            "Person": f"Person {ap['person_num']}",
                                            "Time (s)": f"{wd['timestamp']:.1f}",
                                            "Weapon": wd["weapon"],
                                            "Confidence": f"{wd['confidence']:.0%}"
                                        })
                                if timeline_data:
                                    st.dataframe(timeline_data, use_container_width=True)
                                
                            else:
                                st.markdown(f'<div class="alert-green">✅ CLEAR: No weapons confirmed across {min_confirmation_frames}+ frames</div>', unsafe_allow_html=True)
                            
                            # ========== FULL DETAILS TABLE ==========
                            with st.expander("View full frame-by-frame details"):
                                details_data = []
                                for frame_data in all_frames_results:
                                    for person_result in frame_data["results"]:
                                        if person_result["has_weapon"]:
                                            weapons_str = ", ".join([w["class_name"] for w in person_result["weapons"]])
                                        else:
                                            weapons_str = "None"
                                        details_data.append({
                                            "Frame": frame_data["frame_index"],
                                            "Time (s)": f"{frame_data['timestamp']:.1f}",
                                            "Person": person_result["person_num"],
                                            "Weapons": weapons_str
                                        })
                                if details_data:
                                    st.dataframe(details_data, use_container_width=True)
                            
                            # Helpful tip if no detections
                            if len(alert_persons) == 0 and len(all_frames_results) > 0:
                                st.info("💡 **No weapons detected. Try:**\n"
                                        "- Increasing 'Person Crop Padding' (0.4-0.5) to capture more of the body\n"
                                        "- Lowering the confidence threshold\n"
                                        "- Selecting a different color mapping (HOT, INFERNO, etc.)\n"
                                        "- Extracting more frames from the video")
                            
                            st.session_state.video_queue_results = alert_persons
                            
                            status_text.text("")
                            progress_bar.empty()
                            
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                        
st.markdown("---")
st.caption("⚠️ Disclaimer: This system is for research purposes. Always follow official security protocols.")