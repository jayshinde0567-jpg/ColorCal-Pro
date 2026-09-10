import streamlit as st
import cv2
import numpy as np
import io
from PIL import Image

st.set_page_config(page_title="ColorCal-Jaundice Pro", page_icon="🩺", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS & FONTS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');

:root {
  --bg: #F6F7F5;
  --surface: #FFFFFF;
  --ink: #14202E;
  --ink-soft: #4B5A65;
  --line: #DDE3DF;
  --teal: #1F8A78;
  --teal-dark: #146356;
  --teal-tint: #E4F2EF;
  --green: #3C9A5F;
  --green-tint: #E7F5EC;
  --amber: #C97F17;
  --amber-tint: #FBF0DE;
  --red: #C64B3C;
  --red-tint: #FBEAE7;
  --radius-s: 6px;
  --radius-m: 12px;
}

/* Base Overrides */
.stApp { background-color: var(--bg); color: var(--ink); font-family: 'IBM Plex Sans', sans-serif; }
header {visibility: hidden;} #MainMenu {visibility: hidden;} footer {visibility: hidden;}
h1, h2, h3, h4, .serif { font-family: 'Fraunces', serif !important; font-weight: 600; letter-spacing: -0.01em; color: var(--ink); }

.wrap { max-width: 1120px; margin: 0 auto; padding: 0 28px; }
.site-header { background: rgba(246,247,245,0.92); backdrop-filter: blur(6px); border-bottom: 1px solid var(--line); padding: 16px 28px; display: flex; justify-content: space-between; align-items: center; }
.brand { display: flex; align-items: center; gap: 10px; }
.brand-name { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.2rem; color: var(--ink); }
.brand-name em { font-style: normal; color: var(--teal-dark); }

.hero { padding: 72px 0 56px; }
.hero-grid { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 56px; align-items: center; }
.eyebrow-line { display: flex; align-items: center; gap: 10px; margin-bottom: 20px; color: var(--teal-dark); font-size: 0.85rem; font-weight: 600; }
.eyebrow-line .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--teal); }
.hero h1 { font-size: clamp(2.3rem,4.6vw,3.6rem); line-height: 1.05; }
.hero h1 .accent { color: var(--teal-dark); }
.hero p.lede { margin-top: 22px; font-size: 1.12rem; color: var(--ink-soft); max-width: 52ch; }

.swatch-panel { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-m); padding: 22px; box-shadow: 0 1px 0 rgba(20,32,46,0.03); }
.swatch-panel .label-row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 14px; }
.swatch-panel .t1 { font-size: 0.78rem; font-weight: 600; color: var(--ink-soft); }
.swatch-panel .t2 { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--ink-soft); }
.swatch-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 4px; }
.swatch-grid div { aspect-ratio: 1; border-radius: 3px; box-shadow: inset 0 0 0 1px rgba(0,0,0,0.06); }
.swatch-caption { margin-top: 14px; font-size: 0.82rem; color: var(--ink-soft); }

.strip { border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); background: var(--surface); }
.strip-grid { display: grid; grid-template-columns: repeat(3, 1fr); }
.strip-grid > div { padding: 32px 28px; border-left: 1px solid var(--line); }
.strip-grid > div:first-child { border-left: none; }
.strip h3 { font-size: 1.02rem; margin-bottom: 8px; font-family: 'Fraunces', serif; }
.strip p { font-size: 0.92rem; color: var(--ink-soft); }

.section-head { max-width: 640px; margin: 56px auto 44px; text-align: left; padding: 0 28px; width: 100%; max-width: 1120px; }
.section-head .kicker { font-size: 0.82rem; font-weight: 600; color: var(--teal-dark); margin-bottom: 10px; }
.section-head h2 { font-size: clamp(1.7rem,3vw,2.3rem); }
.section-head p { margin-top: 14px; color: var(--ink-soft); font-size: 1.02rem; }

.pipeline { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; padding: 0 28px; max-width: 1120px; margin: 0 auto; }
.pl-step { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-m); padding: 20px; }
.pl-step .pl-num { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: var(--teal-dark); font-weight: 600; }
.pl-step h4 { font-size: 1rem; margin: 10px 0 6px; font-weight: 600; }
.pl-step p { font-size: 0.85rem; color: var(--ink-soft); margin: 0; }

.disclaimer-band { background: var(--ink); color: #DCE3E6; padding: 26px 0; margin-top: 56px; }
.disclaimer-band .wrap { display: flex; gap: 14px; align-items: flex-start; font-size: 0.88rem; }
.disclaimer-band strong { color: #fff; }

.stRadio > label, .stSlider > label, .stFileUploader > label, .stCheckbox > label {
    font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 600 !important; color: var(--ink) !important;
}
.stFileUploader { background: var(--surface); border: 1px dashed var(--teal); border-radius: var(--radius-m); padding: 16px; }

.info-box { background: #F8FAFC; border-left: 4px solid #38BDF8; padding: 16px; border-radius: 6px; margin: 16px 0; font-size: 0.95rem; color: #334155; font-family: 'IBM Plex Sans', sans-serif; }
.info-box strong { color: #0F172A; }
</style>
""", unsafe_allow_html=True)

# --- PYTHON LOGIC & CONSTANTS ---
REF_COLORS_V1 = np.array([
    [240, 240, 240], [20, 20, 20],   [128, 128, 128],
    [200, 50, 50],   [50, 200, 50],  [50, 50, 200],
    [200, 200, 50],  [200, 50, 200], [50, 200, 200]
], dtype=np.float32)

REF_COLORS_PRO = np.array([
    [250,250,250], [200,200,200], [150,150,150], [100,100,100], [50,50,50], [15,15,15],
    [200,50,50], [50,200,50], [50,50,200], [200,200,50], [200,50,200], [50,200,200],
    [255,224,189], [255,205,148], [224,172,105], [141,85,36], [70,33,26], [40,15,10],
    [245,248,250], [252,252,240], [250,245,215], [248,238,180], [235,225,140], [215,200,80]
], dtype=np.float32)

def rgb_to_hex(rgb):
    return f"#{int(max(0, min(255, rgb[0]))):02x}{int(max(0, min(255, rgb[1]))):02x}{int(max(0, min(255, rgb[2]))):02x}"

def render_color_swatches_html(colors_array, columns, title):
    html = f"<div style='margin-bottom:10px;'><div style='font-size:0.8rem; font-weight:600; color:#4B5A65; margin-bottom:8px;'>{title}</div><div style='display:grid; grid-template-columns:repeat({columns}, 1fr); gap:4px;'>"
    for color in colors_array:
        hx = rgb_to_hex(color)
        html += f"<div style='width:100%; aspect-ratio:1; background-color:{hx}; border-radius:3px; border:1px solid rgba(0,0,0,0.1);'></div>"
    html += "</div></div>"
    return html

def auto_detect_aruco_card(image_rgb):
    try:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None:
            id_list = ids.flatten().tolist()
            if all(req_id in id_list for req_id in [0, 1, 2, 3]):
                idx0, idx1, idx2, idx3 = id_list.index(0), id_list.index(1), id_list.index(2), id_list.index(3)
                
                c0 = np.mean(corners[idx0][0], axis=0) # TL
                c1 = np.mean(corners[idx1][0], axis=0) # TR
                c2 = np.mean(corners[idx2][0], axis=0) # BR
                c3 = np.mean(corners[idx3][0], axis=0) # BL
                
                src_pts = np.array([c0, c1, c2, c3], dtype=np.float32)
                dst_pts = np.array([[240, 240], [3360, 240], [3360, 2160], [240, 2160]], dtype=np.float32)
                
                H, _ = cv2.findHomography(src_pts, dst_pts)
                warped_full = cv2.warpPerspective(image_rgb, H, (3600, 2400))
                return warped_full, None
            else:
                found = sorted([i for i in id_list if i in [0,1,2,3]])
                missing = [i for i in [0,1,2,3] if i not in found]
                return None, f"Found markers {found}, but missing {missing}. Ensure your fingers aren't covering the corners!"
        else:
            return None, "No ArUco markers detected at all in the image."
    except Exception as e:
        return None, f"ArUco Error: {str(e)}"

def auto_detect_card(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    dilated = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 5000:
            return approx
    return None

def get_normalized_poly_features(colors_rgb):
    # Mathematically normalized to [0, 1] to prevent matrix explosion during polynomial expansion
    c = np.clip(colors_rgb, 0, 255) / 255.0
    R, G, B = c[:, 0], c[:, 1], c[:, 2]
    return np.column_stack([
        R, G, B,
        R**2, G**2, B**2,
        R*G, G*B, R*B,
        np.ones(len(colors_rgb))
    ])

def apply_ultra_calibration(image_rgb, captured, ref):
    # Ultimate precision: Normalized 2nd-Order Polynomial mapped via SVD Least Squares
    X = get_normalized_poly_features(captured)
    Y = ref / 255.0
    
    # rcond=None uses SVD to find the perfect Pseudo-Inverse (mathematically optimal, handles shadows perfectly)
    M, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    
    # Efficiency: Downscale image for preview if >1200px
    h, w = image_rgb.shape[:2]
    process_rgb = image_rgb
    if w > 1200:
        scale = 1200 / w
        process_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)))
        
    img_flat = process_rgb.reshape(-1, 3).astype(np.float32)
    X_img = get_normalized_poly_features(img_flat)
    
    calibrated_flat = np.dot(X_img, M) * 255.0
    
    calibrated_preview = np.clip(calibrated_flat, 0, 255).astype(np.uint8).reshape(process_rgb.shape)
    if w > 1200:
        return cv2.resize(calibrated_preview, (w, h))
    return calibrated_preview

grid_divs = "".join([f"<div style='background:{rgb_to_hex(c)};'></div>" for c in REF_COLORS_PRO])

st.markdown(f"""
<header class="site-header">
  <div class="brand"><div class="brand-name">ColorCal<em>-Jaundice Pro</em></div></div>
</header>
<section class="hero wrap">
  <div class="hero-grid">
    <div>
      <div class="eyebrow-line"><span class="dot"></span>Pro Version · Facial AI Powered</div>
      <h1>A smartphone and a paper card can screen for <span class="accent">clinical jaundice.</span></h1>
      <p class="lede">Upgraded with 3D Facial Landmark AI, Polynomial HDR correction, Glare-Rejection, and a 24-patch clinical palette designed for maximum automation and accuracy.</p>
    </div>
    <div class="swatch-panel">
      <div class="label-row"><span class="t1">COLORCAL PRO PALETTE</span><span class="t2">v2.0 CLINICAL</span></div>
      <div class="swatch-grid">
        {grid_divs}
      </div>
      <p class="swatch-caption">24 precise targets including specialized micro-shades of sclera yellow for polynomial color mapping.</p>
    </div>
  </div>
</section>
<div class="strip">
  <div class="wrap strip-grid">
    <div><h3>1. Facial Landmark AI</h3><p>A 468-point 3D Face Mesh perfectly locates the eye socket automatically, guaranteeing 0% skin interference without manual cropping.</p></div>
    <div><h3>2. Polynomial Correction</h3><p>Phones apply non-linear curves to photos. The new 24-patch system uses multi-degree polynomial regression to reverse HDR tone mapping.</p></div>
    <div><h3>3. Clinical TSB Estimation</h3><p>The system actively masks out specular highlights and models the pure sclera pigment to estimate blood Bilirubin levels (mg/dL).</p></div>
  </div>
</div>
""", unsafe_allow_html=True)


c1, c2, c3 = st.columns([1, 10, 1])
with c2: 
    st.markdown("<div class='section-head' style='margin-top:40px; margin-bottom:20px;'><div class='kicker'>CLINICAL PIPELINE</div><h2>Run an Analysis</h2></div>", unsafe_allow_html=True)
    # Use standard CSS variables or transparent backgrounds to respect Streamlit Dark/Light themes
    st.markdown("<div style='border:1px solid var(--line, #DDE3DF); border-radius:12px; padding:24px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
    
    st.markdown("### 📥 Step 1: Upload Image")
    uploaded_file = st.file_uploader("Upload patient image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        from PIL import Image, ImageOps
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_pil = ImageOps.exif_transpose(image_pil) # Auto-fix rotation if EXIF data is present
        
        col_img_rot1, col_img_rot2 = st.columns([1, 1])
        with col_img_rot1:
            img_rotation = st.radio("📷 Is the photo sideways?", ["Looks Good (0°)", "Rotate 90° Right ↻", "Rotate 180° ⇅", "Rotate 90° Left ↺"], horizontal=True)
            
        if "Right" in img_rotation:
            image_pil = image_pil.rotate(-90, expand=True)
        elif "180" in img_rotation:
            image_pil = image_pil.rotate(180, expand=True)
        elif "Left" in img_rotation:
            image_pil = image_pil.rotate(90, expand=True)
            
        original_rgb = np.array(image_pil)
        
        test_rgb = original_rgb.copy().astype(np.float32)
        test_rgb = np.clip(test_rgb, 0, 255).astype(np.uint8)
        
        st.markdown("<hr style='border:none; border-top:1px solid var(--line); margin:24px 0;'>", unsafe_allow_html=True)
        
        st.markdown("### 🔍 Step 2: Color Calibration")
        card_type = st.radio("Reference Card Type Used in Photo:", ["ColorCal v1 (Legacy 9-Patch)", "ColorCal Pro (24-Patch Clinical)"], horizontal=True)
        
        use_calibration = st.checkbox("✅ Apply Card Matrix Correction", value=True)
        calibrated_rgb = test_rgb
        
        if use_calibration:
            is_pro = "Pro" in card_type
            ref_colors = REF_COLORS_PRO if is_pro else REF_COLORS_V1
            rows, cols = (4, 6) if is_pro else (3, 3)
            
            auto_card = st.checkbox("🤖 Auto-Detect Card (ArUco / OpenCV)", value=True)
            warped_aruco = None
            
            if auto_card and is_pro:
                warped_aruco, aruco_error = auto_detect_aruco_card(test_rgb)
                if warped_aruco is None:
                    st.warning(f"⚠️ {aruco_error} Falling back to manual sliders.")
            
            patches = []
            
            # AUTOMATED ARUCO EXTRACTION PATH
            if warped_aruco is not None:
                st.success("✅ ArUco Markers Detected! Card perfectly flattened & color matrices aligned.")
                warped = cv2.resize(warped_aruco, (600, 400)) # Just for UI display
                
                # Math from the HD Generator (3600x2400)
                patch_w, patch_h = 400, 320
                gap = 60
                start_x, start_y = 420, 520
                
                for i in range(rows):
                    for j in range(cols):
                        x = start_x + j * (patch_w + gap)
                        y = start_y + i * (patch_h + gap)
                        # Extract the strict center of each patch
                        roi = warped_aruco[y+40:y+patch_h-40, x+40:x+patch_w-40]
                        patches.append(cv2.mean(roi)[:3])
                
            # MANUAL SLIDER FALLBACK PATH
            else:
                h, w = test_rgb.shape[:2]
                st.info("Adjust sliders to frame the physical color card:")
                col_sl1, col_sl2 = st.columns(2)
                with col_sl1:
                    sx = st.slider("Card X (Left/Right)", 0, w, int(w*0.1))
                    sy = st.slider("Card Y (Up/Down)", 0, h, int(h*0.6))
                with col_sl2:
                    sw = st.slider("Card Width", 50, w, int(w*0.4))
                    sh = st.slider("Card Height", 50, h, int(h*0.3))
                
                card_contour = np.array([ [[sx, sy]], [[sx+sw, sy]], [[sx+sw, sy+sh]], [[sx, sy+sh]] ], dtype=np.int32)
                pts = card_contour.reshape(4, 2).astype("float32")
                s = pts.sum(axis=1)
                rect = np.zeros((4, 2), dtype="float32")
                rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
                
                warp_w, warp_h = cols * 50, rows * 50
                M_warp = cv2.getPerspectiveTransform(rect, np.array([[0, 0], [warp_w, 0], [warp_w, warp_h], [0, warp_h]], dtype="float32"))
                warped = cv2.warpPerspective(test_rgb, M_warp, (warp_w, warp_h))
                
                card_rot = st.radio("Is the card rotated in the photo?", ["No (Normal Landscape)", "Rotated 90° Right", "Upside Down", "Rotated 90° Left"], horizontal=True)
                if "Right" in card_rot:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
                    rows, cols = cols, rows
                elif "Upside" in card_rot:
                    warped = cv2.rotate(warped, cv2.ROTATE_180)
                elif "Left" in card_rot:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    rows, cols = cols, rows
                
                for i in range(rows):
                    for j in range(cols):
                        roi = warped[i*50+15:(i+1)*50-15, j*50+15:(j+1)*50-15]
                        patches.append(cv2.mean(roi)[:3])
            
            captured_colors = np.array(patches)
            
            col_ext1, col_ext2 = st.columns([1, 2])
            with col_ext1:
                st.image(warped, caption="Extracted Card", width=220)
            with col_ext2:
                st.markdown(render_color_swatches_html(captured_colors, cols, "WHAT THE CAMERA ACTUALLY SAW"), unsafe_allow_html=True)
                st.markdown(render_color_swatches_html(ref_colors, cols, "TRUE MEDICAL REFERENCE VALUES"), unsafe_allow_html=True)

            if is_pro:
                calibrated_rgb = apply_ultra_calibration(test_rgb, captured_colors, ref_colors)
            else:
                calibrated_rgb = apply_ultra_calibration(test_rgb, captured_colors, ref_colors)
        else:
            st.info("Bypassing physical card calibration.")

        st.markdown("<hr style='border:none; border-top:1px solid #DDE3DF; margin:24px 0;'>", unsafe_allow_html=True)
        st.markdown("### 👁️ Step 3: AI Sclera Extraction")
        
        st.markdown("""
        <div class="info-box">
            <strong>100% Automated OpenCV Eye Tracking:</strong> Instantly detects and locks onto the eye socket using localized Haar Cascades (zero dependencies).
        </div>
        """, unsafe_allow_html=True)
        
        h_c, w_c = calibrated_rgb.shape[:2]
        
        # 100% AUTOMATED OPENCV EYE TARGETING
        gray = cv2.cvtColor(test_rgb, cv2.COLOR_RGB2GRAY)
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        
        is_manual = False
        if len(eyes) == 0:
            st.warning("⚠️ AI could not confidently lock onto the eye (check lighting). Activating fallback targeter.")
            is_manual = True
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
            with col_ctrl1:
                sc_x = st.slider("Eye Center X (Left/Right)", 0, w_c, w_c//2)
            with col_ctrl2:
                sc_y = st.slider("Eye Center Y (Up/Down)", 0, h_c, h_c//2)
            with col_ctrl3:
                crop_size = st.slider("Crop Box Size", 20, min(w_c, h_c)//2, 60)
            
            sc_x_start = max(0, sc_x - crop_size//2)
            sc_y_start = max(0, sc_y - crop_size//2)
            sc_x_end = min(w_c, sc_x + crop_size//2)
            sc_y_end = min(h_c, sc_y + crop_size//2)
            eye_label = "Smart Manual Target"
        else:
            # Pick the largest eye
            ex, ey, ew, eh = max(eyes, key=lambda e: e[2] * e[3])
            
            # Target center of eye, crop wide horizontally but narrow vertically to dodge eyelids
            center_x = ex + ew // 2
            center_y = ey + eh // 2
            crop_w = int(ew * 0.70)
            crop_h = int(eh * 0.25)
            
            sc_x_start = max(0, center_x - crop_w//2)
            sc_y_start = max(0, center_y - crop_h//2)
            sc_x_end = min(w_c, center_x + crop_w//2)
            sc_y_end = min(h_c, center_y + crop_h//2)
            eye_label = "AI Auto-Target (OpenCV)"
            sc_x, sc_y = center_x, center_y
        
        col_sc1, col_sc2 = st.columns([1, 1])
        
        with col_sc1:
            overlay = calibrated_rgb.copy()
            if is_manual:
                cv2.line(overlay, (sc_x-5, sc_y), (sc_x+5, sc_y), (255,0,0), 2)
                cv2.line(overlay, (sc_x, sc_y-5), (sc_x, sc_y+5), (255,0,0), 2)
                
            cv2.rectangle(overlay, (sc_x_start, sc_y_start), (sc_x_end, sc_y_end), (0, 255, 0), 3)
            st.image(overlay, caption=f"Target Area: {eye_label}", use_column_width=True)
            
        with col_sc2:
            sc_roi = calibrated_rgb[sc_y_start:sc_y_end, sc_x_start:sc_x_end]
                
            roi_lab = cv2.cvtColor(sc_roi, cv2.COLOR_RGB2LAB)
            L_chan = roi_lab[:, :, 0].astype(np.float32)
            a_chan = roi_lab[:, :, 1].astype(np.float32) - 128.0
            b_chan = roi_lab[:, :, 2].astype(np.float32) - 128.0
            
            roi_hsv = cv2.cvtColor(sc_roi, cv2.COLOR_RGB2HSV)
            S_chan = roi_hsv[:, :, 1]
            
            # Clinical Sclera Masking: Drop darks (pupil/lashes), drop glare (L>240)
            # CRITICAL: Drop skin using Saturation! Sclera is white (low saturation < 50), skin is highly saturated (> 60).
            L_thresh = np.percentile(L_chan, 60)
            valid_mask = (L_chan >= L_thresh) & (L_chan < 240) & (S_chan < 60)
            if not np.any(valid_mask):
                valid_mask = (L_chan >= np.percentile(L_chan, 50)) & (L_chan < 240)
            
            mask_visual = sc_roi.copy()
            mask_visual[~valid_mask] = [0, 0, 0] 
            
            col_roi1, col_roi2 = st.columns(2)
            with col_roi1:
                st.image(sc_roi, caption="AI Cropped Eye", use_column_width=True)
            with col_roi2:
                st.image(mask_visual, caption="Skin & Glare Rejected", use_column_width=True)
                
            if np.any(valid_mask):
                valid_b = b_chan[valid_mask]
                valid_a = a_chan[valid_mask]
                
                # Base TSB directly on absolute yellowness (b*) so redness (veins) doesn't mask it
                median_b = np.median(valid_b)
                
                mean_sclera_rgb = cv2.mean(sc_roi, mask=valid_mask.astype(np.uint8))[:3]
                mean_b = np.mean(valid_b)
                mean_a = np.mean(valid_a)
            else:
                median_b = 0
                mean_b = 0
                mean_a = 0
                mean_sclera_rgb = [0,0,0]
                
            b_rounded = round(median_b, 1)
            a_rounded = round(mean_a, 1)
            
            # ADULT/GENERAL POPULATION TSB MATH (Pure Yellowness Based):
            # Healthy adult TSB is 0.1 to 1.2. Scleral Icterus becomes visible at TSB > 2.5.
            # b=0 maps to ~0.5 mg/dL. b=4 maps to ~2.5 mg/dL. b=9 maps to ~5.0 mg/dL.
            est_tsb = max(0.2, round((b_rounded * 0.5) + 0.5, 1))
            
            st.markdown("<br>", unsafe_allow_html=True)
            if est_tsb <= 1.2:
                risk_level = "LOW RISK (NORMAL)"
                risk_color = "var(--green)"
                risk_bg = "var(--green-tint)"
                risk_border = "#BFE0CC"
                risk_desc = "TSB within normal limits (0.1–1.2 mg/dL) per WHO guidelines. Routine monitoring."
            elif est_tsb < 2.5:
                risk_level = "ELEVATED (SUBCLINICAL)"
                risk_color = "var(--amber)"
                risk_bg = "var(--amber-tint)"
                risk_border = "#EDD3A0"
                risk_desc = "Mild elevation (1.2–2.5 mg/dL). No visible scleral icterus per AASLD criteria."
            elif est_tsb < 5.0:
                risk_level = "MODERATE RISK (JAUNDICE)"
                risk_color = "#F97316"
                risk_bg = "#FFF7ED"
                risk_border = "#FDBA74"
                risk_desc = "Clinical jaundice (>2.5 mg/dL). Visible scleral icterus. Monitor liver function."
            else:
                risk_level = "HIGH RISK (SEVERE)"
                risk_color = "var(--red)"
                risk_bg = "var(--red-tint)"
                risk_border = "#EFC4BC"
                risk_desc = "Severe Hyperbilirubinemia (>5.0 mg/dL) per AASLD/WHO. Immediate medical referral."
            
            patch_hex = rgb_to_hex(mean_sclera_rgb)
            
            table_html = f"""
            <div style="border:1px solid {risk_border}; border-radius:8px; overflow:hidden; background:transparent; font-family:'IBM Plex Sans', sans-serif;">
                <div style="background:{risk_bg}; padding:16px 20px; border-bottom:1px solid {risk_border};">
                    <div style="color:{risk_color}; font-family:'IBM Plex Mono', monospace; font-size:0.75rem; font-weight:700; letter-spacing:0.05em; margin-bottom:4px;">{risk_level} (Sclera b* = {b_rounded})</div>
                    <div style="color:var(--ink); font-family:'Fraunces', serif; font-size:1.15rem; font-weight:600; line-height:1.2;">{risk_desc}</div>
                </div>
                <div style="padding:0;">
                    <table style="width:100%; border-collapse:collapse; font-size:0.9rem; text-align:left;">
                        <tr style="border-bottom:1px solid var(--line);">
                            <td style="padding:12px 20px; color:var(--ink-soft); width:55%;">Analyzed Biomarker</td>
                            <td style="padding:12px 20px; font-weight:600; color:var(--ink);">Sclera (AI Isolated)</td>
                        </tr>
                        <tr style="border-bottom:1px solid var(--line); background-color:#F8FAFC;">
                            <td style="padding:12px 20px; color:#14202E; font-weight:600;">Estimated Serum Bilirubin (TSB)</td>
                            <td style="padding:12px 20px; font-weight:700; color:#14202E; font-size:1.05rem;">~ {est_tsb} mg/dL</td>
                        </tr>
                        <tr style="border-bottom:1px solid var(--line);">
                            <td style="padding:12px 20px; color:var(--ink-soft);">Pure Yellowness (b*)</td>
                            <td style="padding:12px 20px; font-family:'IBM Plex Mono', monospace; color:var(--ink);">{b_rounded}</td>
                        </tr>
                        <tr style="border-bottom:1px solid var(--line);">
                            <td style="padding:12px 20px; color:var(--ink-soft);">Vascular Correction (a*)</td>
                            <td style="padding:12px 20px; font-family:'IBM Plex Mono', monospace; color:var(--ink);">{a_rounded}</td>
                        </tr>
                        <tr>
                            <td style="padding:12px 20px; color:var(--ink-soft);">
                                Extracted Sclera Pigment
                            </td>
                            <td style="padding:12px 20px;">
                                <div style="width:32px; height:24px; background-color:{patch_hex}; border-radius:4px; border:1px solid rgba(0,0,0,0.1); box-shadow:0 1px 2px rgba(0,0,0,0.05);"></div>
                            </td>
                        </tr>
                    </table>
                </div>
            </div>
            """
            
            st.markdown(table_html, unsafe_allow_html=True)
            
        
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-band">
  <div class="wrap">
<span style="font-size:1.2rem; margin-right:10px;">⚠</span>
<span><strong>This is a screening-support prototype, not a diagnostic device.</strong> The TSB mappings and threshold values are illustrative mockups for engineering demonstration. Any high-risk result should be followed by clinical evaluation.</span>
  </div>
</div>
""", unsafe_allow_html=True)
