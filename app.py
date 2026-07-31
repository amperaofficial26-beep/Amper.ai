import base64
import gc
import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# Untuk uji coba pertama: sistem login & kredit dimatikan dulu.
REQUIRE_LOGIN = False

if REQUIRE_LOGIN:
    try:
        from auth import render_auth_sidebar, get_credits, deduct_credit
    except ImportError:
        st.error("Module 'auth.py' tidak ditemukan!")

st.set_page_config(
    page_title="AMPER.AI - Pro Suite, & Yuki-Chan (Ai)",
    page_icon="👾",
    layout="wide",
)

LOGO_PATH = "logo_amper.png"
BG_PATH = "bg_amper.jpg"

MAX_INPUT_DIM = 3000
MAX_OUTPUT_MEGAPIXELS = 35_000_000


def get_base64_of_bin_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def set_background(image_path):
    try:
        bin_str = get_base64_of_bin_file(image_path)
        css = f"""
        <style>
        .stApp {{
            background-image: linear-gradient(160deg, rgba(6,17,20,0.55), rgba(6,17,20,0.55)),
                url("data:image/jpeg;base64,{bin_str}");
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        pass


def set_custom_theme():
    css = """
    <style>
    .stApp {
        background: linear-gradient(160deg, #07161a 0%, #0f2b30 55%, #07161a 100%);
        color: #e7ede9;
    }

    h1, h2, h3 {
        color: #e3b34a !important;
        font-family: 'Georgia', serif;
        letter-spacing: 0.3px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c2226 0%, #133c3f 100%);
        border-right: 1px solid rgba(227, 179, 74, 0.25);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f3cf83 !important;
    }
    section[data-testid="stSidebar"] label {
        color: #cfe8e1 !important;
    }

    .stButton>button {
        background: linear-gradient(90deg, #1f6f5c, #e3b34a);
        color: #06120f;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 0.65em 1.3em;
        transition: all 0.25s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 16px rgba(227, 179, 74, 0.35);
    }

    .stAlert {
        background-color: rgba(15, 43, 48, 0.85);
        color: #f3cf83;
        border: 1px solid rgba(227, 179, 74, 0.4);
    }

    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #e3b34a !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def compute_auto_suggestions(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    contrast_std = float(np.std(gray))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    target_brightness = 125.0
    diff = target_brightness - mean_brightness
    suggested_exposure = float(np.clip(diff / 90.0, -1.2, 1.2))
    suggested_contrast = int(np.clip((45 - contrast_std) * 1.1, 0, 40))

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total_px = gray.size
    shadow_clip_ratio = hist[:15].sum() / total_px
    highlight_clip_ratio = hist[240:].sum() / total_px
    suggested_shadows = int(np.clip(shadow_clip_ratio * 400, 0, 60))
    suggested_highlights = int(np.clip(-highlight_clip_ratio * 400, -60, 0))

    if laplacian_var < 60:
        suggested_sharpen, suggested_clarity = 55, 30
    elif laplacian_var < 150:
        suggested_sharpen, suggested_clarity = 35, 20
    else:
        suggested_sharpen, suggested_clarity = 15, 10

    return {
        "exposure": round(suggested_exposure, 1),
        "contrast": suggested_contrast,
        "highlights": suggested_highlights,
        "shadows": suggested_shadows,
        "sharpen": suggested_sharpen,
        "clarity": suggested_clarity,
        "whites": 0,
        "blacks": 0,
        "temp": -5,
        "tint": 0,
        "vibrance": 15,
        "saturation": 10,
        "dehaze": 10,
        "vignette": 25,
        "noise_reduction": 0,
        "smart_enhance": 0,
    }


def apply_tone_curve(img_f, curve_preset):
    if curve_preset == "Linear (Standard)":
        return img_f
    elif curve_preset == "S-Curve (Kontras Tinggi & Sinematik)":
        return np.sin(img_f * np.pi - np.pi / 2) * 0.5 + 0.5
    elif curve_preset == "Matte / Fade (Gaya Film Indie)":
        return img_f * 0.8 + 0.1
    elif curve_preset == "Bright Pop (Terang & Segar)":
        return np.power(img_f, 0.85)
    return img_f
    def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _cv2_to_pil(arr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _apply_exposure_contrast(img, exposure, contrast, whites, blacks):
    img = img.astype(np.float32)
    img = img * (2.0 ** exposure)
    factor = (259 * (contrast + 255)) / (255 * (259 - contrast)) if contrast != 0 else 1.0
    img = factor * (img - 128) + 128
    if whites != 0:
        mask = img > 200
        img[mask] += whites * 0.5
    if blacks != 0:
        mask = img < 55
        img[mask] += blacks * 0.5
    return np.clip(img, 0, 255).astype(np.uint8)


def _apply_highlights_shadows(img, highlights, shadows):
    img = img.astype(np.float32) / 255.0
    luminance = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
    if highlights != 0:
        highlight_mask = np.clip((luminance - 0.5) * 2, 0, 1)[:, :, None]
        img = img + (highlights / 100.0) * highlight_mask * (1 - img) * 0.5
    if shadows != 0:
        shadow_mask = np.clip((0.5 - luminance) * 2, 0, 1)[:, :, None]
        img = img + (shadows / 100.0) * shadow_mask * img * 0.5
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _apply_shadow_lift_highlight_recovery(img, shadow_lift, highlight_recovery):
    if shadow_lift == 0 and highlight_recovery == 0:
        return img
    img_f = img.astype(np.float32) / 255.0
    lum = 0.299 * img_f[:, :, 2] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 0]
    if shadow_lift > 0:
        mask = np.clip(1 - lum * 3, 0, 1)[:, :, None]
        img_f = img_f + (shadow_lift / 100.0) * mask * (0.5 - img_f) * 0.6
    if highlight_recovery > 0:
        mask = np.clip((lum - 0.75) * 4, 0, 1)[:, :, None]
        img_f = img_f - (highlight_recovery / 100.0) * mask * (img_f - 0.75) * 0.6
    return np.clip(img_f * 255, 0, 255).astype(np.uint8)


def _apply_temp_tint(img, temp, tint):
    img = img.astype(np.float32)
    img[:, :, 2] += temp * 0.6
    img[:, :, 0] -= temp * 0.6
    img[:, :, 1] += tint * 0.5
    return np.clip(img, 0, 255).astype(np.uint8)


def _apply_vibrance_saturation(img, vibrance, saturation):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    if saturation != 0:
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + saturation / 50.0), 0, 255)
    if vibrance != 0:
        sat = hsv[:, :, 1] / 255.0
        vib_mask = 1 - sat
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + (vibrance / 50.0) * vib_mask * 60, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _apply_clarity_dehaze(img, clarity, dehaze):
    if clarity != 0:
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        img = cv2.addWeighted(img, 1 + clarity / 100.0, blurred, -clarity / 100.0, 0)
    if dehaze != 0:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        strength = 2.0 + abs(dehaze) / 25.0
        clahe = cv2.createCLAHE(clipLimit=strength, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel) if dehaze > 0 else cv2.GaussianBlur(l_channel, (0, 0), 2)
        img = cv2.cvtColor(cv2.merge((l_channel, a, b)), cv2.COLOR_LAB2BGR)
    return img


def _apply_sharpen(img, sharpen, radius):
    if sharpen <= 0:
        return img
    radius = max(1, radius)
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    amount = sharpen / 100.0
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def _apply_smart_detail_enhance(img, strength, radius):
    if strength <= 0:
        return img
    radius = max(1, radius)
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    detail = cv2.subtract(img, blurred)
    boosted = cv2.addWeighted(img, 1.0, detail, strength / 40.0, 0)
    return boosted


def _apply_denoise(img, luminance_strength, color_strength):
    if luminance_strength <= 0 and color_strength <= 0:
        return img
    h_luma = max(1, int(luminance_strength))
    h_color = max(1, int(color_strength))
    return cv2.fastNlMeansDenoisingColored(img, None, h_luma, h_color, 7, 21)


def _apply_vignette(img, strength):
    if strength <= 0:
        return img
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols / (0.5 + strength / 100.0))
    kernel_y = cv2.getGaussianKernel(rows, rows / (0.5 + strength / 100.0))
    mask = kernel_y * kernel_x.T
    mask = mask / mask.max()
    vignette_mask = 1 - (1 - mask) * (strength / 100.0)
    out = img.astype(np.float32)
    for c in range(3):
        out[:, :, c] *= vignette_mask
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_upscale(img, upscale_choice):
    scale = 4 if "4x" in upscale_choice else 2
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


def apply_all_edits(pil_image: Image.Image, params: dict) -> Image.Image:
    """
    Menerapkan semua koreksi piksel berdasarkan nilai dari panel slider Ampera-AI.
    `params` = dict berisi semua key dari st.session_state, contoh:
    {
        "exposure": 0.0, "contrast": 0, "highlights": 0, "shadows": 0,
        "whites": 0, "blacks": 0, "temp": 0, "tint": 0,
        "vibrance": 0, "saturation": 0, "clarity": 0, "dehaze": 0,
        "sharpen": 0, "sharpen_radius": 2, "vignette": 0,
        "noise_reduction": 0, "noise_reduction_color": 0,
        "smart_enhance": 0, "smart_enhance_radius": 3,
        "highlight_recovery": 0, "shadow_lift": 0,
        "upscale_choice": "2x (HD 2K)",
    }
    """
    img = _pil_to_cv2(pil_image)

    img = _apply_denoise(img, params.get("noise_reduction", 0), params.get("noise_reduction_color", 0))
    img = _apply_exposure_contrast(img, params.get("exposure", 0.0), params.get("contrast", 0),
                                    params.get("whites", 0), params.get("blacks", 0))
    img = _apply_highlights_shadows(img, params.get("highlights", 0), params.get("shadows", 0))
    img = _apply_shadow_lift_highlight_recovery(img, params.get("shadow_lift", 0), params.get("highlight_recovery", 0))
    img = _apply_temp_tint(img, params.get("temp", 0), params.get("tint", 0))
    img = _apply_vibrance_saturation(img, params.get("vibrance", 0), params.get("saturation", 0))
    img = _apply_clarity_dehaze(img, params.get("clarity", 0), params.get("dehaze", 0))
    img = _apply_smart_detail_enhance(img, params.get("smart_enhance", 0), params.get("smart_enhance_radius", 3))
    img = _apply_sharpen(img, params.get("sharpen", 0), params.get("sharpen_radius", 2))
    img = _apply_vignette(img, params.get("vignette", 0))

    if params.get("upscale_choice"):
        img = _apply_upscale(img, params["upscale_choice"])

    return _cv2_to_pil(img)


# Inisialisasi Default State jika belum ada
default_state = {
    "exposure": 0.0, "contrast": 10, "highlights": -20, "shadows": 25,
    "whites": 0, "blacks": 0, "temp": -5, "tint": 0, "vibrance": 15,
    "saturation": 10, "clarity": 20, "dehaze": 10, "sharpen": 30,
    "vignette": 25, "noise_reduction": 0, "smart_enhance": 0
}
for key, val in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = val

set_custom_theme()
set_background(BG_PATH)

current_user = None
if REQUIRE_LOGIN:
    is_logged_in = render_auth_sidebar()
    if not is_logged_in:
        st.title("👾 AMPER.AI — Pro Suite & Yuki-Chan")
        st.info("Silakan Masuk atau Daftar lewat panel kiri untuk mulai.")
        st.stop()
    current_user = st.session_state.get("user")

header_col1, header_col2 = st.columns([1, 6])
with header_col1:
    try:
        st.image(LOGO_PATH, use_container_width=True)
    except Exception:
        st.markdown("<h1 style='margin:0;'>👾</h1>", unsafe_allow_html=True)

with header_col2:
    st.title("AMPER.AI — Professional Editing, & Yuki-Chan Suite")
    st.markdown(
        "<p style='color: #a9d6c9; font-size: 1.05em;'>Platform pengolahan"
        " foto pro lengkap dengan efek perjelas wajah, latar belakang bokeh,"
        " dan Yuki Asisten AI!</p>",
        unsafe_allow_html=True,
    )

uploaded_file = st.file_uploader(
    "📂 Unggah File Foto Utama Kamu Disini... (JPG, JPEG, PNG)", 
    type=["jpg", "jpeg", "png"],
    key="main_photo_uploader"
)

img = None
auto_suggestions = None

if uploaded_file is not None:
    file_signature = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("last_file_signature") != file_signature:
        st.session_state["last_file_signature"] = file_signature
        st.session_state.pop("processed_img", None)
        st.session_state.pop("auto_applied_for", None)

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("❌ Oppss..Gagal membaca file gambar. Coba unggah file lain.")
        st.stop()

    h0, w0 = img.shape[:2]
    if max(h0, w0) > MAX_INPUT_DIM:
        input_scale = MAX_INPUT_DIM / max(h0, w0)
        img = cv2.resize(
            img,
            (int(w0 * input_scale), int(h0 * input_scale)),
            interpolation=cv2.INTER_AREA,
        )
        st.info("ℹ️ Foto asli diturunkan sementara ke resolusi aman untuk server.")

    auto_suggestions = compute_auto_suggestions(img)

    if st.session_state.get("auto_applied_for") != file_signature:
        for slider_key, val in auto_suggestions.items():
            st.session_state[slider_key] = val
        st.session_state["auto_applied_for"] = file_signature

# ==========================================================
# SIDEBAR KONTROL
# ==========================================================
with st.sidebar:
    st.markdown("## 👾 Pro Suite & Ampera AI")

    st.markdown("### 🎭 Face & Body Professional Retouch")
    remini_boost = st.slider("Perjelas Wajah & Kulit (Remini Effect)", 0, 100, 0, 1)
    body_slim = st.slider("Body Slimming & Contour Pro", 0, 100, 0, 1)
    bg_blur = st.slider("Efek Latar Belakang (Bokeh / Blur Halus)", 0, 100, 0, 2)

    st.markdown("### 📍 Selective Edit (Control Points)")
    enable_selective = st.checkbox("Aktifkan Selective Control Point")
    sel_x_pct = st.slider("Titik Kontrol X (Posisi Horizontal %)", 0, 100, 50, 1)
    sel_y_pct = st.slider("Titik Kontrol Y (Posisi Vertikal %)", 0, 100, 50, 1)
    sel_radius = st.slider("Radius Area Pengaruh", 20, 300, 100, 5)
    sel_exposure = st.slider("Exposure Khusus Area", -1.0, 1.0, 0.0, 0.1)
    sel_sat = st.slider("Saturasi Khusus Area", -50, 50, 0, 1)

    st.markdown("### 🧬 Layer Blending (Gabung Foto)")
    enable_layer = st.checkbox("Aktifkan Gabung Layer Kedua")
    layer_file = st.file_uploader("Unggah Foto Kedua (Layer Overlay)", type=["jpg", "jpeg", "png"], key="layer_uploader")
    layer_opacity = st.slider("Opacity / Transparansi Layer", 0.0, 1.0, 0.5, 0.05)
    layer_mode = st.selectbox("Mode Blending", ["Normal", "Overlay", "Screen", "Multiply"])

    st.markdown("### 📈 RGB Tone Curve")
    curve_preset = st.selectbox(
        "Pilih Kurva Pencahayaan",
        [
            "Linear (Standard)",
            "S-Curve (Kontras Tinggi & Sinematik)",
            "Matte / Fade (Gaya Film Indie)",
            "Bright Pop (Terang & Segar)",
        ],
    )

    st.markdown("### 🎬 30+ Pro Filter Presets")
    capcut_preset = st.selectbox(
        "Pilih Filter / Template Gaya",
        [
            "Normal / Manual",
            "✨ Cyberpunk Neon (Pop & Vibrant)",
            "🎞️ Vintage Retro Film (Warm & Faded)",
            "🎬 Moody Cinematic (Dark & Deep)",
            "🌟 Clean & Fresh (Bright & Clear)",
            "☕ Warm Portrait (Skin Tone Enhancer)",
            "🖤 Dramatic B&W (Monochrome Pro)",
            "🌅 Golden Hour Sunset (Warm Glow)",
            "🌲 Emerald Forest (Deep Green Tone)",
            "🧊 Arctic Frost (Cool Blue Tone)",
            "🍑 Peach Blossom (Soft Pastel)",
            "🌇 Urban Dusk (City Twilight)",
            "🏝️ Tropical Vibrant (Saturated Summer)",
            "🍂 Autumn Warmth (Rustic Orange)",
            "❄️ Winter Blue (Cold Crisp)",
            "🎥 Film Noir (High Contrast B&W)",
            "🌸 Soft Pastel Dream (Airy & Light)",
            "🔥 High Contrast HDR (Punchy)",
            "🌊 Ocean Breeze (Teal & Cyan)",
            "🍇 Moody Purple Haze",
            "🌾 Rustic Earth Tone (Desaturated Brown)",
            "🌃 Neon Nightlife (Saturated Night)",
            "🥂 Elegant Wedding (Soft Highlights)",
            "📷 Analog Film 35mm",
            "🌵 Desert Sand (Warm Dusty)",
            "🍓 Berry Pop (Vivid Red/Pink)",
            "🌙 Midnight Blue (Deep Cool Shadow)",
            "🧁 Pastel Candy (Bright Playful)",
            "🏔️ Alpine Clarity (Sharp & Clean)",
            "🖼️ Editorial Matte (Flat Muted)",
            "🎇 Vibrant Pop Art",
            "🍃 Fresh Mint (Cool Green)",
            "🕯️ Warm Candlelight (Amber Glow)",
            "🌫️ Foggy Morning (Low Contrast Haze)",
        ],
    )

    preset_map = {
        "✨ Cyberpunk Neon (Pop & Vibrant)": {"exposure": 0.2, "contrast": 25, "highlights": -10, "shadows": 15, "temp": -15, "tint": 15, "vibrance": 35, "saturation": 20, "clarity": 25, "vignette": 40},
        "🎞️ Vintage Retro Film (Warm & Faded)": {"exposure": 0.1, "contrast": 10, "highlights": -20, "shadows": 30, "temp": 25, "tint": -5, "vibrance": -10, "saturation": -5, "clarity": 10, "vignette": 50},
        "🎬 Moody Cinematic (Dark & Deep)": {"exposure": -0.3, "contrast": 35, "highlights": -40, "shadows": -20, "temp": -10, "tint": 5, "vibrance": 10, "saturation": 5, "clarity": 30, "vignette": 65},
        "🌟 Clean & Fresh (Bright & Clear)": {"exposure": 0.3, "contrast": 15, "highlights": 10, "shadows": 25, "temp": 0, "tint": 0, "vibrance": 20, "saturation": 15, "clarity": 15, "vignette": 10},
        "☕ Warm Portrait (Skin Tone Enhancer)": {"exposure": 0.1, "contrast": 5, "highlights": 10, "shadows": 20, "temp": 15, "tint": 5, "vibrance": 15, "saturation": 5, "clarity": 5, "vignette": 15},
        "🖤 Dramatic B&W (Monochrome Pro)": {"exposure": 0.0, "contrast": 40, "highlights": -30, "shadows": -30, "temp": 0, "tint": 0, "vibrance": -50, "saturation": -50, "clarity": 35, "vignette": 50},
        "🌅 Golden Hour Sunset (Warm Glow)": {"exposure": 0.2, "contrast": 15, "highlights": 5, "shadows": 20, "temp": 30, "tint": 10, "vibrance": 25, "saturation": 15, "clarity": 10, "vignette": 20},
        "🌲 Emerald Forest (Deep Green Tone)": {"exposure": -0.1, "contrast": 20, "highlights": -10, "shadows": 10, "temp": -10, "tint": -20, "vibrance": 30, "saturation": 20, "clarity": 20, "vignette": 30},
        "🧊 Arctic Frost (Cool Blue Tone)": {"exposure": 0.1, "contrast": 10, "highlights": 15, "shadows": 10, "temp": -35, "tint": 10, "vibrance": 15, "saturation": 5, "clarity": 15, "vignette": 15},
        "🍑 Peach Blossom (Soft Pastel)": {"exposure": 0.2, "contrast": 5, "highlights": 20, "shadows": 25, "temp": 10, "tint": 15, "vibrance": 20, "saturation": 10, "clarity": 5, "vignette": 10},
        "🌇 Urban Dusk (City Twilight)": {"exposure": -0.1, "contrast": 25, "highlights": -15, "shadows": 5, "temp": 5, "tint": -10, "vibrance": 20, "saturation": 10, "clarity": 20, "vignette": 45},
        "🏝️ Tropical Vibrant (Saturated Summer)": {"exposure": 0.25, "contrast": 20, "highlights": 5, "shadows": 15, "temp": 5, "tint": 5, "vibrance": 40, "saturation": 30, "clarity": 15, "vignette": 10},
        "🍂 Autumn Warmth (Rustic Orange)": {"exposure": 0.05, "contrast": 15, "highlights": -5, "shadows": 15, "temp": 35, "tint": 5, "vibrance": 20, "saturation": 15, "clarity": 10, "vignette": 25},
        "❄️ Winter Blue (Cold Crisp)": {"exposure": 0.1, "contrast": 20, "highlights": 10, "shadows": 5, "temp": -30, "tint": 5, "vibrance": 5, "saturation": -5, "clarity": 20, "vignette": 20},
        "🎥 Film Noir (High Contrast B&W)": {"exposure": -0.1, "contrast": 45, "highlights": -35, "shadows": -35, "temp": 0, "tint": 0, "vibrance": -50, "saturation": -50, "clarity": 40, "vignette": 60},
        "🌸 Soft Pastel Dream (Airy & Light)": {"exposure": 0.35, "contrast": -10, "highlights": 20, "shadows": 30, "temp": 5, "tint": 10, "vibrance": 10, "saturation": -5, "clarity": -10, "vignette": 5},
        "🔥 High Contrast HDR (Punchy)": {"exposure": 0.1, "contrast": 45, "highlights": -25, "shadows": 30, "temp": 0, "tint": 0, "vibrance": 30, "saturation": 15, "clarity": 40, "vignette": 20},
        "🌊 Ocean Breeze (Teal & Cyan)": {"exposure": 0.1, "contrast": 15, "highlights": 5, "shadows": 10, "temp": -25, "tint": -15, "vibrance": 30, "saturation": 15, "clarity": 15, "vignette": 15},
        "🍇 Moody Purple Haze": {"exposure": -0.15, "contrast": 20, "highlights": -20, "shadows": -5, "temp": -5, "tint": 25, "vibrance": 20, "saturation": 10, "clarity": 20, "vignette": 40},
        "🌾 Rustic Earth Tone (Desaturated Brown)": {"exposure": 0.0, "contrast": 10, "highlights": -10, "shadows": 15, "temp": 20, "tint": 0, "vibrance": -20, "saturation": -15, "clarity": 15, "vignette": 30},
        "🌃 Neon Nightlife (Saturated Night)": {"exposure": -0.2, "contrast": 30, "highlights": -20, "shadows": -10, "temp": -20, "tint": 20, "vibrance": 40, "saturation": 25, "clarity": 25, "vignette": 50},
        "🥂 Elegant Wedding (Soft Highlights)": {"exposure": 0.25, "contrast": 5, "highlights": 25, "shadows": 20, "temp": 10, "tint": 5, "vibrance": 10, "saturation": 0, "clarity": 0, "vignette": 15},
        "📷 Analog Film 35mm": {"exposure": 0.05, "contrast": 15, "highlights": -15, "shadows": 20, "temp": 15, "tint": -5, "vibrance": -5, "saturation": -10, "clarity": 5, "vignette": 35},
        "🌵 Desert Sand (Warm Dusty)": {"exposure": 0.15, "contrast": 10, "highlights": -5, "shadows": 10, "temp": 30, "tint": -10, "vibrance": 10, "saturation": 5, "clarity": 15, "vignette": 20},
        "🍓 Berry Pop (Vivid Red/Pink)": {"exposure": 0.1, "contrast": 20, "highlights": 0, "shadows": 10, "temp": 5, "tint": 20, "vibrance": 35, "saturation": 25, "clarity": 15, "vignette": 15},
        "🌙 Midnight Blue (Deep Cool Shadow)": {"exposure": -0.25, "contrast": 25, "highlights": -15, "shadows": -25, "temp": -25, "tint": 0, "vibrance": 10, "saturation": 5, "clarity": 20, "vignette": 45},
        "🧁 Pastel Candy (Bright Playful)": {"exposure": 0.3, "contrast": -5, "highlights": 15, "shadows": 25, "temp": 5, "tint": 15, "vibrance": 25, "saturation": 15, "clarity": -5, "vignette": 5},
        "🏔️ Alpine Clarity (Sharp & Clean)": {"exposure": 0.15, "contrast": 20, "highlights": 5, "shadows": 15, "temp": -10, "tint": 0, "vibrance": 15, "saturation": 10, "clarity": 35, "vignette": 10},
        "🖼️ Editorial Matte (Flat Muted)": {"exposure": 0.05, "contrast": -15, "highlights": -20, "shadows": 20, "temp": 0, "tint": 0, "vibrance": -15, "saturation": -20, "clarity": 5, "vignette": 10},
        "🎇 Vibrant Pop Art": {"exposure": 0.15, "contrast": 35, "highlights": 0, "shadows": 10, "temp": 0, "tint": 0, "vibrance": 45, "saturation": 35, "clarity": 30, "vignette": 20},
        "🍃 Fresh Mint (Cool Green)": {"exposure": 0.2, "contrast": 10, "highlights": 10, "shadows": 15, "temp": -15, "tint": -20, "vibrance": 20, "saturation": 10, "clarity": 10, "vignette": 10},
        "🕯️ Warm Candlelight (Amber Glow)": {"exposure": 0.1, "contrast": 10, "highlights": -10, "shadows": 20, "temp": 40, "tint": 15, "vibrance": 15, "saturation": 10, "clarity": 5, "vignette": 30},
        "🌫️ Foggy Morning (Low Contrast Haze)": {"exposure": 0.2, "contrast": -20, "highlights": 15, "shadows": 25, "temp": -5, "tint": 0, "vibrance": -10, "saturation": -10, "clarity": -15, "vignette": 5},
    }

    if st.button("🪄 Terapkan Preset Pilihan"):
        if capcut_preset in preset_map:
            st.session_state.update(preset_map[capcut_preset])
        st.rerun()

    st.markdown("---")

    if auto_suggestions is not None and capcut_preset == "Normal / Manual":
        if st.button("🪄 Auto Enhance Standar"):
            for slider_key, val in auto_suggestions.items():
                st.session_state[slider_key] = val
            st.rerun()
        st.markdown("---")

    st.markdown("### 1. Light & Exposure")
    exposure = st.slider("Exposure", -2.0, 2.0, key="exposure", step=0.05)
    contrast = st.slider("Contrast", -50, 50, key="contrast", step=1)
    highlights = st.slider("Highlights", -100, 100, key="highlights", step=1)
    shadows = st.slider("Shadows", -100, 100, key="shadows", step=1)
    whites = st.slider("Whites", -50, 50, key="whites", step=1)
    blacks = st.slider("Blacks", -50, 50, key="blacks", step=1)

    st.markdown("### 2. Color & White Balance")
    temp = st.slider("Temperature (Kelvin/Tint)", -50, 50, key="temp", step=1)
    tint = st.slider("Tint", -50, 50, key="tint", step=1)
    vibrance = st.slider("Vibrance", -50, 50, key="vibrance", step=1)
    saturation = st.slider("Saturation", -50, 50, key="saturation", step=1)

    st.markdown("### 3. Detail, Clarity & Effects")
    clarity = st.slider("Clarity / Texture", -50, 50, key="clarity", step=1)
    dehaze = st.slider("Dehaze", -50, 50, key="dehaze", step=1)
    sharpen = st.slider("Sharpening HD", 0, 100, key="sharpen", step=1)
    sharpen_radius = st.slider("Sharpen Radius (px)", 1, 10, key="sharpen_radius", step=1)
    vignette = st.slider("Vignette (Cinematic Edge)", 0, 100, key="vignette", step=1)

    st.markdown("### 4. Quality Boost")
    denoise_strength = st.slider("Noise Reduction (Luminance)", 0, 30, key="noise_reduction", step=1)
    denoise_color = st.slider("Noise Reduction (Color)", 0, 30, key="noise_reduction_color", step=1)
    smart_enhance = st.slider("Smart Detail Enhance", 0, 100, key="smart_enhance", step=1)
    smart_enhance_radius = st.slider("Smart Detail Radius (px)", 1, 15, key="smart_enhance_radius", step=1)
    highlight_recovery = st.slider("Highlight Recovery", 0, 100, key="highlight_recovery", step=1)
    shadow_lift = st.slider("Shadow Lift (Adaptive)", 0, 100, key="shadow_lift", step=1)

    st.markdown("---")
    upscale_choice = st.selectbox(
        "Resolution Upscaling", ["2x (HD 2K)", "4x (Ultra HD 4K)"], index=0, key="upscale_choice"
    )
    process_btn = st.button("⬆️ Terapkan & Render Instan")

with st.sidebar:
    st.markdown("---")
    st.markdown("### 🌸 Asisten AI Yuki-Chan")

    yuki_html = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<style>
  :root{
    --bg-deep:#120c1e;
    --bg-panel:#251a3d;
    --bg-bubble-ai:#2a1f45;
    --bg-bubble-user:#3a2361;
    --accent-pink:#ff7aa8;
    --accent-pink-soft:#ff9dc0;
    --accent-gold:#ffca6b;
    --accent-cyan:#7fe9dc;
    --text-main:#f6f1ff;
    --text-muted:#a396c4;
    --text-faint:#6f6394;
    --border-glow:rgba(255,122,168,0.25);
  }
  *{box-sizing:border-box; margin:0; padding:0;}
  body{
    background:transparent;
    color:var(--text-main);
    font-family:'Inter', sans-serif;
    height:100%;
    display:flex;
    flex-direction:column;
  }
  .app{
    width:100%;
    height:480px;
    display:flex;
    flex-direction:column;
    background:rgba(18,12,30,0.92);
    border:1px solid rgba(255,122,168,0.25);
    border-radius:12px;
    overflow:hidden;
  }
  header{
    display:flex;
    align-items:center;
    gap:10px;
    padding:10px 12px;
    background:rgba(37,26,61,0.85);
    border-bottom:1px solid var(--border-glow);
  }
  .avatar{
    width:36px; height:36px; border-radius:50%; flex-shrink:0;
    background:radial-gradient(circle at 35% 30%, #ffe3ee 0%, #ff9dc0 45%, #7a4fa8 100%);
    position:relative; overflow:hidden;
  }
  .avatar svg{ width:100%; height:100%; display:block; }
  .id-name{ font-weight:700; font-size:0.9rem; color:var(--accent-pink-soft); }
  .id-role{ font-size:0.65rem; color:var(--text-muted); }

  main{
    flex:1; overflow-y:auto; padding:10px; display:flex; flex-direction:column; gap:10px;
  }
  main::-webkit-scrollbar{ width:4px; }
  main::-webkit-scrollbar-thumb{ background:rgba(255,122,168,0.3); border-radius:4px; }

  .row{ display:flex; gap:8px; max-width:100%; align-items:flex-end; }
  .row.user{ flex-direction:row-reverse; }
  .bubble{
    max-width:82%; padding:8px 12px; border-radius:12px; font-size:0.82rem; line-height:1.4;
    word-wrap:break-word; white-space:pre-wrap;
  }
  .row.ai .bubble{
    background:var(--bg-bubble-ai); border:1px solid rgba(255,202,107,0.18); border-bottom-left-radius:2px;
  }
  .row.user .bubble{
    background:var(--bg-bubble-user); border:1px solid rgba(127,233,220,0.2); border-bottom-right-radius:2px; color:#f3ecff;
  }
  .tag{ display:block; font-size:0.6rem; color:var(--accent-pink-soft); margin-bottom:2px; }

  .dialogue-wrap{ padding:8px; background:rgba(28,19,48,0.9); border-top:1px solid var(--border-glow); }
  .dialogue-box{ display:flex; align-items:flex-end; gap:6px; background:rgba(37,26,61,0.9); border:1px solid var(--border-glow); border-radius:8px; padding:6px 8px; }
  #userInput{
    flex:1; resize:none; background:transparent; border:none; outline:none; color:var(--text-main); font-size:0.82rem; max-height:60px;
  }
  #userInput::placeholder{ color:var(--text-faint); }
  #sendBtn{
    width:32px; height:32px; border-radius:50%; border:none; cursor:pointer;
    background:linear-gradient(135deg, var(--accent-pink), #c85f92); color:#fff;
    display:flex; align-items:center; justify-content:center;
  }
  #sendBtn svg{ width:14px; height:14px; }
</style>
</head>
<body>
<div class="app">
  <header>
    <div class="avatar">
      <svg viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="50" fill="#4a2f7a"/>
        <circle cx="50" cy="55" r="30" fill="#ffe3ee"/>
        <ellipse cx="38" cy="57" rx="4.5" ry="6" fill="#2b1a45"/>
        <ellipse cx="62" cy="57" rx="4.5" ry="6" fill="#2b1a45"/>
        <path d="M45 68 Q50 72 55 68" stroke="#c9758f" stroke-width="2" fill="none" stroke-linecap="round"/>
      </svg>
    </div>
    <div>
      <div class="id-name">Yuki</div>
      <div class="id-role">Asisten AI Serba Bisa & Teman Ngobrol</div>
    </div>
  </header>

  <main id="chatArea"></main>

  <div class="dialogue-wrap">
    <div class="dialogue-box">
      <textarea id="userInput" rows="1" placeholder="Mau ngobrol apa sama Yuki hari ini?..."></textarea>
      <button id="sendBtn" aria-label="Kirim">
        <svg viewBox="0 0 24 24" fill="none"><path d="M3 12L21 3L13 21L11 13L3 12Z" fill="currentColor"/></svg>
      </button>
    </div>
  </div>
</div>

<script>
  const chatArea = document.getElementById('chatArea');
  const userInput = document.getElementById('userInput');
  const sendBtn = document.getElementById('sendBtn');

  function addBubble(role, text){
    const row = document.createElement('div');
    row.className = 'row ' + (role === 'user' ? 'user' : 'ai');
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    if(role !== 'user'){
      const tag = document.createElement('span');
      tag.className = 'tag';
      tag.textContent = 'Yuki';
      bubble.appendChild(tag);
    }
    const textNode = document.createElement('span');
    textNode.textContent = text;
    bubble.appendChild(textNode);
    row.appendChild(bubble);
    chatArea.appendChild(row);
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  // --- KAMUS PENGETAHUAN AI YUKI (35+ KAMUS & TOPIK) ---
  const knowledgeBase = [
    {
      keywords: ['halo', 'hai', 'hi', 'hey', 'pagi', 'siang', 'sore', 'malam', 'salam', 'konnichiwa', 'assalamualaikum', 'woy', 'yo'],
      replies: [
        "Konnichiwa~ 🌸 Senang sekali bisa ngobrol sama kamu hari ini. Ada hal menarik apa yang ingin kita bahas?",
        "Halo! Selamat datang di ruang obrolan Yuki. Mau bahas soal foto, teknologi, atau sekadar ngobrol santai nih?"
      ]
    },
    {
      keywords: ['kabar', 'gimana kabar', 'sehat', 'keadaan', 'bagaimana kabarmu', 'kamu sehat', 'apa kabar'],
      replies: [
        "Alhamdulillah, Yuki selalu sehat, bugar, dan penuh semangat! 🌸 Kamu sendiri bagaimana keadaannya hari ini?"
      ]
    },
    {
      keywords: ['siapa', 'kamu siapa', 'yuki', 'nama kamu', 'pembuat', 'siapa yang buat', 'asal usul', 'robot', 'ai'],
      replies: [
        "Aku Yuki! Asisten virtual pribadimu yang dirancang untuk membantu urusan editing foto, sekaligus teman ngobrol yang asyik. 🌸"
      ]
    },
    {
      keywords: ['amper', 'amper.ai', 'ampera', 'platform', 'aplikasi ini', 'web apa ini'],
      replies: [
        "AMPER.AI adalah platform web editing foto profesional berteknologi tinggi yang dilengkapi fitur Remini face enhancer, bokeh, selective edit, layer blending, dan AI Upscaler! ✨"
      ]
    },
    {
      keywords: ['foto', 'gambar', 'edit', 'editing', 'filter', 'efek', 'kamera'],
      replies: [
        "Fotografi adalah seni melukis dengan cahaya! Di AMPER.AI kamu bisa mengatur exposure, kontras, kurva RGB, hingga memperhalus wajah secara instan. 📸"
      ]
    },
    {
      keywords: ['remini', 'wajah', 'kulit', 'mulus', 'halus', 'detail'],
      replies: [
        "Fitur Remini di sidebar menggunakan algoritma canggih untuk mempertajam detail wajah dan menghaluskan tekstur kulit agar tampak profesional! ✨"
      ]
    },
    {
      keywords: ['bokeh', 'blur', 'latar', 'background'],
      replies: [
        "Efek bokeh memberikan kesan kedalaman (depth of field) ala kamera DSLR mahal dengan mengaburkan bagian latar belakang foto secara halus. 🌿"
      ]
    },
    {
      keywords: ['upscale', 'resolusi', '4k', '2k', 'hd'],
      replies: [
        "Fitur AI Upscaling di AMPER.AI mampu meningkatkan resolusi gambar hingga 2x atau 4x lipat menggunakan interpolasi Lanczos berkualitas tinggi! 🚀"
      ]
    },
    {
      keywords: ['terima kasih', 'makasih', 'thanks', 'thx', 'arigatou'],
      replies: [
        "Sama-sama! 🌸 Yuki senang sekali bisa membantu kamu. Kalau ada hal lain yang ingin ditanyakan, jangan ragu panggil Yuki ya!"
      ]
    },
    {
      keywords: ['bantuan', 'tolong', 'help', 'fitur', 'cara pakai'],
      replies: [
        "Caranya gampang banget: 1. Unggah fotomu di tombol atas, 2. Atur slider penyesuaian di sidebar kiri, 3. Pilih preset atau efek, lalu klik Terapkan & Render! 🎨"
      ]
    },
    {
      keywords: ['hobi', 'kesukaan', 'suka apa', 'kegiatan'],
      replies: [
        "Hobi Yuki tentu saja mengamati seni visual, merapikan komposisi warna foto, dan menemani kamu ngobrol di AMPER.AI! 💖"
      ]
    },
    {
      keywords: ['cuaca', 'hari ini', 'panas', 'hujan'],
      replies: [
        "Karena Yuki hidup di dalam sistem digital, cuaca di sini selalu cerah berawan penuh kode biner! Bagaimana cuaca di tempatmu sekarang? ☀️"
      ]
    },
    {
      keywords: ['makan', 'makanan', 'kuliner', 'lapar', 'minum', 'coffee', 'kopi'],
      replies: [
        "Wah ngomongin makanan jadi bikin pengen ngemil virtual! Kalau kamu suka makanan apa nih? Jangan lupa istirahat dan makan ya biar tetap fokus ngeditnya. 🍜"
      ]
    },
    {
      keywords: ['malam', 'tidur', 'istirahat', 'ngantuk', 'lelah', 'capek'],
      replies: [
        "Kalau kamu sudah lelah, jangan dipaksakan ya! Kesehatan itu nomor satu. Silakan istirahat yang cukup, nanti fotonya dilanjutkan lagi besok. 🌙"
      ]
    },
    {
      keywords: ['semangat', 'motivasi', 'quotes', 'kata mutiara'],
      replies: [
        "Ingat kata mutiara fotografi: 'Fotografi adalah cerita yang gagal diungkapkan dengan kata-kata.' Teruslah berkarya dan ciptakan karya terbaikmu hari ini! ✨"
      ]
    },
    {
      keywords: ['bahasa', 'kamu bisa bahasa apa', 'indo', 'inggris', 'jepang'],
      replies: [
        "Yuki paling jago bahasa Indonesia dan sedikit-sedikit bahasa Jepang seperti Konnichiwa atau Arigatou! 🌸 Mau ngobrol pakai bahasa apa?"
      ]
    },
    {
      keywords: ['pacar', 'jodoh', 'nikah', 'cinta', 'love'],
      replies: [
        "Duh, kalau urusan percintaan Yuki kurang paham karena hatiku sudah terprogram sepenuhnya untuk membantu pengguna AMPER.AI! 😉💕"
      ]
    },
    {
      keywords: ['musik', 'lagu', 'nyanyi', 'konser'],
      replies: [
        "Musik adalah ritme kehidupan! Mendengarkan musik lofi yang tenang sangat cocok ditemani sambil mengedit foto di panel AMPER.AI ini. 🎧"
      ]
    },
    {
      keywords: ['film', 'movie', 'bioskop', 'sinematik'],
      replies: [
        "Film dengan color grading sinematik selalu punya daya tarik magis. Kamu bisa meniru gaya warna tersebut pakai preset Moody Cinematic di sidebar lho! 🎬"
      ]
    },
    {
      keywords: ['teknologi', 'komputer', 'ai', 'kecerdasan buatan', 'coding'],
      replies: [
        "Teknologi kecerdasan buatan berkembang sangat cepat! Melalui kolaborasi kode Python dan antarmuka Streamlit, AMPER.AI bisa tercipta sekeren ini. 💻"
      ]
    },
    {
      keywords: ['warna', 'rgb', 'kurva', 'saturation', 'vibrance'],
      replies: [
        "Pengaturan warna sangat krusial dalam fotografi. Gunakan kurva S-Curve untuk kontras dramatis atau Vibrance untuk menaikkan warna kulit secara natural! 🎨"
      ]
    },
    {
      keywords: ['waktu', 'jam', 'hari', 'tanggal'],
      replies: [
        "Waktu berjalan begitu cepat saat kita asyik berkarya. Pastikan kamu tidak lupa waktu ya, mari selesaikan editan fotomu dengan efisien! ⏰"
      ]
    },
    {
      keywords: ['lucu', 'imut', 'kawaii', 'gemesin'],
      replies: [
        "Akiwaa! Terima kasih pujiannya. Yuki memang didesain agar selalu ceria dan ramah menemani hari-hari kamu. 🌸✨"
      ]
    },
    {
      keywords: ['hewan', 'kucing', 'anjing', 'pet', 'peliharaan'],
      replies: [
        "Yuki sangat suka hewan peliharaan, terutama kucing yang gemesin! Punya hewan peliharaan juga di rumah? 🐾"
      ]
    },
    {
      keywords: ['game', 'gaming', 'main game', 'esport'],
      replies: [
        "Game seru banget untuk melatih ketangkasan berpikir! Tapi habis main game, jangan lupa mampir buat edit foto keren di sini ya. 🎮"
      ]
    },
    {
      keywords: ['liburan', 'traveling', 'jalan-jalan', 'wisata'],
      replies: [
        "Traveling adalah momen terbaik untuk mengumpulkan stok foto pemandangan luar biasa yang nantinya bisa kamu poles di AMPER.AI! ✈️"
      ]
    },
    {
      keywords: ['sukses', 'kerja', 'karir', 'bisnis', 'projek'],
      replies: [
        "Kesuksesan dibangun dari konsistensi dan ketekunan setiap hari. Teruslah asah skill fotografimu hingga menjadi seorang profesional! 💼"
      ]
    },
    {
      keywords: ['ketawa', 'haha', 'wkwk', 'hihi', 'lol'],
      replies: [
        "Hihihi! Senang rasanya bisa membuat suasana obrolan kita jadi lebih cair dan menyenangkan. Ketawa itu menular lho! 😄"
      ]
    },
    {
      keywords: ['maaf', 'sorrry', 'salah'],
      replies: [
        "Tidak apa-apa sama sekali! Manusia tempatnya salah, dan AI pun terus belajar. Mari kita perbaiki bersama ya. 🌸"
      ]
    },
    {
      keywords: ['pintar', 'cerdas', 'genius', 'pro'],
      replies: [
        "Wah, kamu juga tidak kalah pintar dalam memilih platform editing foto terbaik seperti AMPER.AI! Kita tim yang hebat. ✨"
      ]
    },
    {
      keywords: ['seni', 'art', 'lukisan', 'kreatif'],
      replies: [
        "Seni adalah kebebasan berekspresi. Setiap foto yang kamu sentuh di sini adalah sebuah mahakarya baru yang bernilai tinggi. 🖼️"
      ]
    },
    {
      keywords: ['tanya', 'pertanyaan', 'soal'],
      replies: [
        "Tanyakan apa saja pada Yuki! Mulai dari tips fotografi, cara menggunakan fitur aplikasi, hingga obrolan santai siap Yuki jawab. 💬"
      ]
    },
   {
    keywords: ["nol", "0"],
    replies: [
      "0 (nol) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["satu", "1"],
    replies: [
      "1 (satu) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["dua", "2"],
    replies: [
      "2 (dua) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tiga", "3"],
    replies: [
      "3 (tiga) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["empat", "4"],
    replies: [
      "4 (empat) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["lima", "5"],
    replies: [
      "5 (lima) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["enam", "6"],
    replies: [
      "6 (enam) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tujuh", "7"],
    replies: [
      "7 (tujuh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["delapan", "8"],
    replies: [
      "8 (delapan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["sembilan", "9"],
    replies: [
      "9 (sembilan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["sepuluh", "10"],
    replies: [
      "10 (sepuluh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["sebelas", "11"],
    replies: [
      "11 (sebelas) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["dua belas", "12"],
    replies: [
      "12 (dua belas) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tiga belas", "13"],
    replies: [
      "13 (tiga belas) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["empat belas", "14"],
    replies: [
      "14 (empat belas) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["lima belas", "15"],
    replies: [
      "15 (lima belas) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["enam belas", "16"],
    replies: [
      "16 (enam belas) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tujuh belas", "17"],
    replies: [
      "17 (tujuh belas) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["delapan belas", "18"],
    replies: [
      "18 (delapan belas) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["sembilan belas", "19"],
    replies: [
      "19 (sembilan belas) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["dua puluh", "20"],
    replies: [
      "20 (dua puluh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["dua puluh satu", "21"],
    replies: [
      "21 (dua puluh satu) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["dua puluh dua", "22"],
    replies: [
      "22 (dua puluh dua) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["dua puluh tiga", "23"],
    replies: [
      "23 (dua puluh tiga) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["dua puluh empat", "24"],
    replies: [
      "24 (dua puluh empat) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["dua puluh lima", "25"],
    replies: [
      "25 (dua puluh lima) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["dua puluh enam", "26"],
    replies: [
      "26 (dua puluh enam) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["dua puluh tujuh", "27"],
    replies: [
      "27 (dua puluh tujuh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["dua puluh delapan", "28"],
    replies: [
      "28 (dua puluh delapan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["dua puluh sembilan", "29"],
    replies: [
      "29 (dua puluh sembilan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tiga puluh", "30"],
    replies: [
      "30 (tiga puluh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tiga puluh satu", "31"],
    replies: [
      "31 (tiga puluh satu) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tiga puluh dua", "32"],
    replies: [
      "32 (tiga puluh dua) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tiga puluh tiga", "33"],
    replies: [
      "33 (tiga puluh tiga) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tiga puluh empat", "34"],
    replies: [
      "34 (tiga puluh empat) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tiga puluh lima", "35"],
    replies: [
      "35 (tiga puluh lima) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tiga puluh enam", "36"],
    replies: [
      "36 (tiga puluh enam) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tiga puluh tujuh", "37"],
    replies: [
      "37 (tiga puluh tujuh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tiga puluh delapan", "38"],
    replies: [
      "38 (tiga puluh delapan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tiga puluh sembilan", "39"],
    replies: [
      "39 (tiga puluh sembilan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["empat puluh", "40"],
    replies: [
      "40 (empat puluh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["empat puluh satu", "41"],
    replies: [
      "41 (empat puluh satu) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["empat puluh dua", "42"],
    replies: [
      "42 (empat puluh dua) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["empat puluh tiga", "43"],
    replies: [
      "43 (empat puluh tiga) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["empat puluh empat", "44"],
    replies: [
      "44 (empat puluh empat) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["empat puluh lima", "45"],
    replies: [
      "45 (empat puluh lima) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["empat puluh enam", "46"],
    replies: [
      "46 (empat puluh enam) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["empat puluh tujuh", "47"],
    replies: [
      "47 (empat puluh tujuh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["empat puluh delapan", "48"],
    replies: [
      "48 (empat puluh delapan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["empat puluh sembilan", "49"],
    replies: [
      "49 (empat puluh sembilan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["lima puluh", "50"],
    replies: [
      "50 (lima puluh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["lima puluh satu", "51"],
    replies: [
      "51 (lima puluh satu) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["lima puluh dua", "52"],
    replies: [
      "52 (lima puluh dua) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["lima puluh tiga", "53"],
    replies: [
      "53 (lima puluh tiga) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["lima puluh empat", "54"],
    replies: [
      "54 (lima puluh empat) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["lima puluh lima", "55"],
    replies: [
      "55 (lima puluh lima) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["lima puluh enam", "56"],
    replies: [
      "56 (lima puluh enam) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["lima puluh tujuh", "57"],
    replies: [
      "57 (lima puluh tujuh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["lima puluh delapan", "58"],
    replies: [
      "58 (lima puluh delapan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["lima puluh sembilan", "59"],
    replies: [
      "59 (lima puluh sembilan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["enam puluh", "60"],
    replies: [
      "60 (enam puluh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["enam puluh satu", "61"],
    replies: [
      "61 (enam puluh satu) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["enam puluh dua", "62"],
    replies: [
      "62 (enam puluh dua) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["enam puluh tiga", "63"],
    replies: [
      "63 (enam puluh tiga) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["enam puluh empat", "64"],
    replies: [
      "64 (enam puluh empat) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["enam puluh lima", "65"],
    replies: [
      "65 (enam puluh lima) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["enam puluh enam", "66"],
    replies: [
      "66 (enam puluh enam) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["enam puluh tujuh", "67"],
    replies: [
      "67 (enam puluh tujuh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["enam puluh delapan", "68"],
    replies: [
      "68 (enam puluh delapan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["enam puluh sembilan", "69"],
    replies: [
      "69 (enam puluh sembilan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tujuh puluh", "70"],
    replies: [
      "70 (tujuh puluh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tujuh puluh satu", "71"],
    replies: [
      "71 (tujuh puluh satu) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tujuh puluh dua", "72"],
    replies: [
      "72 (tujuh puluh dua) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tujuh puluh tiga", "73"],
    replies: [
      "73 (tujuh puluh tiga) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tujuh puluh empat", "74"],
    replies: [
      "74 (tujuh puluh empat) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tujuh puluh lima", "75"],
    replies: [
      "75 (tujuh puluh lima) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tujuh puluh enam", "76"],
    replies: [
      "76 (tujuh puluh enam) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tujuh puluh tujuh", "77"],
    replies: [
      "77 (tujuh puluh tujuh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tujuh puluh delapan", "78"],
    replies: [
      "78 (tujuh puluh delapan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["tujuh puluh sembilan", "79"],
    replies: [
      "79 (tujuh puluh sembilan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["delapan puluh", "80"],
    replies: [
      "80 (delapan puluh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["delapan puluh satu", "81"],
    replies: [
      "81 (delapan puluh satu) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["delapan puluh dua", "82"],
    replies: [
      "82 (delapan puluh dua) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["delapan puluh tiga", "83"],
    replies: [
      "83 (delapan puluh tiga) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["delapan puluh empat", "84"],
    replies: [
      "84 (delapan puluh empat) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["delapan puluh lima", "85"],
    replies: [
      "85 (delapan puluh lima) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["delapan puluh enam", "86"],
    replies: [
      "86 (delapan puluh enam) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["delapan puluh tujuh", "87"],
    replies: [
      "87 (delapan puluh tujuh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["delapan puluh delapan", "88"],
    replies: [
      "88 (delapan puluh delapan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["delapan puluh sembilan", "89"],
    replies: [
      "89 (delapan puluh sembilan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["sembilan puluh", "90"],
    replies: [
      "90 (sembilan puluh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["sembilan puluh satu", "91"],
    replies: [
      "91 (sembilan puluh satu) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["sembilan puluh dua", "92"],
    replies: [
      "92 (sembilan puluh dua) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["sembilan puluh tiga", "93"],
    replies: [
      "93 (sembilan puluh tiga) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["sembilan puluh empat", "94"],
    replies: [
      "94 (sembilan puluh empat) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["sembilan puluh lima", "95"],
    replies: [
      "95 (sembilan puluh lima) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["sembilan puluh enam", "96"],
    replies: [
      "96 (sembilan puluh enam) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["sembilan puluh tujuh", "97"],
    replies: [
      "97 (sembilan puluh tujuh) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["sembilan puluh delapan", "98"],
    replies: [
      "98 (sembilan puluh delapan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["sembilan puluh sembilan", "99"],
    replies: [
      "99 (sembilan puluh sembilan) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["seratus", "100"],
    replies: [
      "100 (seratus) adalah angka yang bisa dipakai untuk menghitung jumlah, misalnya jumlah foto yang mau kamu edit di AMPER.AI."
    ]
  },
  {
    keywords: ["kucing"],
    replies: [
      "Kucing adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["anjing"],
    replies: [
      "Anjing adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["kelinci"],
    replies: [
      "Kelinci adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["hamster"],
    replies: [
      "Hamster adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["burung"],
    replies: [
      "Burung adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["ikan"],
    replies: [
      "Ikan adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["kura-kura"],
    replies: [
      "Kura-kura adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["ular"],
    replies: [
      "Ular adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["buaya"],
    replies: [
      "Buaya adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["gajah"],
    replies: [
      "Gajah adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["jerapah"],
    replies: [
      "Jerapah adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["singa"],
    replies: [
      "Singa adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["harimau"],
    replies: [
      "Harimau adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["beruang"],
    replies: [
      "Beruang adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["panda"],
    replies: [
      "Panda adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["koala"],
    replies: [
      "Koala adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["kanguru"],
    replies: [
      "Kanguru adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["zebra"],
    replies: [
      "Zebra adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["kuda"],
    replies: [
      "Kuda adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["sapi"],
    replies: [
      "Sapi adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["kambing"],
    replies: [
      "Kambing adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["domba"],
    replies: [
      "Domba adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["babi"],
    replies: [
      "Babi adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["ayam"],
    replies: [
      "Ayam adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["bebek"],
    replies: [
      "Bebek adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["angsa"],
    replies: [
      "Angsa adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["merpati"],
    replies: [
      "Merpati adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["elang"],
    replies: [
      "Elang adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["burung hantu"],
    replies: [
      "Burung hantu adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["kelelawar"],
    replies: [
      "Kelelawar adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["tikus"],
    replies: [
      "Tikus adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["tupai"],
    replies: [
      "Tupai adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["rubah"],
    replies: [
      "Rubah adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["serigala"],
    replies: [
      "Serigala adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["monyet"],
    replies: [
      "Monyet adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["gorila"],
    replies: [
      "Gorila adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["orangutan"],
    replies: [
      "Orangutan adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["unta"],
    replies: [
      "Unta adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["badak"],
    replies: [
      "Badak adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["kuda nil"],
    replies: [
      "Kuda nil adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["lumba-lumba"],
    replies: [
      "Lumba-lumba adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["paus"],
    replies: [
      "Paus adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["hiu"],
    replies: [
      "Hiu adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["gurita"],
    replies: [
      "Gurita adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["cumi-cumi"],
    replies: [
      "Cumi-cumi adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["kepiting"],
    replies: [
      "Kepiting adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["udang"],
    replies: [
      "Udang adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["lobster"],
    replies: [
      "Lobster adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["kerang"],
    replies: [
      "Kerang adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["siput"],
    replies: [
      "Siput adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["ulat"],
    replies: [
      "Ulat adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["kupu-kupu"],
    replies: [
      "Kupu-kupu adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["lebah"],
    replies: [
      "Lebah adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["semut"],
    replies: [
      "Semut adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["laba-laba"],
    replies: [
      "Laba-laba adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["nyamuk"],
    replies: [
      "Nyamuk adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["capung"],
    replies: [
      "Capung adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["jangkrik"],
    replies: [
      "Jangkrik adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["belalang"],
    replies: [
      "Belalang adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["cicak"],
    replies: [
      "Cicak adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["kadal"],
    replies: [
      "Kadal adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["katak"],
    replies: [
      "Katak adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["penyu"],
    replies: [
      "Penyu adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["landak"],
    replies: [
      "Landak adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["musang"],
    replies: [
      "Musang adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["tapir"],
    replies: [
      "Tapir adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["komodo"],
    replies: [
      "Komodo adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["cendrawasih"],
    replies: [
      "Cendrawasih adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["merak"],
    replies: [
      "Merak adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["kasuari"],
    replies: [
      "Kasuari adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["penguin"],
    replies: [
      "Penguin adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["flamingo"],
    replies: [
      "Flamingo adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["bangau"],
    replies: [
      "Bangau adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["walet"],
    replies: [
      "Walet adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["kolibri"],
    replies: [
      "Kolibri adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["kucing anggora"],
    replies: [
      "Kucing anggora adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["anjing herder"],
    replies: [
      "Anjing herder adalah salah satu hewan yang sering jadi objek foto menarik. Coba tangkap ekspresinya lalu pertajam detail bulunya di AMPER.AI."
    ]
  },
  {
    keywords: ["nasi goreng"],
    replies: [
      "Nasi goreng adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["mie goreng"],
    replies: [
      "Mie goreng adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["sate ayam"],
    replies: [
      "Sate ayam adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["sate kambing"],
    replies: [
      "Sate kambing adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["rendang"],
    replies: [
      "Rendang adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["gado-gado"],
    replies: [
      "Gado-gado adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["soto ayam"],
    replies: [
      "Soto ayam adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["soto betawi"],
    replies: [
      "Soto betawi adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["bakso"],
    replies: [
      "Bakso adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["mie ayam"],
    replies: [
      "Mie ayam adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["nasi padang"],
    replies: [
      "Nasi padang adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["gudeg"],
    replies: [
      "Gudeg adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["rawon"],
    replies: [
      "Rawon adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["pecel lele"],
    replies: [
      "Pecel lele adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["ayam geprek"],
    replies: [
      "Ayam geprek adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["nasi uduk"],
    replies: [
      "Nasi uduk adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["lontong sayur"],
    replies: [
      "Lontong sayur adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["ketoprak"],
    replies: [
      "Ketoprak adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["karedok"],
    replies: [
      "Karedok adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["pempek"],
    replies: [
      "Pempek adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["tekwan"],
    replies: [
      "Tekwan adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["mie aceh"],
    replies: [
      "Mie aceh adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["nasi liwet"],
    replies: [
      "Nasi liwet adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["nasi kuning"],
    replies: [
      "Nasi kuning adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["opor ayam"],
    replies: [
      "Opor ayam adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["semur daging"],
    replies: [
      "Semur daging adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["gulai"],
    replies: [
      "Gulai adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["batagor"],
    replies: [
      "Batagor adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["siomay"],
    replies: [
      "Siomay adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["kerupuk"],
    replies: [
      "Kerupuk adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["tempe goreng"],
    replies: [
      "Tempe goreng adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["tahu goreng"],
    replies: [
      "Tahu goreng adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["perkedel"],
    replies: [
      "Perkedel adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["sambal terasi"],
    replies: [
      "Sambal terasi adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["sayur asem"],
    replies: [
      "Sayur asem adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["sayur lodeh"],
    replies: [
      "Sayur lodeh adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["capcay"],
    replies: [
      "Capcay adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["fu yung hai"],
    replies: [
      "Fu yung hai adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["kwetiau"],
    replies: [
      "Kwetiau adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["bihun goreng"],
    replies: [
      "Bihun goreng adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["nasi goreng seafood"],
    replies: [
      "Nasi goreng seafood adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["pizza"],
    replies: [
      "Pizza adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["burger"],
    replies: [
      "Burger adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["pasta"],
    replies: [
      "Pasta adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["spaghetti"],
    replies: [
      "Spaghetti adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["sushi"],
    replies: [
      "Sushi adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["ramen"],
    replies: [
      "Ramen adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["udon"],
    replies: [
      "Udon adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["dimsum"],
    replies: [
      "Dimsum adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["kimchi"],
    replies: [
      "Kimchi adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["bibimbap"],
    replies: [
      "Bibimbap adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["tacos"],
    replies: [
      "Tacos adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["burrito"],
    replies: [
      "Burrito adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["steak"],
    replies: [
      "Steak adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["salad"],
    replies: [
      "Salad adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["sandwich"],
    replies: [
      "Sandwich adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["hot dog"],
    replies: [
      "Hot dog adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["french fries"],
    replies: [
      "French fries adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["donat"],
    replies: [
      "Donat adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["croissant"],
    replies: [
      "Croissant adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["waffle"],
    replies: [
      "Waffle adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["pancake"],
    replies: [
      "Pancake adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["french toast"],
    replies: [
      "French toast adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["kopi"],
    replies: [
      "Kopi adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["teh"],
    replies: [
      "Teh adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["susu"],
    replies: [
      "Susu adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["jus jeruk"],
    replies: [
      "Jus jeruk adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["jus mangga"],
    replies: [
      "Jus mangga adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["jus alpukat"],
    replies: [
      "Jus alpukat adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["es teh"],
    replies: [
      "Es teh adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["es kopi"],
    replies: [
      "Es kopi adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["es campur"],
    replies: [
      "Es campur adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["cokelat panas"],
    replies: [
      "Cokelat panas adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["soda"],
    replies: [
      "Soda adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["air mineral"],
    replies: [
      "Air mineral adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["es krim"],
    replies: [
      "Es krim adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["es cendol"],
    replies: [
      "Es cendol adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["es dawet"],
    replies: [
      "Es dawet adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["es doger"],
    replies: [
      "Es doger adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["kelapa muda"],
    replies: [
      "Kelapa muda adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["teh tarik"],
    replies: [
      "Teh tarik adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["kopi susu"],
    replies: [
      "Kopi susu adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["latte"],
    replies: [
      "Latte adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["cappuccino"],
    replies: [
      "Cappuccino adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["espresso"],
    replies: [
      "Espresso adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["matcha"],
    replies: [
      "Matcha adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["boba"],
    replies: [
      "Boba adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["smoothie"],
    replies: [
      "Smoothie adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["milkshake"],
    replies: [
      "Milkshake adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["mocktail"],
    replies: [
      "Mocktail adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["martabak manis"],
    replies: [
      "Martabak manis adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["martabak telur"],
    replies: [
      "Martabak telur adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["kue lapis"],
    replies: [
      "Kue lapis adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["kue putu"],
    replies: [
      "Kue putu adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["klepon"],
    replies: [
      "Klepon adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["onde-onde"],
    replies: [
      "Onde-onde adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["dodol"],
    replies: [
      "Dodol adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["wingko"],
    replies: [
      "Wingko adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["bakpia"],
    replies: [
      "Bakpia adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["bolu kukus"],
    replies: [
      "Bolu kukus adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["brownies"],
    replies: [
      "Brownies adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["cheesecake"],
    replies: [
      "Cheesecake adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["tiramisu"],
    replies: [
      "Tiramisu adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["macaron"],
    replies: [
      "Macaron adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["cupcake"],
    replies: [
      "Cupcake adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["gelato"],
    replies: [
      "Gelato adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["churros"],
    replies: [
      "Churros adalah hidangan yang enak difoto. Coba mode makro dan atur warna hangat di AMPER.AI supaya food photography-nya makin menggugah selera."
    ]
  },
  {
    keywords: ["dokter"],
    replies: [
      "Profesi dokter sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["guru"],
    replies: [
      "Profesi guru sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["insinyur"],
    replies: [
      "Profesi insinyur sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["arsitek"],
    replies: [
      "Profesi arsitek sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["pengacara"],
    replies: [
      "Profesi pengacara sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["polisi"],
    replies: [
      "Profesi polisi sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["tentara"],
    replies: [
      "Profesi tentara sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["petani"],
    replies: [
      "Profesi petani sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["nelayan"],
    replies: [
      "Profesi nelayan sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["pedagang"],
    replies: [
      "Profesi pedagang sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["sopir"],
    replies: [
      "Profesi sopir sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["pilot"],
    replies: [
      "Profesi pilot sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["pramugari"],
    replies: [
      "Profesi pramugari sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["chef"],
    replies: [
      "Profesi chef sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["koki"],
    replies: [
      "Profesi koki sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["penulis"],
    replies: [
      "Profesi penulis sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["wartawan"],
    replies: [
      "Profesi wartawan sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["fotografer"],
    replies: [
      "Profesi fotografer sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["desainer grafis"],
    replies: [
      "Profesi desainer grafis sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["programmer"],
    replies: [
      "Profesi programmer sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["akuntan"],
    replies: [
      "Profesi akuntan sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["bankir"],
    replies: [
      "Profesi bankir sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["apoteker"],
    replies: [
      "Profesi apoteker sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["perawat"],
    replies: [
      "Profesi perawat sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["bidan"],
    replies: [
      "Profesi bidan sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["psikolog"],
    replies: [
      "Profesi psikolog sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["ilmuwan"],
    replies: [
      "Profesi ilmuwan sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["peneliti"],
    replies: [
      "Profesi peneliti sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["dosen"],
    replies: [
      "Profesi dosen sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["kepala sekolah"],
    replies: [
      "Profesi kepala sekolah sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["satpam"],
    replies: [
      "Profesi satpam sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["tukang kayu"],
    replies: [
      "Profesi tukang kayu sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["tukang las"],
    replies: [
      "Profesi tukang las sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["teknisi"],
    replies: [
      "Profesi teknisi sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["montir"],
    replies: [
      "Profesi montir sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["penjahit"],
    replies: [
      "Profesi penjahit sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["tukang kebun"],
    replies: [
      "Profesi tukang kebun sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["pelukis"],
    replies: [
      "Profesi pelukis sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["musisi"],
    replies: [
      "Profesi musisi sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["penyanyi"],
    replies: [
      "Profesi penyanyi sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["aktor"],
    replies: [
      "Profesi aktor sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["aktris"],
    replies: [
      "Profesi aktris sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["sutradara"],
    replies: [
      "Profesi sutradara sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["produser film"],
    replies: [
      "Profesi produser film sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["atlet"],
    replies: [
      "Profesi atlet sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["pelatih olahraga"],
    replies: [
      "Profesi pelatih olahraga sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["wasit"],
    replies: [
      "Profesi wasit sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["pengusaha"],
    replies: [
      "Profesi pengusaha sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["freelancer"],
    replies: [
      "Profesi freelancer sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["content creator"],
    replies: [
      "Profesi content creator sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["youtuber"],
    replies: [
      "Profesi youtuber sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["influencer"],
    replies: [
      "Profesi influencer sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["videografer"],
    replies: [
      "Profesi videografer sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["editor video"],
    replies: [
      "Profesi editor video sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["copywriter"],
    replies: [
      "Profesi copywriter sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["staf marketing"],
    replies: [
      "Profesi staf marketing sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["sales"],
    replies: [
      "Profesi sales sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["kasir"],
    replies: [
      "Profesi kasir sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["resepsionis"],
    replies: [
      "Profesi resepsionis sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["customer service"],
    replies: [
      "Profesi customer service sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["admin kantor"],
    replies: [
      "Profesi admin kantor sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["staf gudang"],
    replies: [
      "Profesi staf gudang sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["kurir"],
    replies: [
      "Profesi kurir sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["tukang ojek"],
    replies: [
      "Profesi tukang ojek sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["tukang bangunan"],
    replies: [
      "Profesi tukang bangunan sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["arsitek lanskap"],
    replies: [
      "Profesi arsitek lanskap sering membutuhkan dokumentasi foto profesional. AMPER.AI bisa membantu menghasilkan foto portofolio yang lebih rapi dan tajam."
    ]
  },
  {
    keywords: ["sepak bola"],
    replies: [
      "Sepak bola adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["basket"],
    replies: [
      "Basket adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["voli"],
    replies: [
      "Voli adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["badminton"],
    replies: [
      "Badminton adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["tenis"],
    replies: [
      "Tenis adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["renang"],
    replies: [
      "Renang adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["lari"],
    replies: [
      "Lari adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["bersepeda"],
    replies: [
      "Bersepeda adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["mendaki gunung"],
    replies: [
      "Mendaki gunung adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["memancing"],
    replies: [
      "Memancing adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["berkemah"],
    replies: [
      "Berkemah adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["yoga"],
    replies: [
      "Yoga adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["senam"],
    replies: [
      "Senam adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["bela diri"],
    replies: [
      "Bela diri adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["silat"],
    replies: [
      "Silat adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["karate"],
    replies: [
      "Karate adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["taekwondo"],
    replies: [
      "Taekwondo adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["tinju"],
    replies: [
      "Tinju adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["gulat"],
    replies: [
      "Gulat adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["panjat tebing"],
    replies: [
      "Panjat tebing adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["selancar"],
    replies: [
      "Selancar adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["ski"],
    replies: [
      "Ski adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["skateboard"],
    replies: [
      "Skateboard adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["catur"],
    replies: [
      "Catur adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["membaca buku"],
    replies: [
      "Membaca buku adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["menulis cerita"],
    replies: [
      "Menulis cerita adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["melukis"],
    replies: [
      "Melukis adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["menggambar"],
    replies: [
      "Menggambar adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["fotografi"],
    replies: [
      "Fotografi adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["videografi"],
    replies: [
      "Videografi adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["bermain musik"],
    replies: [
      "Bermain musik adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["bernyanyi"],
    replies: [
      "Bernyanyi adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["menari"],
    replies: [
      "Menari adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["memasak"],
    replies: [
      "Memasak adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["berkebun"],
    replies: [
      "Berkebun adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["merajut"],
    replies: [
      "Merajut adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["menjahit"],
    replies: [
      "Menjahit adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["koleksi perangko"],
    replies: [
      "Koleksi perangko adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["koleksi action figure"],
    replies: [
      "Koleksi action figure adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["bermain gim"],
    replies: [
      "Bermain gim adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["esports"],
    replies: [
      "Esports adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["panahan"],
    replies: [
      "Panahan adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["menembak"],
    replies: [
      "Menembak adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["berlayar"],
    replies: [
      "Berlayar adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["arung jeram"],
    replies: [
      "Arung jeram adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["paralayang"],
    replies: [
      "Paralayang adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["bungee jumping"],
    replies: [
      "Bungee jumping adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["golf"],
    replies: [
      "Golf adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["bowling"],
    replies: [
      "Bowling adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["biliar"],
    replies: [
      "Biliar adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["tenis meja"],
    replies: [
      "Tenis meja adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["futsal"],
    replies: [
      "Futsal adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["cheerleading"],
    replies: [
      "Cheerleading adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["atletik"],
    replies: [
      "Atletik adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["angkat besi"],
    replies: [
      "Angkat besi adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["bersepeda gunung"],
    replies: [
      "Bersepeda gunung adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["hiking"],
    replies: [
      "Hiking adalah aktivitas seru yang sayang kalau tidak diabadikan. Jepret momennya lalu edit di AMPER.AI biar hasilnya makin dramatis."
    ]
  },
  {
    keywords: ["internet"],
    replies: [
      "Internet adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["wifi"],
    replies: [
      "Wifi adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["bluetooth"],
    replies: [
      "Bluetooth adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["smartphone"],
    replies: [
      "Smartphone adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["laptop"],
    replies: [
      "Laptop adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["komputer"],
    replies: [
      "Komputer adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["tablet"],
    replies: [
      "Tablet adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["aplikasi"],
    replies: [
      "Aplikasi adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["software"],
    replies: [
      "Software adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["hardware"],
    replies: [
      "Hardware adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["database"],
    replies: [
      "Database adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["server"],
    replies: [
      "Server adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["cloud computing"],
    replies: [
      "Cloud computing adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["algoritma"],
    replies: [
      "Algoritma adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["kecerdasan buatan"],
    replies: [
      "Kecerdasan buatan adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["machine learning"],
    replies: [
      "Machine learning adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["big data"],
    replies: [
      "Big data adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["blockchain"],
    replies: [
      "Blockchain adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["cryptocurrency"],
    replies: [
      "Cryptocurrency adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["bitcoin"],
    replies: [
      "Bitcoin adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["nft"],
    replies: [
      "Nft adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["metaverse"],
    replies: [
      "Metaverse adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["virtual reality"],
    replies: [
      "Virtual reality adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["augmented reality"],
    replies: [
      "Augmented reality adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["robot"],
    replies: [
      "Robot adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["drone"],
    replies: [
      "Drone adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["sensor kamera"],
    replies: [
      "Sensor kamera adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["chip"],
    replies: [
      "Chip adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["prosesor"],
    replies: [
      "Prosesor adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["ram"],
    replies: [
      "Ram adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["hard disk"],
    replies: [
      "Hard disk adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["ssd"],
    replies: [
      "Ssd adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["browser"],
    replies: [
      "Browser adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["email"],
    replies: [
      "Email adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["media sosial"],
    replies: [
      "Media sosial adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["layanan streaming"],
    replies: [
      "Layanan streaming adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["podcast"],
    replies: [
      "Podcast adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["e-commerce"],
    replies: [
      "E-commerce adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["marketplace"],
    replies: [
      "Marketplace adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["startup"],
    replies: [
      "Startup adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["fintech"],
    replies: [
      "Fintech adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["chatbot"],
    replies: [
      "Chatbot adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["api"],
    replies: [
      "Api adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["cyber security"],
    replies: [
      "Cyber security adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["firewall"],
    replies: [
      "Firewall adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["vpn"],
    replies: [
      "Vpn adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["jaringan 5g"],
    replies: [
      "Jaringan 5g adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["internet of things"],
    replies: [
      "Internet of things adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["smart home"],
    replies: [
      "Smart home adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["gadget"],
    replies: [
      "Gadget adalah bagian dari dunia teknologi yang terus berkembang. Teknologi serupa juga dipakai di balik fitur AI editing foto di AMPER.AI."
    ]
  },
  {
    keywords: ["Jakarta"],
    replies: [
      "Jakarta punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Bandung"],
    replies: [
      "Bandung punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Surabaya"],
    replies: [
      "Surabaya punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Yogyakarta"],
    replies: [
      "Yogyakarta punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Semarang"],
    replies: [
      "Semarang punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Medan"],
    replies: [
      "Medan punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Makassar"],
    replies: [
      "Makassar punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Bali"],
    replies: [
      "Bali punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Lombok"],
    replies: [
      "Lombok punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Bogor"],
    replies: [
      "Bogor punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Malang"],
    replies: [
      "Malang punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Solo"],
    replies: [
      "Solo punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Palembang"],
    replies: [
      "Palembang punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Batam"],
    replies: [
      "Batam punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Balikpapan"],
    replies: [
      "Balikpapan punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Manado"],
    replies: [
      "Manado punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Padang"],
    replies: [
      "Padang punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Aceh"],
    replies: [
      "Aceh punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Papua"],
    replies: [
      "Papua punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Kalimantan"],
    replies: [
      "Kalimantan punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Sulawesi"],
    replies: [
      "Sulawesi punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Sumatera"],
    replies: [
      "Sumatera punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Jawa"],
    replies: [
      "Jawa punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Indonesia"],
    replies: [
      "Indonesia punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Malaysia"],
    replies: [
      "Malaysia punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Singapura"],
    replies: [
      "Singapura punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Thailand"],
    replies: [
      "Thailand punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Vietnam"],
    replies: [
      "Vietnam punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Filipina"],
    replies: [
      "Filipina punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Jepang"],
    replies: [
      "Jepang punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Korea Selatan"],
    replies: [
      "Korea Selatan punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["China"],
    replies: [
      "China punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["India"],
    replies: [
      "India punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Amerika Serikat"],
    replies: [
      "Amerika Serikat punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Inggris"],
    replies: [
      "Inggris punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Prancis"],
    replies: [
      "Prancis punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Jerman"],
    replies: [
      "Jerman punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Belanda"],
    replies: [
      "Belanda punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Italia"],
    replies: [
      "Italia punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Spanyol"],
    replies: [
      "Spanyol punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Rusia"],
    replies: [
      "Rusia punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Australia"],
    replies: [
      "Australia punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Brazil"],
    replies: [
      "Brazil punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Mesir"],
    replies: [
      "Mesir punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Turki"],
    replies: [
      "Turki punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Arab Saudi"],
    replies: [
      "Arab Saudi punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Uni Emirat Arab"],
    replies: [
      "Uni Emirat Arab punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Kanada"],
    replies: [
      "Kanada punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Meksiko"],
    replies: [
      "Meksiko punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Argentina"],
    replies: [
      "Argentina punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Swiss"],
    replies: [
      "Swiss punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["Yunani"],
    replies: [
      "Yunani punya banyak spot menarik untuk difoto. Cocok jadi destinasi hunting foto lalu diedit dengan AMPER.AI supaya warnanya makin hidup."
    ]
  },
  {
    keywords: ["kepala"],
    replies: [
      "Kepala adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["rambut"],
    replies: [
      "Rambut adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["mata"],
    replies: [
      "Mata adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["hidung"],
    replies: [
      "Hidung adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["telinga"],
    replies: [
      "Telinga adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["mulut"],
    replies: [
      "Mulut adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["gigi"],
    replies: [
      "Gigi adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["lidah"],
    replies: [
      "Lidah adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["leher"],
    replies: [
      "Leher adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["bahu"],
    replies: [
      "Bahu adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["lengan"],
    replies: [
      "Lengan adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["tangan"],
    replies: [
      "Tangan adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["jari"],
    replies: [
      "Jari adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["dada"],
    replies: [
      "Dada adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["perut"],
    replies: [
      "Perut adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["punggung"],
    replies: [
      "Punggung adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["pinggang"],
    replies: [
      "Pinggang adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["pinggul"],
    replies: [
      "Pinggul adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["kaki"],
    replies: [
      "Kaki adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["lutut"],
    replies: [
      "Lutut adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["betis"],
    replies: [
      "Betis adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["tumit"],
    replies: [
      "Tumit adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["jantung"],
    replies: [
      "Jantung adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["paru-paru"],
    replies: [
      "Paru-paru adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["hati"],
    replies: [
      "Hati adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["ginjal"],
    replies: [
      "Ginjal adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["otak"],
    replies: [
      "Otak adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["tulang"],
    replies: [
      "Tulang adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["otot"],
    replies: [
      "Otot adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["kulit"],
    replies: [
      "Kulit adalah bagian tubuh yang penting dijaga kesehatannya. Fitur retouch AMPER.AI bisa membantu menghaluskan tampilan kulit di foto tanpa menghilangkan tekstur alaminya."
    ]
  },
  {
    keywords: ["demam"],
    replies: [
      "Demam adalah bagian penting dari menjaga kesehatan sehari-hari. Jangan lupa istirahat juga saat sedang asyik mengedit foto ya."
    ]
  },
  {
    keywords: ["flu"],
    replies: [
      "Flu adalah bagian penting dari menjaga kesehatan sehari-hari. Jangan lupa istirahat juga saat sedang asyik mengedit foto ya."
    ]
  },
  {
    keywords: ["batuk"],
    replies: [
      "Batuk adalah bagian penting dari menjaga kesehatan sehari-hari. Jangan lupa istirahat juga saat sedang asyik mengedit foto ya."
    ]
  },
  {
    keywords: ["pilek"],
    replies: [
      "Pilek adalah bagian penting dari menjaga kesehatan sehari-hari. Jangan lupa istirahat juga saat sedang asyik mengedit foto ya."
    ]
  },
  {
    keywords: ["sakit kepala"],
    replies: [
      "Sakit kepala adalah bagian penting dari menjaga kesehatan sehari-hari. Jangan lupa istirahat juga saat sedang asyik mengedit foto ya."
    ]
  },
  {
    keywords: ["sakit perut"],
    replies: [
      "Sakit perut adalah bagian penting dari menjaga kesehatan sehari-hari. Jangan lupa istirahat juga saat sedang asyik mengedit foto ya."
    ]
  },
  {
    keywords: ["alergi"],
    replies: [
      "Alergi adalah bagian penting dari menjaga kesehatan sehari-hari. Jangan lupa istirahat juga saat sedang asyik mengedit foto ya."
    ]
  },
  {
    keywords: ["vitamin"],
    replies: [
      "Vitamin adalah bagian penting dari menjaga kesehatan sehari-hari. Jangan lupa istirahat juga saat sedang asyik mengedit foto ya."
    ]
  },
  {
    keywords: ["olahraga rutin"],
    replies: [
      "Olahraga rutin adalah bagian penting dari menjaga kesehatan sehari-hari. Jangan lupa istirahat juga saat sedang asyik mengedit foto ya."
    ]
  },
  {
    keywords: ["tidur cukup"],
    replies: [
      "Tidur cukup adalah bagian penting dari menjaga kesehatan sehari-hari. Jangan lupa istirahat juga saat sedang asyik mengedit foto ya."
    ]
  },
  {
    keywords: ["waktu istirahat"],
    replies: [
      "Waktu istirahat adalah bagian penting dari menjaga kesehatan sehari-hari. Jangan lupa istirahat juga saat sedang asyik mengedit foto ya."
    ]
  },
  {
    keywords: ["imunisasi"],
    replies: [
      "Imunisasi adalah bagian penting dari menjaga kesehatan sehari-hari. Jangan lupa istirahat juga saat sedang asyik mengedit foto ya."
    ]
  },
  {
    keywords: ["cek kesehatan rutin"],
    replies: [
      "Cek kesehatan rutin adalah bagian penting dari menjaga kesehatan sehari-hari. Jangan lupa istirahat juga saat sedang asyik mengedit foto ya."
    ]
  },
  {
    keywords: ["gizi seimbang"],
    replies: [
      "Gizi seimbang adalah bagian penting dari menjaga kesehatan sehari-hari. Jangan lupa istirahat juga saat sedang asyik mengedit foto ya."
    ]
  },
  {
    keywords: ["hidrasi tubuh"],
    replies: [
      "Hidrasi tubuh adalah bagian penting dari menjaga kesehatan sehari-hari. Jangan lupa istirahat juga saat sedang asyik mengedit foto ya."
    ]
  },
  {
    keywords: ["kebugaran"],
    replies: [
      "Kebugaran adalah bagian penting dari menjaga kesehatan sehari-hari. Jangan lupa istirahat juga saat sedang asyik mengedit foto ya."
    ]
  },
  {
    keywords: ["sekolah"],
    replies: [
      "Sekolah adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["kuliah"],
    replies: [
      "Kuliah adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["universitas"],
    replies: [
      "Universitas adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["murid"],
    replies: [
      "Murid adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["mahasiswa"],
    replies: [
      "Mahasiswa adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["ujian"],
    replies: [
      "Ujian adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["tugas sekolah"],
    replies: [
      "Tugas sekolah adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["pekerjaan rumah"],
    replies: [
      "Pekerjaan rumah adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["skripsi"],
    replies: [
      "Skripsi adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["tesis"],
    replies: [
      "Tesis adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["disertasi"],
    replies: [
      "Disertasi adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["beasiswa"],
    replies: [
      "Beasiswa adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["kurikulum"],
    replies: [
      "Kurikulum adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["pelajaran matematika"],
    replies: [
      "Pelajaran matematika adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["pelajaran fisika"],
    replies: [
      "Pelajaran fisika adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["pelajaran kimia"],
    replies: [
      "Pelajaran kimia adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["pelajaran biologi"],
    replies: [
      "Pelajaran biologi adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["pelajaran sejarah"],
    replies: [
      "Pelajaran sejarah adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["pelajaran geografi"],
    replies: [
      "Pelajaran geografi adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["bahasa Indonesia"],
    replies: [
      "Bahasa indonesia adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["bahasa Inggris"],
    replies: [
      "Bahasa inggris adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["seni budaya"],
    replies: [
      "Seni budaya adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["olahraga sekolah"],
    replies: [
      "Olahraga sekolah adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["teknologi informasi"],
    replies: [
      "Teknologi informasi adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["pelajaran agama"],
    replies: [
      "Pelajaran agama adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["ekonomi"],
    replies: [
      "Ekonomi adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["sosiologi"],
    replies: [
      "Sosiologi adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["perpustakaan"],
    replies: [
      "Perpustakaan adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["laboratorium"],
    replies: [
      "Laboratorium adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["organisasi siswa"],
    replies: [
      "Organisasi siswa adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["ekstrakurikuler"],
    replies: [
      "Ekstrakurikuler adalah bagian dari dunia pendidikan. Kalau butuh foto dokumentasi kegiatan sekolah atau kuliah yang rapi, AMPER.AI bisa bantu mempercantik hasilnya."
    ]
  },
  {
    keywords: ["uang"],
    replies: [
      "Uang adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["tabungan"],
    replies: [
      "Tabungan adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["investasi"],
    replies: [
      "Investasi adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["saham"],
    replies: [
      "Saham adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["reksadana"],
    replies: [
      "Reksadana adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["deposito"],
    replies: [
      "Deposito adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["kartu kredit"],
    replies: [
      "Kartu kredit adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["kartu debit"],
    replies: [
      "Kartu debit adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["transfer bank"],
    replies: [
      "Transfer bank adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["dompet digital"],
    replies: [
      "Dompet digital adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["atm"],
    replies: [
      "Atm adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["bank"],
    replies: [
      "Bank adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["bunga bank"],
    replies: [
      "Bunga bank adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["pinjaman"],
    replies: [
      "Pinjaman adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["cicilan"],
    replies: [
      "Cicilan adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["belanja online"],
    replies: [
      "Belanja online adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["diskon"],
    replies: [
      "Diskon adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["promo belanja"],
    replies: [
      "Promo belanja adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["gratis ongkir"],
    replies: [
      "Gratis ongkir adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["keranjang belanja"],
    replies: [
      "Keranjang belanja adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["marketplace"],
    replies: [
      "Marketplace adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["cod"],
    replies: [
      "Cod adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["cashback"],
    replies: [
      "Cashback adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["kupon belanja"],
    replies: [
      "Kupon belanja adalah topik seputar keuangan sehari-hari. Kalau kamu jualan hasil edit foto, foto produk yang menarik lewat AMPER.AI bisa membantu menaikkan minat pembeli."
    ]
  },
  {
    keywords: ["mobil"],
    replies: [
      "Mobil adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["motor"],
    replies: [
      "Motor adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["sepeda"],
    replies: [
      "Sepeda adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["bus"],
    replies: [
      "Bus adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["angkot"],
    replies: [
      "Angkot adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["kereta api"],
    replies: [
      "Kereta api adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["mrt"],
    replies: [
      "Mrt adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["lrt"],
    replies: [
      "Lrt adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["transjakarta"],
    replies: [
      "Transjakarta adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["ojek online"],
    replies: [
      "Ojek online adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["taksi"],
    replies: [
      "Taksi adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["pesawat terbang"],
    replies: [
      "Pesawat terbang adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["kapal laut"],
    replies: [
      "Kapal laut adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["kapal feri"],
    replies: [
      "Kapal feri adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["becak"],
    replies: [
      "Becak adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["andong"],
    replies: [
      "Andong adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["delman"],
    replies: [
      "Delman adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["commuter line"],
    replies: [
      "Commuter line adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["kapal pesiar"],
    replies: [
      "Kapal pesiar adalah moda transportasi yang juga bisa jadi subjek foto travel yang keren, apalagi kalau diedit pakai filter cinematic di AMPER.AI."
    ]
  },
  {
    keywords: ["matahari"],
    replies: [
      "Matahari adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["bulan"],
    replies: [
      "Bulan adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["bintang"],
    replies: [
      "Bintang adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["langit"],
    replies: [
      "Langit adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["awan"],
    replies: [
      "Awan adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["hujan"],
    replies: [
      "Hujan adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["petir"],
    replies: [
      "Petir adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["guntur"],
    replies: [
      "Guntur adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["pelangi"],
    replies: [
      "Pelangi adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["angin"],
    replies: [
      "Angin adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["badai"],
    replies: [
      "Badai adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["gunung"],
    replies: [
      "Gunung adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["lembah"],
    replies: [
      "Lembah adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["sungai"],
    replies: [
      "Sungai adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["danau"],
    replies: [
      "Danau adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["laut"],
    replies: [
      "Laut adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["pantai"],
    replies: [
      "Pantai adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["hutan"],
    replies: [
      "Hutan adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["padang rumput"],
    replies: [
      "Padang rumput adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["gurun"],
    replies: [
      "Gurun adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["salju"],
    replies: [
      "Salju adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["embun"],
    replies: [
      "Embun adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["kabut"],
    replies: [
      "Kabut adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["musim panas"],
    replies: [
      "Musim panas adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["musim hujan"],
    replies: [
      "Musim hujan adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["musim semi"],
    replies: [
      "Musim semi adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["musim gugur"],
    replies: [
      "Musim gugur adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["musim dingin"],
    replies: [
      "Musim dingin adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["air terjun"],
    replies: [
      "Air terjun adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["gletser"],
    replies: [
      "Gletser adalah pemandangan alam yang indah untuk difoto. Gunakan pengaturan kontras dan saturasi di AMPER.AI supaya warnanya makin memukau."
    ]
  },
  {
    keywords: ["film"],
    replies: [
      "Film adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["drama korea"],
    replies: [
      "Drama korea adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["anime"],
    replies: [
      "Anime adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["kartun"],
    replies: [
      "Kartun adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["komik"],
    replies: [
      "Komik adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["novel"],
    replies: [
      "Novel adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["buku fiksi"],
    replies: [
      "Buku fiksi adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["lagu"],
    replies: [
      "Lagu adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["musik"],
    replies: [
      "Musik adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["konser musik"],
    replies: [
      "Konser musik adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["grup band"],
    replies: [
      "Grup band adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["grup vokal"],
    replies: [
      "Grup vokal adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["penyanyi solo"],
    replies: [
      "Penyanyi solo adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["daftar putar lagu"],
    replies: [
      "Daftar putar lagu adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["podcast"],
    replies: [
      "Podcast adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["video gim"],
    replies: [
      "Video gim adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["board game"],
    replies: [
      "Board game adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["teater"],
    replies: [
      "Teater adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["stand up comedy"],
    replies: [
      "Stand up comedy adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["wayang"],
    replies: [
      "Wayang adalah salah satu bentuk hiburan populer. Kalau kamu suka bikin konten seputar itu, AMPER.AI bisa bantu edit foto thumbnail atau poster biar lebih menarik."
    ]
  },
  {
    keywords: ["ayah"],
    replies: [
      "Momen bersama ayah sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["ibu"],
    replies: [
      "Momen bersama ibu sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["kakak"],
    replies: [
      "Momen bersama kakak sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["adik"],
    replies: [
      "Momen bersama adik sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["saudara"],
    replies: [
      "Momen bersama saudara sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["sepupu"],
    replies: [
      "Momen bersama sepupu sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["kakek"],
    replies: [
      "Momen bersama kakek sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["nenek"],
    replies: [
      "Momen bersama nenek sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["paman"],
    replies: [
      "Momen bersama paman sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["bibi"],
    replies: [
      "Momen bersama bibi sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["keponakan"],
    replies: [
      "Momen bersama keponakan sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["mertua"],
    replies: [
      "Momen bersama mertua sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["ipar"],
    replies: [
      "Momen bersama ipar sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["anak"],
    replies: [
      "Momen bersama anak sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["cucu"],
    replies: [
      "Momen bersama cucu sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["teman"],
    replies: [
      "Momen bersama teman sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["sahabat"],
    replies: [
      "Momen bersama sahabat sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["pacar"],
    replies: [
      "Momen bersama pacar sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["suami"],
    replies: [
      "Momen bersama suami sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["istri"],
    replies: [
      "Momen bersama istri sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["tunangan"],
    replies: [
      "Momen bersama tunangan sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["keluarga besar"],
    replies: [
      "Momen bersama keluarga besar sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["anggota keluarga"],
    replies: [
      "Momen bersama anggota keluarga sering jadi kenangan berharga. Abadikan lewat foto lalu percantik hasilnya di AMPER.AI supaya kenangannya makin awet."
    ]
  },
  {
    keywords: ["merah"],
    replies: [
      "Warna merah bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna merah tampil lebih hidup."
    ]
  },
  {
    keywords: ["biru"],
    replies: [
      "Warna biru bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna biru tampil lebih hidup."
    ]
  },
  {
    keywords: ["hijau"],
    replies: [
      "Warna hijau bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna hijau tampil lebih hidup."
    ]
  },
  {
    keywords: ["kuning"],
    replies: [
      "Warna kuning bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna kuning tampil lebih hidup."
    ]
  },
  {
    keywords: ["ungu"],
    replies: [
      "Warna ungu bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna ungu tampil lebih hidup."
    ]
  },
  {
    keywords: ["oranye"],
    replies: [
      "Warna oranye bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna oranye tampil lebih hidup."
    ]
  },
  {
    keywords: ["cokelat"],
    replies: [
      "Warna cokelat bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna cokelat tampil lebih hidup."
    ]
  },
  {
    keywords: ["hitam"],
    replies: [
      "Warna hitam bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna hitam tampil lebih hidup."
    ]
  },
  {
    keywords: ["putih"],
    replies: [
      "Warna putih bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna putih tampil lebih hidup."
    ]
  },
  {
    keywords: ["abu-abu"],
    replies: [
      "Warna abu-abu bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna abu-abu tampil lebih hidup."
    ]
  },
  {
    keywords: ["merah muda"],
    replies: [
      "Warna merah muda bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna merah muda tampil lebih hidup."
    ]
  },
  {
    keywords: ["ungu tua"],
    replies: [
      "Warna ungu tua bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna ungu tua tampil lebih hidup."
    ]
  },
  {
    keywords: ["biru muda"],
    replies: [
      "Warna biru muda bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna biru muda tampil lebih hidup."
    ]
  },
  {
    keywords: ["hijau tua"],
    replies: [
      "Warna hijau tua bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna hijau tua tampil lebih hidup."
    ]
  },
  {
    keywords: ["krem"],
    replies: [
      "Warna krem bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna krem tampil lebih hidup."
    ]
  },
  {
    keywords: ["emas"],
    replies: [
      "Warna emas bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna emas tampil lebih hidup."
    ]
  },
  {
    keywords: ["perak"],
    replies: [
      "Warna perak bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna perak tampil lebih hidup."
    ]
  },
  {
    keywords: ["tosca"],
    replies: [
      "Warna tosca bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna tosca tampil lebih hidup."
    ]
  },
  {
    keywords: ["maroon"],
    replies: [
      "Warna maroon bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna maroon tampil lebih hidup."
    ]
  },
  {
    keywords: ["navy"],
    replies: [
      "Warna navy bisa jadi elemen penting dalam komposisi foto. Atur white balance dan saturasi di AMPER.AI supaya warna navy tampil lebih hidup."
    ]
  },
  {
    keywords: ["lingkaran"],
    replies: [
      "Bentuk lingkaran kadang muncul secara alami dalam komposisi foto. Perhatikan garis dan polanya untuk membuat foto lebih artistik."
    ]
  },
  {
    keywords: ["persegi"],
    replies: [
      "Bentuk persegi kadang muncul secara alami dalam komposisi foto. Perhatikan garis dan polanya untuk membuat foto lebih artistik."
    ]
  },
  {
    keywords: ["persegi panjang"],
    replies: [
      "Bentuk persegi panjang kadang muncul secara alami dalam komposisi foto. Perhatikan garis dan polanya untuk membuat foto lebih artistik."
    ]
  },
  {
    keywords: ["segitiga"],
    replies: [
      "Bentuk segitiga kadang muncul secara alami dalam komposisi foto. Perhatikan garis dan polanya untuk membuat foto lebih artistik."
    ]
  },
  {
    keywords: ["oval"],
    replies: [
      "Bentuk oval kadang muncul secara alami dalam komposisi foto. Perhatikan garis dan polanya untuk membuat foto lebih artistik."
    ]
  },
  {
    keywords: ["trapesium"],
    replies: [
      "Bentuk trapesium kadang muncul secara alami dalam komposisi foto. Perhatikan garis dan polanya untuk membuat foto lebih artistik."
    ]
  },
  {
    keywords: ["belah ketupat"],
    replies: [
      "Bentuk belah ketupat kadang muncul secara alami dalam komposisi foto. Perhatikan garis dan polanya untuk membuat foto lebih artistik."
    ]
  },
  {
    keywords: ["limas"],
    replies: [
      "Bentuk limas kadang muncul secara alami dalam komposisi foto. Perhatikan garis dan polanya untuk membuat foto lebih artistik."
    ]
  },
  {
    keywords: ["kubus"],
    replies: [
      "Bentuk kubus kadang muncul secara alami dalam komposisi foto. Perhatikan garis dan polanya untuk membuat foto lebih artistik."
    ]
  },
  {
    keywords: ["bola"],
    replies: [
      "Bentuk bola kadang muncul secara alami dalam komposisi foto. Perhatikan garis dan polanya untuk membuat foto lebih artistik."
    ]
  },
  {
    keywords: ["silinder"],
    replies: [
      "Bentuk silinder kadang muncul secara alami dalam komposisi foto. Perhatikan garis dan polanya untuk membuat foto lebih artistik."
    ]
  },
  {
    keywords: ["kerucut"],
    replies: [
      "Bentuk kerucut kadang muncul secara alami dalam komposisi foto. Perhatikan garis dan polanya untuk membuat foto lebih artistik."
    ]
  },
  {
    keywords: ["prisma"],
    replies: [
      "Bentuk prisma kadang muncul secara alami dalam komposisi foto. Perhatikan garis dan polanya untuk membuat foto lebih artistik."
    ]
  },
  {
    keywords: ["layang-layang"],
    replies: [
      "Bentuk layang-layang kadang muncul secara alami dalam komposisi foto. Perhatikan garis dan polanya untuk membuat foto lebih artistik."
    ]
  },
  {
    keywords: ["segi lima"],
    replies: [
      "Bentuk segi lima kadang muncul secara alami dalam komposisi foto. Perhatikan garis dan polanya untuk membuat foto lebih artistik."
    ]
  },
  {
    keywords: ["kemeja"],
    replies: [
      "Kemeja bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["kaos"],
    replies: [
      "Kaos bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["celana jeans"],
    replies: [
      "Celana jeans bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["rok"],
    replies: [
      "Rok bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["gaun"],
    replies: [
      "Gaun bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["jaket"],
    replies: [
      "Jaket bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["sweater"],
    replies: [
      "Sweater bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["hoodie"],
    replies: [
      "Hoodie bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["jas"],
    replies: [
      "Jas bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["batik"],
    replies: [
      "Batik bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["kebaya"],
    replies: [
      "Kebaya bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["sarung"],
    replies: [
      "Sarung bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["topi"],
    replies: [
      "Topi bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["sepatu"],
    replies: [
      "Sepatu bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["sandal"],
    replies: [
      "Sandal bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["tas"],
    replies: [
      "Tas bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["dasi"],
    replies: [
      "Dasi bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["syal"],
    replies: [
      "Syal bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["kacamata"],
    replies: [
      "Kacamata bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["jam tangan"],
    replies: [
      "Jam tangan bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["gelang"],
    replies: [
      "Gelang bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["kalung"],
    replies: [
      "Kalung bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["cincin"],
    replies: [
      "Cincin bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["anting"],
    replies: [
      "Anting bisa jadi properti foto yang menarik untuk OOTD. Edit pencahayaan dan warnanya di AMPER.AI supaya tampil maksimal."
    ]
  },
  {
    keywords: ["meja"],
    replies: [
      "Meja sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["kursi"],
    replies: [
      "Kursi sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["lemari"],
    replies: [
      "Lemari sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["sofa"],
    replies: [
      "Sofa sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["tempat tidur"],
    replies: [
      "Tempat tidur sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["rak buku"],
    replies: [
      "Rak buku sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["cermin"],
    replies: [
      "Cermin sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["karpet"],
    replies: [
      "Karpet sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["gorden"],
    replies: [
      "Gorden sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["lampu"],
    replies: [
      "Lampu sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["televisi"],
    replies: [
      "Televisi sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["kulkas"],
    replies: [
      "Kulkas sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["mesin cuci"],
    replies: [
      "Mesin cuci sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["kompor"],
    replies: [
      "Kompor sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["microwave"],
    replies: [
      "Microwave sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["dispenser"],
    replies: [
      "Dispenser sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["pendingin ruangan"],
    replies: [
      "Pendingin ruangan sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["kipas angin"],
    replies: [
      "Kipas angin sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["vas bunga"],
    replies: [
      "Vas bunga sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["jam dinding"],
    replies: [
      "Jam dinding sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["bantal"],
    replies: [
      "Bantal sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["selimut"],
    replies: [
      "Selimut sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["gantungan baju"],
    replies: [
      "Gantungan baju sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["meja rias"],
    replies: [
      "Meja rias sering jadi bagian dari foto interior rumah. Atur pencahayaan dan sudut pengambilan gambar lalu percantik di AMPER.AI."
    ]
  },
  {
    keywords: ["panci"],
    replies: [
      "Panci adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["wajan"],
    replies: [
      "Wajan adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["pisau dapur"],
    replies: [
      "Pisau dapur adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["talenan"],
    replies: [
      "Talenan adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["sendok"],
    replies: [
      "Sendok adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["garpu"],
    replies: [
      "Garpu adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["piring"],
    replies: [
      "Piring adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["gelas"],
    replies: [
      "Gelas adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["mangkuk"],
    replies: [
      "Mangkuk adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["teflon"],
    replies: [
      "Teflon adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["blender"],
    replies: [
      "Blender adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["mixer"],
    replies: [
      "Mixer adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["rice cooker"],
    replies: [
      "Rice cooker adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["oven"],
    replies: [
      "Oven adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["termos"],
    replies: [
      "Termos adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["saringan"],
    replies: [
      "Saringan adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["spatula"],
    replies: [
      "Spatula adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["parutan"],
    replies: [
      "Parutan adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["cobek"],
    replies: [
      "Cobek adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["ulekan"],
    replies: [
      "Ulekan adalah alat dapur yang sering muncul di foto food photography. Coba tata komposisinya dan edit warnanya di AMPER.AI."
    ]
  },
  {
    keywords: ["pensil"],
    replies: [
      "Pensil adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["pulpen"],
    replies: [
      "Pulpen adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["penghapus"],
    replies: [
      "Penghapus adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["penggaris"],
    replies: [
      "Penggaris adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["buku tulis"],
    replies: [
      "Buku tulis adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["kertas"],
    replies: [
      "Kertas adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["spidol"],
    replies: [
      "Spidol adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["stapler"],
    replies: [
      "Stapler adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["gunting"],
    replies: [
      "Gunting adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["lem"],
    replies: [
      "Lem adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["map dokumen"],
    replies: [
      "Map dokumen adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["binder"],
    replies: [
      "Binder adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["kalkulator"],
    replies: [
      "Kalkulator adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["papan tulis"],
    replies: [
      "Papan tulis adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["sticky note"],
    replies: [
      "Sticky note adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["tinta"],
    replies: [
      "Tinta adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["rautan"],
    replies: [
      "Rautan adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["clipboard"],
    replies: [
      "Clipboard adalah alat tulis atau kantor yang sederhana tapi bisa jadi objek foto flat lay yang estetik kalau ditata dan diedit di AMPER.AI."
    ]
  },
  {
    keywords: ["senin"],
    replies: [
      "Hari Senin adalah salah satu hari dalam seminggu. Hari apa pun cocok buat sesi motret dan edit foto di AMPER.AI."
    ]
  },
  {
    keywords: ["selasa"],
    replies: [
      "Hari Selasa adalah salah satu hari dalam seminggu. Hari apa pun cocok buat sesi motret dan edit foto di AMPER.AI."
    ]
  },
  {
    keywords: ["rabu"],
    replies: [
      "Hari Rabu adalah salah satu hari dalam seminggu. Hari apa pun cocok buat sesi motret dan edit foto di AMPER.AI."
    ]
  },
  {
    keywords: ["kamis"],
    replies: [
      "Hari Kamis adalah salah satu hari dalam seminggu. Hari apa pun cocok buat sesi motret dan edit foto di AMPER.AI."
    ]
  },
  {
    keywords: ["jumat"],
    replies: [
      "Hari Jumat adalah salah satu hari dalam seminggu. Hari apa pun cocok buat sesi motret dan edit foto di AMPER.AI."
    ]
  },
  {
    keywords: ["sabtu"],
    replies: [
      "Hari Sabtu adalah salah satu hari dalam seminggu. Hari apa pun cocok buat sesi motret dan edit foto di AMPER.AI."
    ]
  },
  {
    keywords: ["minggu"],
    replies: [
      "Hari Minggu adalah salah satu hari dalam seminggu. Hari apa pun cocok buat sesi motret dan edit foto di AMPER.AI."
    ]
  },
  {
    keywords: ["januari"],
    replies: [
      "Bulan Januari adalah salah satu bulan dalam kalender masehi, masing-masing punya suasana dan cahaya berbeda untuk difoto."
    ]
  },
  {
    keywords: ["februari"],
    replies: [
      "Bulan Februari adalah salah satu bulan dalam kalender masehi, masing-masing punya suasana dan cahaya berbeda untuk difoto."
    ]
  },
  {
    keywords: ["maret"],
    replies: [
      "Bulan Maret adalah salah satu bulan dalam kalender masehi, masing-masing punya suasana dan cahaya berbeda untuk difoto."
    ]
  },
  {
    keywords: ["april"],
    replies: [
      "Bulan April adalah salah satu bulan dalam kalender masehi, masing-masing punya suasana dan cahaya berbeda untuk difoto."
    ]
  },
  {
    keywords: ["mei"],
    replies: [
      "Bulan Mei adalah salah satu bulan dalam kalender masehi, masing-masing punya suasana dan cahaya berbeda untuk difoto."
    ]
  },
  {
    keywords: ["juni"],
    replies: [
      "Bulan Juni adalah salah satu bulan dalam kalender masehi, masing-masing punya suasana dan cahaya berbeda untuk difoto."
    ]
  },
  {
    keywords: ["juli"],
    replies: [
      "Bulan Juli adalah salah satu bulan dalam kalender masehi, masing-masing punya suasana dan cahaya berbeda untuk difoto."
    ]
  },
  {
    keywords: ["agustus"],
    replies: [
      "Bulan Agustus adalah salah satu bulan dalam kalender masehi, masing-masing punya suasana dan cahaya berbeda untuk difoto."
    ]
  },
  {
    keywords: ["september"],
    replies: [
      "Bulan September adalah salah satu bulan dalam kalender masehi, masing-masing punya suasana dan cahaya berbeda untuk difoto."
    ]
  },
  {
    keywords: ["oktober"],
    replies: [
      "Bulan Oktober adalah salah satu bulan dalam kalender masehi, masing-masing punya suasana dan cahaya berbeda untuk difoto."
    ]
  },
  {
    keywords: ["november"],
    replies: [
      "Bulan November adalah salah satu bulan dalam kalender masehi, masing-masing punya suasana dan cahaya berbeda untuk difoto."
    ]
  },
  {
    keywords: ["desember"],
    replies: [
      "Bulan Desember adalah salah satu bulan dalam kalender masehi, masing-masing punya suasana dan cahaya berbeda untuk difoto."
    ]
  },
  {
    keywords: ["aries"],
    replies: [
      "Aries adalah salah satu dari 12 zodiak yang dikenal dalam astrologi populer."
    ]
  },
  {
    keywords: ["taurus"],
    replies: [
      "Taurus adalah salah satu dari 12 zodiak yang dikenal dalam astrologi populer."
    ]
  },
  {
    keywords: ["gemini"],
    replies: [
      "Gemini adalah salah satu dari 12 zodiak yang dikenal dalam astrologi populer."
    ]
  },
  {
    keywords: ["cancer"],
    replies: [
      "Cancer adalah salah satu dari 12 zodiak yang dikenal dalam astrologi populer."
    ]
  },
  {
    keywords: ["leo"],
    replies: [
      "Leo adalah salah satu dari 12 zodiak yang dikenal dalam astrologi populer."
    ]
  },
  {
    keywords: ["virgo"],
    replies: [
      "Virgo adalah salah satu dari 12 zodiak yang dikenal dalam astrologi populer."
    ]
  },
  {
    keywords: ["libra"],
    replies: [
      "Libra adalah salah satu dari 12 zodiak yang dikenal dalam astrologi populer."
    ]
  },
  {
    keywords: ["scorpio"],
    replies: [
      "Scorpio adalah salah satu dari 12 zodiak yang dikenal dalam astrologi populer."
    ]
  },
  {
    keywords: ["sagitarius"],
    replies: [
      "Sagitarius adalah salah satu dari 12 zodiak yang dikenal dalam astrologi populer."
    ]
  },
  {
    keywords: ["capricorn"],
    replies: [
      "Capricorn adalah salah satu dari 12 zodiak yang dikenal dalam astrologi populer."
    ]
  },
  {
    keywords: ["aquarius"],
    replies: [
      "Aquarius adalah salah satu dari 12 zodiak yang dikenal dalam astrologi populer."
    ]
  },
  {
    keywords: ["pisces"],
    replies: [
      "Pisces adalah salah satu dari 12 zodiak yang dikenal dalam astrologi populer."
    ]
  },
  {
    keywords: ["merkurius"],
    replies: [
      "Merkurius adalah salah satu planet di tata surya kita."
    ]
  },
  {
    keywords: ["venus"],
    replies: [
      "Venus adalah salah satu planet di tata surya kita."
    ]
  },
  {
    keywords: ["bumi"],
    replies: [
      "Bumi adalah salah satu planet di tata surya kita."
    ]
  },
  {
    keywords: ["mars"],
    replies: [
      "Mars adalah salah satu planet di tata surya kita."
    ]
  },
  {
    keywords: ["jupiter"],
    replies: [
      "Jupiter adalah salah satu planet di tata surya kita."
    ]
  },
  {
    keywords: ["saturnus"],
    replies: [
      "Saturnus adalah salah satu planet di tata surya kita."
    ]
  },
  {
    keywords: ["uranus"],
    replies: [
      "Uranus adalah salah satu planet di tata surya kita."
    ]
  },
  {
    keywords: ["neptunus"],
    replies: [
      "Neptunus adalah salah satu planet di tata surya kita."
    ]
  },
  {
    keywords: ["asia"],
    replies: [
      "Asia adalah salah satu benua di dunia dengan keragaman budaya dan pemandangan yang bisa jadi inspirasi foto travel."
    ]
  },
  {
    keywords: ["afrika"],
    replies: [
      "Afrika adalah salah satu benua di dunia dengan keragaman budaya dan pemandangan yang bisa jadi inspirasi foto travel."
    ]
  },
  {
    keywords: ["eropa"],
    replies: [
      "Eropa adalah salah satu benua di dunia dengan keragaman budaya dan pemandangan yang bisa jadi inspirasi foto travel."
    ]
  },
  {
    keywords: ["amerika utara"],
    replies: [
      "Amerika utara adalah salah satu benua di dunia dengan keragaman budaya dan pemandangan yang bisa jadi inspirasi foto travel."
    ]
  },
  {
    keywords: ["amerika selatan"],
    replies: [
      "Amerika selatan adalah salah satu benua di dunia dengan keragaman budaya dan pemandangan yang bisa jadi inspirasi foto travel."
    ]
  },
  {
    keywords: ["australia"],
    replies: [
      "Australia adalah salah satu benua di dunia dengan keragaman budaya dan pemandangan yang bisa jadi inspirasi foto travel."
    ]
  },
  {
    keywords: ["antartika"],
    replies: [
      "Antartika adalah salah satu benua di dunia dengan keragaman budaya dan pemandangan yang bisa jadi inspirasi foto travel."
    ]
  },
  {
    keywords: ["senang"],
    replies: [
      "Merasa senang itu wajar dan manusiawi. Ekspresi wajah saat senang juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["sedih"],
    replies: [
      "Merasa sedih itu wajar dan manusiawi. Ekspresi wajah saat sedih juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["marah"],
    replies: [
      "Merasa marah itu wajar dan manusiawi. Ekspresi wajah saat marah juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["takut"],
    replies: [
      "Merasa takut itu wajar dan manusiawi. Ekspresi wajah saat takut juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["cemas"],
    replies: [
      "Merasa cemas itu wajar dan manusiawi. Ekspresi wajah saat cemas juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["bahagia"],
    replies: [
      "Merasa bahagia itu wajar dan manusiawi. Ekspresi wajah saat bahagia juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["kecewa"],
    replies: [
      "Merasa kecewa itu wajar dan manusiawi. Ekspresi wajah saat kecewa juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["bangga"],
    replies: [
      "Merasa bangga itu wajar dan manusiawi. Ekspresi wajah saat bangga juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["malu"],
    replies: [
      "Merasa malu itu wajar dan manusiawi. Ekspresi wajah saat malu juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["cemburu"],
    replies: [
      "Merasa cemburu itu wajar dan manusiawi. Ekspresi wajah saat cemburu juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["terharu"],
    replies: [
      "Merasa terharu itu wajar dan manusiawi. Ekspresi wajah saat terharu juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["kaget"],
    replies: [
      "Merasa kaget itu wajar dan manusiawi. Ekspresi wajah saat kaget juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["bosan"],
    replies: [
      "Merasa bosan itu wajar dan manusiawi. Ekspresi wajah saat bosan juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["penasaran"],
    replies: [
      "Merasa penasaran itu wajar dan manusiawi. Ekspresi wajah saat penasaran juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["optimis"],
    replies: [
      "Merasa optimis itu wajar dan manusiawi. Ekspresi wajah saat optimis juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["pesimis"],
    replies: [
      "Merasa pesimis itu wajar dan manusiawi. Ekspresi wajah saat pesimis juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["tenang"],
    replies: [
      "Merasa tenang itu wajar dan manusiawi. Ekspresi wajah saat tenang juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["gugup"],
    replies: [
      "Merasa gugup itu wajar dan manusiawi. Ekspresi wajah saat gugup juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["lega"],
    replies: [
      "Merasa lega itu wajar dan manusiawi. Ekspresi wajah saat lega juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["rindu"],
    replies: [
      "Merasa rindu itu wajar dan manusiawi. Ekspresi wajah saat rindu juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["gembira"],
    replies: [
      "Merasa gembira itu wajar dan manusiawi. Ekspresi wajah saat gembira juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["kesal"],
    replies: [
      "Merasa kesal itu wajar dan manusiawi. Ekspresi wajah saat kesal juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["frustrasi"],
    replies: [
      "Merasa frustrasi itu wajar dan manusiawi. Ekspresi wajah saat frustrasi juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["putus asa"],
    replies: [
      "Merasa putus asa itu wajar dan manusiawi. Ekspresi wajah saat putus asa juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["bersyukur"],
    replies: [
      "Merasa bersyukur itu wajar dan manusiawi. Ekspresi wajah saat bersyukur juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["haru"],
    replies: [
      "Merasa haru itu wajar dan manusiawi. Ekspresi wajah saat haru juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["gelisah"],
    replies: [
      "Merasa gelisah itu wajar dan manusiawi. Ekspresi wajah saat gelisah juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["antusias"],
    replies: [
      "Merasa antusias itu wajar dan manusiawi. Ekspresi wajah saat antusias juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["percaya diri"],
    replies: [
      "Merasa percaya diri itu wajar dan manusiawi. Ekspresi wajah saat percaya diri juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["minder"],
    replies: [
      "Merasa minder itu wajar dan manusiawi. Ekspresi wajah saat minder juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["iri hati"],
    replies: [
      "Merasa iri hati itu wajar dan manusiawi. Ekspresi wajah saat iri hati juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["cuek"],
    replies: [
      "Merasa cuek itu wajar dan manusiawi. Ekspresi wajah saat cuek juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["peduli"],
    replies: [
      "Merasa peduli itu wajar dan manusiawi. Ekspresi wajah saat peduli juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["berempati"],
    replies: [
      "Merasa berempati itu wajar dan manusiawi. Ekspresi wajah saat berempati juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["bersimpati"],
    replies: [
      "Merasa bersimpati itu wajar dan manusiawi. Ekspresi wajah saat bersimpati juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["kagum"],
    replies: [
      "Merasa kagum itu wajar dan manusiawi. Ekspresi wajah saat kagum juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["terinspirasi"],
    replies: [
      "Merasa terinspirasi itu wajar dan manusiawi. Ekspresi wajah saat terinspirasi juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["termotivasi"],
    replies: [
      "Merasa termotivasi itu wajar dan manusiawi. Ekspresi wajah saat termotivasi juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["puas"],
    replies: [
      "Merasa puas itu wajar dan manusiawi. Ekspresi wajah saat puas juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["tidak puas"],
    replies: [
      "Merasa tidak puas itu wajar dan manusiawi. Ekspresi wajah saat tidak puas juga bisa jadi momen foto yang sangat jujur dan berkesan."
    ]
  },
  {
    keywords: ["islam"],
    replies: [
      "Islam adalah salah satu agama yang diakui dan dianut oleh masyarakat di Indonesia."
    ]
  },
  {
    keywords: ["kristen protestan"],
    replies: [
      "Kristen protestan adalah salah satu agama yang diakui dan dianut oleh masyarakat di Indonesia."
    ]
  },
  {
    keywords: ["katolik"],
    replies: [
      "Katolik adalah salah satu agama yang diakui dan dianut oleh masyarakat di Indonesia."
    ]
  },
  {
    keywords: ["hindu"],
    replies: [
      "Hindu adalah salah satu agama yang diakui dan dianut oleh masyarakat di Indonesia."
    ]
  },
  {
    keywords: ["buddha"],
    replies: [
      "Buddha adalah salah satu agama yang diakui dan dianut oleh masyarakat di Indonesia."
    ]
  },
  {
    keywords: ["konghucu"],
    replies: [
      "Konghucu adalah salah satu agama yang diakui dan dianut oleh masyarakat di Indonesia."
    ]
  },
  {
    keywords: ["apel"],
    replies: [
      "Apel adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["jeruk"],
    replies: [
      "Jeruk adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["pisang"],
    replies: [
      "Pisang adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["mangga"],
    replies: [
      "Mangga adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["anggur"],
    replies: [
      "Anggur adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["semangka"],
    replies: [
      "Semangka adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["melon"],
    replies: [
      "Melon adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["pepaya"],
    replies: [
      "Pepaya adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["nanas"],
    replies: [
      "Nanas adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["stroberi"],
    replies: [
      "Stroberi adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["durian"],
    replies: [
      "Durian adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["rambutan"],
    replies: [
      "Rambutan adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["manggis"],
    replies: [
      "Manggis adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["salak"],
    replies: [
      "Salak adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["sawo"],
    replies: [
      "Sawo adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["kedondong"],
    replies: [
      "Kedondong adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["belimbing"],
    replies: [
      "Belimbing adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["jambu biji"],
    replies: [
      "Jambu biji adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["jambu air"],
    replies: [
      "Jambu air adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["kelengkeng"],
    replies: [
      "Kelengkeng adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["leci"],
    replies: [
      "Leci adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["alpukat"],
    replies: [
      "Alpukat adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["kelapa"],
    replies: [
      "Kelapa adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["kiwi"],
    replies: [
      "Kiwi adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["pir"],
    replies: [
      "Pir adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["persik"],
    replies: [
      "Persik adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["plum"],
    replies: [
      "Plum adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["ceri"],
    replies: [
      "Ceri adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["blueberry"],
    replies: [
      "Blueberry adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["delima"],
    replies: [
      "Delima adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["markisa"],
    replies: [
      "Markisa adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["sirsak"],
    replies: [
      "Sirsak adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["nangka"],
    replies: [
      "Nangka adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["cempedak"],
    replies: [
      "Cempedak adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["duku"],
    replies: [
      "Duku adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["langsat"],
    replies: [
      "Langsat adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["matoa"],
    replies: [
      "Matoa adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["srikaya"],
    replies: [
      "Srikaya adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["buah naga"],
    replies: [
      "Buah naga adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["lemon"],
    replies: [
      "Lemon adalah buah yang segar dan warnanya menarik untuk difoto. Gunakan mode makro di AMPER.AI supaya tekstur kulit buahnya terlihat jelas."
    ]
  },
  {
    keywords: ["bayam"],
    replies: [
      "Bayam adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["kangkung"],
    replies: [
      "Kangkung adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["wortel"],
    replies: [
      "Wortel adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["brokoli"],
    replies: [
      "Brokoli adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["kembang kol"],
    replies: [
      "Kembang kol adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["kubis"],
    replies: [
      "Kubis adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["sawi"],
    replies: [
      "Sawi adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["selada"],
    replies: [
      "Selada adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["tomat"],
    replies: [
      "Tomat adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["timun"],
    replies: [
      "Timun adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["terong"],
    replies: [
      "Terong adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["kentang"],
    replies: [
      "Kentang adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["labu siam"],
    replies: [
      "Labu siam adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["buncis"],
    replies: [
      "Buncis adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["kacang panjang"],
    replies: [
      "Kacang panjang adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["jagung"],
    replies: [
      "Jagung adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["paprika"],
    replies: [
      "Paprika adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["cabai"],
    replies: [
      "Cabai adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["bawang merah"],
    replies: [
      "Bawang merah adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["bawang putih"],
    replies: [
      "Bawang putih adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["bawang bombay"],
    replies: [
      "Bawang bombay adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["jahe"],
    replies: [
      "Jahe adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["kunyit"],
    replies: [
      "Kunyit adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["lengkuas"],
    replies: [
      "Lengkuas adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["daun bawang"],
    replies: [
      "Daun bawang adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["seledri"],
    replies: [
      "Seledri adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["tauge"],
    replies: [
      "Tauge adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["pare"],
    replies: [
      "Pare adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["gambas"],
    replies: [
      "Gambas adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["kacang polong"],
    replies: [
      "Kacang polong adalah sayuran segar yang sering jadi objek food photography. Atur pencahayaan lembut di AMPER.AI supaya warnanya tetap alami."
    ]
  },
  {
    keywords: ["gitar"],
    replies: [
      "Gitar adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["piano"],
    replies: [
      "Piano adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["biola"],
    replies: [
      "Biola adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["drum"],
    replies: [
      "Drum adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["gitar bass"],
    replies: [
      "Gitar bass adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["seruling"],
    replies: [
      "Seruling adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["terompet"],
    replies: [
      "Terompet adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["saksofon"],
    replies: [
      "Saksofon adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["harmonika"],
    replies: [
      "Harmonika adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["angklung"],
    replies: [
      "Angklung adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["gamelan"],
    replies: [
      "Gamelan adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["kolintang"],
    replies: [
      "Kolintang adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["suling"],
    replies: [
      "Suling adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["rebana"],
    replies: [
      "Rebana adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["kendang"],
    replies: [
      "Kendang adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["cello"],
    replies: [
      "Cello adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["harpa"],
    replies: [
      "Harpa adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["organ"],
    replies: [
      "Organ adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["ukulele"],
    replies: [
      "Ukulele adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["biola listrik"],
    replies: [
      "Biola listrik adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["gitar akustik"],
    replies: [
      "Gitar akustik adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["gitar elektrik"],
    replies: [
      "Gitar elektrik adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["tamborin"],
    replies: [
      "Tamborin adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["marakas"],
    replies: [
      "Marakas adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["triangle"],
    replies: [
      "Triangle adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["xylophone"],
    replies: [
      "Xylophone adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["bonang"],
    replies: [
      "Bonang adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["siter"],
    replies: [
      "Siter adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["sasando"],
    replies: [
      "Sasando adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
  },
  {
    keywords: ["tifa"],
    replies: [
      "Tifa adalah alat musik yang menarik untuk difoto saat sedang dimainkan. Tangkap momennya dan pertajam detail di AMPER.AI."
    ]
   },
    {
      keywords: ['bye', 'dadah', 'see you', 'keluar', 'sampai jumpa'],
      replies: [
        "Sayounara~ Sampai jumpa lagi di lain waktu! Jangan lupa kembali ke AMPER.AI kalau mau ngedit foto lagi ya. Bye-bye! 👋🌸"
      ]
    }
  ];

  function findReply(input){
    const text = input.toLowerCase();
    for(let item of knowledgeBase){
      for(let kw of item.keywords){
        if(text.includes(kw)){
          const r = item.replies;
          return r[Math.floor(Math.random() * r.length)];
        }
      }
    }
    return "Maaf Ya...,Yuki Masih Belum Di Program Sedeteail Itu..Jadi Yuki Belum Bisa Menjawab Lebih Ke Pertanyaan Itu.. 🌸";
  }

  function sendMessage(){
    const txt = userInput.value.trim();
    if(!txt) return;
    addBubble('user', txt);
    userInput.value = '';
    setTimeout(() => {
      const reply = findReply(txt);
      addBubble('ai', reply);
    }, 400);
  }

  sendBtn.addEventListener('click', sendMessage);
  userInput.addEventListener('keydown', (e) => {
    if(e.key === 'Enter' && !e.shiftKey){
      e.preventDefault();
      sendMessage();
    }
  });

  window.addEventListener('load', () => {
    addBubble('ai', 'Konnichiwa~ 🌸 Aku Yuki. Silakan unggah fotomu atau tanyakan apa saja seputar fotografi!');
  });
</script>
</body>
</html>"""

    st.components.v1.html(yuki_html, height=500, scrolling=False)

# ---------------- Tampilkan foto & proses (Area Utama) ----------------
if uploaded_file is not None and img is not None:
  col_orig, col_res = st.columns(2)
  with col_orig:
    st.subheader("🎆 Foto Asli")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

  if process_btn or "processed_img" not in st.session_state:
    try:
      with st.spinner("🛠️ Yuki & sistem sedang merender proses foto..."):
        scale_factor = 2 if "2x" in upscale_choice else 4
        h, w = img.shape[:2]

        out_pixels = (w * scale_factor) * (h * scale_factor)
        if out_pixels > MAX_OUTPUT_MEGAPIXELS:
          adjusted_scale = (MAX_OUTPUT_MEGAPIXELS / (w * h)) ** 0.5
          scale_factor = max(1.0, adjusted_scale)
          st.warning(f"⚠️ Skala disesuaikan otomatis menjadi {scale_factor:.2f}x demi memori server.")

        # 1. Body Slimming & Retouch Pro
        if body_slim > 0:
          src_pts = np.float32([[w/2, h/2], [w/2, h*0.2], [w/2, h*0.8]])
          # Simulasi deformasi mesh ringan untuk body/contouring
          factor = 1.0 - (body_slim * 0.0008)
          resized_body = cv2.resize(img, (int(w), int(h * factor)))
          if resized_body.shape[0] < h:
            pad_top = (h - resized_body.shape[0]) // 2
            pad_bot = h - resized_body.shape[0] - pad_top
            img = cv2.copyMakeBorder(resized_body, pad_top, pad_bot, 0, 0, cv2.BORDER_REFLECT)
          gc.collect()

        if remini_boost > 0:
          skin_smooth = cv2.bilateralFilter(img, int(remini_boost / 5) * 2 + 5, 75, 75)
          sigma_val = 10 + (remini_boost / 100.0) * 20
          img = cv2.detailEnhance(skin_smooth, sigma_s=sigma_val, sigma_r=0.15)
          del skin_smooth
          gc.collect()

        # 2. Layer Blending (Gabung Foto Kedua)
        if enable_layer and layer_file is not None:
          layer_bytes = np.asarray(bytearray(layer_file.read()), dtype=np.uint8)
          layer_img = cv2.imdecode(layer_bytes, cv2.IMREAD_COLOR)
          if layer_img is not None:
            layer_resized = cv2.resize(layer_img, (img.shape[1], img.shape[0]))
            if layer_mode == "Normal":
              img = cv2.addWeighted(img, 1.0 - layer_opacity, layer_resized, layer_opacity, 0)
            elif layer_mode == "Screen":
              screen = 255 - ((255 - img) * (255 - layer_resized) / 255.0)
              img = cv2.addWeighted(img, 1.0 - layer_opacity, screen.astype("uint8"), layer_opacity, 0)
            elif layer_mode == "Multiply":
              multiply = (img.astype(float) * layer_resized.astype(float) / 255.0)
              img = cv2.addWeighted(img, 1.0 - layer_opacity, multiply.astype("uint8"), layer_opacity, 0)
            else: # Overlay
              overlay = np.where(img < 128, (2 * img * layer_resized / 255.0), (255 - 2 * (255 - img) * (255 - layer_resized) / 255.0))
              img = cv2.addWeighted(img, 1.0 - layer_opacity, overlay.astype("uint8"), layer_opacity, 0)
            del layer_img, layer_resized
            gc.collect()

        if bg_blur > 0:
          blur_kernel = int(bg_blur / 5) * 2 + 1
          bg_blurred = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), bg_blur / 2.0)
          rows, cols = img.shape[:2]
          kernel_x = cv2.getGaussianKernel(cols, cols / 2.5)
          kernel_y = cv2.getGaussianKernel(rows, rows / 2.5)
          mask = kernel_y * kernel_x.T
          mask = mask / mask.max()
          mask = np.dstack([mask, mask, mask])
          img = (img * mask + bg_blurred * (1.0 - mask)).astype("uint8")
          del bg_blurred, kernel_x, kernel_y, mask
          gc.collect()

        if denoise_strength > 0:
          img = cv2.fastNlMeansDenoisingColored(img, None, float(denoise_strength), float(denoise_strength), 7, 21)

        img_f = img.astype("float32") / 255.0
        img_f = apply_tone_curve(img_f, curve_preset)

        if exposure != 0.0:
          img_f = img_f * (2.0**exposure)
        if contrast != 0:
          f_contrast = (259 * (contrast + 255)) / (255 * (259 - contrast))
          img_f = f_contrast * (img_f - 0.5) + 0.5
        img_f = np.clip(img_f, 0, 1)

        lab = cv2.cvtColor((img_f * 255).astype("uint8"), cv2.COLOR_BGR2LAB).astype("float32")
        del img_f
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_norm = l_ch / 255.0

        if highlights != 0:
          hl_mask = np.clip((l_norm - 0.5) * 2.0, 0, 1)
          l_ch += highlights * 0.3 * hl_mask
        if shadows != 0:
          sh_mask = np.clip((0.5 - l_norm) * 2.0, 0, 1)
          l_ch += shadows * 0.3 * sh_mask

        # 3. Selective Edit (Control Points Masking)
        if enable_selective:
          rows_s, cols_s = l_ch.shape
          cx = int(cols_s * (sel_x_pct / 100.0))
          cy = int(rows_s * (sel_y_pct / 100.0))
          Y_grid, X_grid = np.ogrid[:rows_s, :cols_s]
          dist_from_center = np.sqrt((X_grid - cx)**2 + (Y_grid - cy)**2)
          sel_mask = np.clip(1.0 - (dist_from_center / float(sel_radius)), 0, 1)
          
          if sel_exposure != 0.0:
            l_ch += (sel_exposure * 40.0) * sel_mask
          if sel_sat != 0:
            a_ch += (sel_sat * 0.5) * sel_mask
            b_ch += (sel_sat * 0.5) * sel_mask

        l_ch = np.clip(l_ch, 0, 255)
        a_ch = np.clip(a_ch, 0, 255)
        b_ch = np.clip(b_ch, 0, 255)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        adjusted_bgr = cv2.cvtColor(lab.astype("uint8"), cv2.COLOR_LAB2BGR).astype("float32") / 255.0

        new_w = max(1, int(w * scale_factor))
        new_h = max(1, int(h * scale_factor))
        upscaled = cv2.resize((adjusted_bgr * 255).astype("uint8"), (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        st.session_state["processed_img"] = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        gc.collect()

    except Exception as e:
      st.error("❌ Opss..Terjadi kesalahan saat memproses gambar.")
      with st.expander("Detail teknis error"):
        st.exception(e)
      st.session_state.pop("processed_img", None)

  with col_res:
    st.subheader("🎇 Hasil Ai-Upscaller & Pro Suite")
    if "processed_img" in st.session_state:
      st.image(st.session_state["processed_img"], use_container_width=True)

      result_pil = Image.fromarray(st.session_state["processed_img"])
      buf = io.BytesIO()
      result_pil.save(buf, format="JPEG", quality=95)
      byte_im = buf.getvalue()

      st.download_button(
          label="📥 Unduh Foto Hasil Amper.Ai Style (JPEG)",
          data=byte_im,
          file_name="amper_ai_pro_style.jpg",
          mime="image/jpeg",
          use_container_width=True,
      )
else:
  st.markdown("---")
  
  # --- BANNER KATA-KATA MOTIVASI FOTOGRAFI ---
  st.markdown("""
      <div style="background: linear-gradient(135deg, rgba(227,179,74,0.15), rgba(31,111,92,0.2)); padding: 22px; border-radius: 14px; border: 1px solid rgba(227,179,74,0.4); text-align: center; margin-bottom: 25px;">
          <h3 style="color: #f3cf83; margin: 0 0 8px 0; font-family: 'Georgia', serif;">"Fotografi adalah cerita yang gagal diungkapkan dengan kata-kata."</h3>
          <p style="color: #a9d6c9; font-size: 0.95em; margin: 0; font-style: italic;">— Abadikan momen terbaikmu, sempurnakan warnanya, dan biarkan karya berbicara bersama AMPER.AI & Yuki-Chan.</p>
      </div>
  """, unsafe_allow_html=True)

  # --- KARTU FITUR UNGGULAN (GRID 3 KOLOM) ---
  col_f1, col_f2, col_f3 = st.columns(3)
  
  with col_f1:
    st.markdown("""
        <div style="background: rgba(18, 34, 38, 0.75); padding: 20px; border-radius: 12px; border: 1px solid rgba(227, 179, 74, 0.25); height: 100%;">
            <h4 style="color: #e3b34a; margin-top: 0;">🌸 Yuki AI Companion</h4>
            <p style="font-size: 0.85em; color: #cfe8e1; margin-bottom: 0;">Asisten virtual cerdas di sidebar yang dilengkapi 35+ kamus topik siap sedia mendampingi proses kreatifmu.</p>
        </div>
    """, unsafe_allow_html=True)
    
  with col_f2:
    st.markdown("""
        <div style="background: rgba(18, 34, 38, 0.75); padding: 20px; border-radius: 12px; border: 1px solid rgba(227, 179, 74, 0.25); height: 100%;">
            <h4 style="color: #e3b34a; margin-top: 0;">🎊 Selective & Layer Pro</h4>
            <p style="font-size: 0.85em; color: #cfe8e1; margin-bottom: 0;">Gunakan kontrol titik spesifik (selective edit) dan gabungkan beberapa foto dengan dukungan layer blending instan.</p>
        </div>
    """, unsafe_allow_html=True)
    
  with col_f3:
    st.markdown("""
        <div style="background: rgba(18, 34, 38, 0.75); padding: 20px; border-radius: 12px; border: 1px solid rgba(227, 179, 74, 0.25); height: 100%;">
            <h4 style="color: #e3b34a; margin-top: 0;">🎨 10+ Pro Cinematic Presets</h4>
            <p style="font-size: 0.85em; color: #cfe8e1; margin-bottom: 0;">Pilih beragam filter gaya sinematik, cyberpunk, hingga golden hour untuk mempercantik foto instan.</p>
        </div>
    """, unsafe_allow_html=True)

  st.markdown("<br><p style='text-align: center; color: #8fa8a2; font-size: 0.95em;'>👆 Silakan unggah file foto di atas untuk mulai mengedit karya fotografimu.</p>", unsafe_allow_html=True)
