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
  from auth import render_auth_sidebar, get_credits, deduct_credit

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


set_custom_theme()
set_background(BG_PATH)

current_user = None
if REQUIRE_LOGIN:
  is_logged_in = render_auth_sidebar()
  if not is_logged_in:
    st.title("👾 AMPER.AI — Pro Suite & Yuki-Chan")
    st.info("Silakan Masuk atau Daftar lewat panel kiri untuk mulai.")
    st.stop()
  current_user = st.session_state["user"]

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
    "📂 Unggah File Foto Keren Kamu Disini... (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
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

  st.markdown("### 🎭 Ampera.ai Face Enhancer & Portrait")
  remini_boost = st.slider(
      "Perjelas Wajah & Kulit (Remini Effect)", 0, 100, 0, 1
  )
  bg_blur = st.slider(
      "Efek Latar Belakang (Bokeh / Blur Halus)", 0, 100, 0, 2
  )

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

  st.markdown("### 🎬 CapCut & Pro Filter Presets")
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
      ],
  )

  if st.button("🪄 Terapkan Preset Pilihan"):
    if capcut_preset.startswith("✨ Cyberpunk"):
      st.session_state.update(
          {
              "exposure": 0.2,
              "contrast": 25,
              "highlights": -10,
              "shadows": 15,
              "temp": -15,
              "tint": 15,
              "vibrance": 35,
              "saturation": 20,
              "clarity": 25,
              "vignette": 40,
          }
      )
    elif capcut_preset.startswith("🎞️ Vintage"):
      st.session_state.update(
          {
              "exposure": 0.1,
              "contrast": 10,
              "highlights": -20,
              "shadows": 30,
              "temp": 25,
              "tint": -5,
              "vibrance": -10,
              "saturation": -5,
              "clarity": 10,
              "vignette": 50,
          }
      )
    elif capcut_preset.startswith("🎬 Moody"):
      st.session_state.update(
          {
              "exposure": -0.3,
              "contrast": 35,
              "highlights": -40,
              "shadows": -20,
              "temp": -10,
              "tint": 5,
              "vibrance": 10,
              "saturation": 5,
              "clarity": 30,
              "vignette": 65,
          }
      )
    elif capcut_preset.startswith("🌟 Clean"):
      st.session_state.update(
          {
              "exposure": 0.3,
              "contrast": 15,
              "highlights": 10,
              "shadows": 25,
              "temp": 0,
              "tint": 0,
              "vibrance": 20,
              "saturation": 15,
              "clarity": 15,
              "vignette": 10,
          }
      )
    elif capcut_preset.startswith("☕ Warm Portrait"):
      st.session_state.update(
          {
              "exposure": 0.1,
              "contrast": 5,
              "highlights": 10,
              "shadows": 20,
              "temp": 15,
              "tint": 5,
              "vibrance": 15,
              "saturation": 5,
              "clarity": 5,
              "vignette": 15,
          }
      )
    elif capcut_preset.startswith("🖤 Dramatic"):
      st.session_state.update(
          {
              "exposure": 0.0,
              "contrast": 40,
              "highlights": -30,
              "shadows": -30,
              "temp": 0,
              "tint": 0,
              "vibrance": -50,
              "saturation": -50,
              "clarity": 35,
              "vignette": 50,
          }
      )
    st.rerun()

  st.markdown("---")

  if auto_suggestions is not None and capcut_preset == "Normal / Manual":
    if st.button("🪄 Auto Enhance Standar"):
      for slider_key, val in auto_suggestions.items():
        st.session_state[slider_key] = val
      st.rerun()
    st.markdown("---")

  st.markdown("### 1. Light & Exposure")
  exposure = st.slider("Exposure", -2.0, 2.0, 0.0, 0.1, key="exposure")
  contrast = st.slider("Contrast", -50, 50, 10, 1, key="contrast")
  highlights = st.slider("Highlights", -100, 100, -20, 1, key="highlights")
  shadows = st.slider("Shadows", -100, 100, 25, 1, key="shadows")
  whites = st.slider("Whites", -50, 50, 0, 1, key="whites")
  blacks = st.slider("Blacks", -50, 50, 0, 1, key="blacks")

  st.markdown("### 2. Color & White Balance")
  temp = st.slider("Temperature (Kelvin/Tint)", -50, 50, -5, 1, key="temp")
  tint = st.slider("Tint", -50, 50, 0, 1, key="tint")
  vibrance = st.slider("Vibrance", -50, 50, 15, 1, key="vibrance")
  saturation = st.slider("Saturation", -50, 50, 10, 1, key="saturation")

  st.markdown("### 3. Detail, Clarity & Effects")
  clarity = st.slider("Clarity / Texture", -50, 50, 20, 1, key="clarity")
  dehaze = st.slider("Dehaze", -50, 50, 10, 1, key="dehaze")
  sharpen = st.slider("Sharpening HD", 0, 100, 30, 1, key="sharpen")
  vignette = st.slider("Vignette (Cinematic Edge)", 0, 100, 25, 1, key="vignette")

  st.markdown("### 4. Quality Boost")
  denoise_strength = st.slider("Noise Reduction", 0, 30, 0, 1)
  smart_enhance = st.slider("Smart Detail Enhance", 0, 100, 0, 1)

  st.markdown("---")
  upscale_choice = st.selectbox(
      "Resolution Upscaling", ["2x (HD 2K)", "4x (Ultra HD 4K)"], index=0
  )
  process_btn = st.button("⬆️ Terapkan & Render Instan")

  # --- CHATBOT YUKI DI SIDEBAR (DIPASANG MENGGUNAKAN COMPONENTS.HTML) ---
  st.markdown("---")
  st.markdown("### 🌸 Asisten AI Yuki-Chan")

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
  from auth import render_auth_sidebar, get_credits, deduct_credit

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


set_custom_theme()
set_background(BG_PATH)

current_user = None
if REQUIRE_LOGIN:
  is_logged_in = render_auth_sidebar()
  if not is_logged_in:
    st.title("👾 AMPER.AI — Pro Suite & Yuki-Chan")
    st.info("Silakan Masuk atau Daftar lewat panel kiri untuk mulai.")
    st.stop()
  current_user = st.session_state["user"]

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
    "📂 Unggah File Foto Keren Kamu Disini... (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
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

  st.markdown("### 🎭 Ampera.ai Face Enhancer & Portrait")
  remini_boost = st.slider(
      "Perjelas Wajah & Kulit (Remini Effect)", 0, 100, 0, 1
  )
  bg_blur = st.slider(
      "Efek Latar Belakang (Bokeh / Blur Halus)", 0, 100, 0, 2
  )

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

  st.markdown("### 🎬 CapCut & Pro Filter Presets")
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
      ],
  )

  if st.button("🪄 Terapkan Preset Pilihan"):
    if capcut_preset.startswith("✨ Cyberpunk"):
      st.session_state.update(
          {
              "exposure": 0.2,
              "contrast": 25,
              "highlights": -10,
              "shadows": 15,
              "temp": -15,
              "tint": 15,
              "vibrance": 35,
              "saturation": 20,
              "clarity": 25,
              "vignette": 40,
          }
      )
    elif capcut_preset.startswith("🎞️ Vintage"):
      st.session_state.update(
          {
              "exposure": 0.1,
              "contrast": 10,
              "highlights": -20,
              "shadows": 30,
              "temp": 25,
              "tint": -5,
              "vibrance": -10,
              "saturation": -5,
              "clarity": 10,
              "vignette": 50,
          }
      )
    elif capcut_preset.startswith("🎬 Moody"):
      st.session_state.update(
          {
              "exposure": -0.3,
              "contrast": 35,
              "highlights": -40,
              "shadows": -20,
              "temp": -10,
              "tint": 5,
              "vibrance": 10,
              "saturation": 5,
              "clarity": 30,
              "vignette": 65,
          }
      )
    elif capcut_preset.startswith("🌟 Clean"):
      st.session_state.update(
          {
              "exposure": 0.3,
              "contrast": 15,
              "highlights": 10,
              "shadows": 25,
              "temp": 0,
              "tint": 0,
              "vibrance": 20,
              "saturation": 15,
              "clarity": 15,
              "vignette": 10,
          }
      )
    elif capcut_preset.startswith("☕ Warm Portrait"):
      st.session_state.update(
          {
              "exposure": 0.1,
              "contrast": 5,
              "highlights": 10,
              "shadows": 20,
              "temp": 15,
              "tint": 5,
              "vibrance": 15,
              "saturation": 5,
              "clarity": 5,
              "vignette": 15,
          }
      )
    elif capcut_preset.startswith("🖤 Dramatic"):
      st.session_state.update(
          {
              "exposure": 0.0,
              "contrast": 40,
              "highlights": -30,
              "shadows": -30,
              "temp": 0,
              "tint": 0,
              "vibrance": -50,
              "saturation": -50,
              "clarity": 35,
              "vignette": 50,
          }
      )
    st.rerun()

  st.markdown("---")

  if auto_suggestions is not None and capcut_preset == "Normal / Manual":
    if st.button("🪄 Auto Enhance Standar"):
      for slider_key, val in auto_suggestions.items():
        st.session_state[slider_key] = val
      st.rerun()
    st.markdown("---")

  st.markdown("### 1. Light & Exposure")
  exposure = st.slider("Exposure", -2.0, 2.0, 0.0, 0.1, key="exposure")
  contrast = st.slider("Contrast", -50, 50, 10, 1, key="contrast")
  highlights = st.slider("Highlights", -100, 100, -20, 1, key="highlights")
  shadows = st.slider("Shadows", -100, 100, 25, 1, key="shadows")
  whites = st.slider("Whites", -50, 50, 0, 1, key="whites")
  blacks = st.slider("Blacks", -50, 50, 0, 1, key="blacks")

  st.markdown("### 2. Color & White Balance")
  temp = st.slider("Temperature (Kelvin/Tint)", -50, 50, -5, 1, key="temp")
  tint = st.slider("Tint", -50, 50, 0, 1, key="tint")
  vibrance = st.slider("Vibrance", -50, 50, 15, 1, key="vibrance")
  saturation = st.slider("Saturation", -50, 50, 10, 1, key="saturation")

  st.markdown("### 3. Detail, Clarity & Effects")
  clarity = st.slider("Clarity / Texture", -50, 50, 20, 1, key="clarity")
  dehaze = st.slider("Dehaze", -50, 50, 10, 1, key="dehaze")
  sharpen = st.slider("Sharpening HD", 0, 100, 30, 1, key="sharpen")
  vignette = st.slider("Vignette (Cinematic Edge)", 0, 100, 25, 1, key="vignette")

  st.markdown("### 4. Quality Boost")
  denoise_strength = st.slider("Noise Reduction", 0, 30, 0, 1)
  smart_enhance = st.slider("Smart Detail Enhance", 0, 100, 0, 1)

  st.markdown("---")
  upscale_choice = st.selectbox(
      "Resolution Upscaling", ["2x (HD 2K)", "4x (Ultra HD 4K)"], index=0
  )
  process_btn = st.button("⬆️ Terapkan & Render Instan")

  # --- CHATBOT YUKI DI SIDEBAR (DIPASANG MENGGUNAKAN COMPONENTS.HTML) ---
  st.markdown("---")
  st.markdown("### 🌸 Asisten AI Yuki-Chan")

     st.components.v1.html(yuki_html, height=500, scrolling=False)

  components.html(yuki_html, height=470)

# ---------------- Tampilkan foto & proses (Area Utama) ----------------
if uploaded_file is not None and img is not None:
  col_orig, col_res = st.columns(2)
  with col_orig:
    st.subheader("🎆 Foto Asli")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

  if process_btn or "processed_img" not in st.session_state:
    if REQUIRE_LOGIN:
      user_credits = get_credits(current_user["id"])
      if user_credits <= 0:
        st.error("💳 Kredit Kamu habis. Silakan top up.")
        st.stop()
    try:
      with st.spinner("🛠️ Yuki & sistem sedang merender Remini & Upscaler..."):
        scale_factor = 2 if "2x" in upscale_choice else 4
        h, w = img.shape[:2]

        out_pixels = (w * scale_factor) * (h * scale_factor)
        if out_pixels > MAX_OUTPUT_MEGAPIXELS:
          adjusted_scale = (MAX_OUTPUT_MEGAPIXELS / (w * h)) ** 0.5
          scale_factor = max(1.0, adjusted_scale)
          st.warning(
              f"⚠️ Maaf Ya.. Skala disesuaikan otomatis menjadi {scale_factor:.2f}x"
              " demi keamanan memori server."
          )

        # --- APLIKASI EFEK REMINI (Perjelas Wajah & Kulit) ---
        if remini_boost > 0:
          skin_smooth = cv2.bilateralFilter(
              img, int(remini_boost / 5) * 2 + 5, 75, 75
          )
          sigma_val = 10 + (remini_boost / 100.0) * 20
          img = cv2.detailEnhance(skin_smooth, sigma_s=sigma_val, sigma_r=0.15)
          del skin_smooth
          gc.collect()

        # --- APLIKASI EFEK LATAR BELAKANG (Bokeh / Blur) ---
        if bg_blur > 0:
          blur_kernel = int(bg_blur / 5) * 2 + 1
          bg_blurred = cv2.GaussianBlur(
              img, (blur_kernel, blur_kernel), bg_blur / 2.0
          )
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
          img = cv2.fastNlMeansDenoisingColored(
              img, None, float(denoise_strength), float(denoise_strength), 7, 21
          )

        img_f = img.astype("float32") / 255.0

        # Terapkan Tone Curve
        img_f = apply_tone_curve(img_f, curve_preset)

        if exposure != 0.0:
          img_f = img_f * (2.0**exposure)
        if contrast != 0:
          f_contrast = (259 * (contrast + 255)) / (255 * (259 - contrast))
          img_f = f_contrast * (img_f - 0.5) + 0.5
        img_f = np.clip(img_f, 0, 1)

        lab = cv2.cvtColor(
            (img_f * 255).astype("uint8"), cv2.COLOR_BGR2LAB
        ).astype("float32")
        del img_f
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_norm = l_ch / 255.0

        if highlights != 0:
          hl_mask = np.clip((l_norm - 0.5) * 2.0, 0, 1)
          l_ch += highlights * 0.3 * hl_mask
        if shadows != 0:
          sh_mask = np.clip((0.5 - l_norm) * 2.0, 0, 1)
          l_ch += shadows * 0.3 * sh_mask
        if whites != 0:
          w_mask = np.clip(l_norm, 0, 1)
          l_ch += whites * 0.2 * w_mask
        if blacks != 0:
          b_mask = np.clip(1.0 - l_norm, 0, 1)
          l_ch += blacks * 0.2 * b_mask

        l_ch = np.clip(l_ch, 0, 255)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        del l_ch, a_ch, b_ch, l_norm
        adjusted_bgr = (
            cv2.cvtColor(lab.astype("uint8"), cv2.COLOR_LAB2BGR).astype(
                "float32"
            )
            / 255.0
        )
        del lab

        if temp != 0:
          adjusted_bgr[:, :, 0] -= temp * 0.002
          adjusted_bgr[:, :, 2] += temp * 0.002
        if tint != 0:
          adjusted_bgr[:, :, 1] += tint * 0.002
        adjusted_bgr = np.clip(adjusted_bgr, 0, 1)

        hsv = cv2.cvtColor(
            (adjusted_bgr * 255).astype("uint8"), cv2.COLOR_BGR2HSV
        ).astype("float32")
        del adjusted_bgr
        if saturation != 0:
          sat_mult = 1.0 + (saturation / 100.0)
          hsv[:, :, 1] *= sat_mult
        if vibrance != 0:
          v_mask = 1.0 - (hsv[:, :, 1] / 255.0)
          hsv[:, :, 1] += vibrance * 0.5 * v_mask
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

        sat_adj_small = cv2.cvtColor(
            hsv.astype("uint8"), cv2.COLOR_HSV2BGR
        )
        del hsv
        gc.collect()

        new_w = max(1, int(w * scale_factor))
        new_h = max(1, int(h * scale_factor))
        upscaled = cv2.resize(
            sat_adj_small, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
        )
        del sat_adj_small
        gc.collect()

        sat_adj = upscaled.astype("float32") / 255.0
        del upscaled

        if clarity != 0 or dehaze != 0 or sharpen > 0:
          if dehaze != 0:
            dark_channel = cv2.min(
                cv2.min(sat_adj[:, :, 0], sat_adj[:, :, 1]), sat_adj[:, :, 2]
            )
            dehaze_mask = 1.0 - (dark_channel * (dehaze / 50.0))
            sat_adj = sat_adj * np.dstack(
                [dehaze_mask, dehaze_mask, dehaze_mask]
            )
            del dark_channel, dehaze_mask

          blur_radius = max(1, int(sat_adj.shape[0] / 200)) * 2 + 1
          gaussian = cv2.GaussianBlur(sat_adj, (blur_radius, blur_radius), 0)
          sharp_weight = (sharpen + abs(clarity)) / 40.0
          sat_adj = cv2.addWeighted(
              sat_adj, 1.0 + sharp_weight, gaussian, -sharp_weight, 0
          )
          del gaussian
          sat_adj = np.clip(sat_adj, 0, 1)

        if vignette > 0:
          rows, cols = sat_adj.shape[:2]
          kernel_x = cv2.getGaussianKernel(cols, cols / 1.5)
          kernel_y = cv2.getGaussianKernel(rows, rows / 1.5)
          kernel = kernel_y * kernel_x.T
          mask = kernel / kernel.max()
          v_factor = vignette / 100.0
          mask = np.power(mask, 1.0 - v_factor * 0.4)
          mask = np.dstack([mask, mask, mask])
          sat_adj = np.clip(sat_adj * mask, 0, 1)
          del kernel_x, kernel_y, kernel, mask

        final_bgr = (sat_adj * 255).astype("uint8")
        del sat_adj
        gc.collect()

        if smart_enhance > 0:
          sigma_s = 10 + (smart_enhance / 100.0) * 40
          sigma_r = 0.15 + (smart_enhance / 100.0) * 0.35
          final_bgr = cv2.detailEnhance(
              final_bgr, sigma_s=sigma_s, sigma_r=sigma_r
          )
          gc.collect()

        st.session_state["processed_img"] = cv2.cvtColor(
            final_bgr, cv2.COLOR_BGR2RGB
        )
        del final_bgr
        gc.collect()

        if REQUIRE_LOGIN:
          deduct_credit(current_user["id"])
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
  st.info("👆 Silakan unggah foto terlebih dahulu di atas.")
  components.html(yuki_html, height=470)

# ---------------- Tampilkan foto & proses (Area Utama) ----------------
if uploaded_file is not None and img is not None:
  col_orig, col_res = st.columns(2)
  with col_orig:
    st.subheader("🎆 Foto Asli")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

  if process_btn or "processed_img" not in st.session_state:
    if REQUIRE_LOGIN:
      user_credits = get_credits(current_user["id"])
      if user_credits <= 0:
        st.error("💳 Kredit Kamu habis. Silakan top up.")
        st.stop()
    try:
      with st.spinner("🛠️ Yuki & sistem sedang merender Remini & Upscaler..."):
        scale_factor = 2 if "2x" in upscale_choice else 4
        h, w = img.shape[:2]

        out_pixels = (w * scale_factor) * (h * scale_factor)
        if out_pixels > MAX_OUTPUT_MEGAPIXELS:
          adjusted_scale = (MAX_OUTPUT_MEGAPIXELS / (w * h)) ** 0.5
          scale_factor = max(1.0, adjusted_scale)
          st.warning(
              f"⚠️ Maaf Ya.. Skala disesuaikan otomatis menjadi {scale_factor:.2f}x"
              " demi keamanan memori server."
          )

        # --- APLIKASI EFEK REMINI (Perjelas Wajah & Kulit) ---
        if remini_boost > 0:
          skin_smooth = cv2.bilateralFilter(
              img, int(remini_boost / 5) * 2 + 5, 75, 75
          )
          sigma_val = 10 + (remini_boost / 100.0) * 20
          img = cv2.detailEnhance(skin_smooth, sigma_s=sigma_val, sigma_r=0.15)
          del skin_smooth
          gc.collect()

        # --- APLIKASI EFEK LATAR BELAKANG (Bokeh / Blur) ---
        if bg_blur > 0:
          blur_kernel = int(bg_blur / 5) * 2 + 1
          bg_blurred = cv2.GaussianBlur(
              img, (blur_kernel, blur_kernel), bg_blur / 2.0
          )
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
          img = cv2.fastNlMeansDenoisingColored(
              img, None, float(denoise_strength), float(denoise_strength), 7, 21
          )

        img_f = img.astype("float32") / 255.0

        # Terapkan Tone Curve
        img_f = apply_tone_curve(img_f, curve_preset)

        if exposure != 0.0:
          img_f = img_f * (2.0**exposure)
        if contrast != 0:
          f_contrast = (259 * (contrast + 255)) / (255 * (259 - contrast))
          img_f = f_contrast * (img_f - 0.5) + 0.5
        img_f = np.clip(img_f, 0, 1)

        lab = cv2.cvtColor(
            (img_f * 255).astype("uint8"), cv2.COLOR_BGR2LAB
        ).astype("float32")
        del img_f
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_norm = l_ch / 255.0

        if highlights != 0:
          hl_mask = np.clip((l_norm - 0.5) * 2.0, 0, 1)
          l_ch += highlights * 0.3 * hl_mask
        if shadows != 0:
          sh_mask = np.clip((0.5 - l_norm) * 2.0, 0, 1)
          l_ch += shadows * 0.3 * sh_mask
        if whites != 0:
          w_mask = np.clip(l_norm, 0, 1)
          l_ch += whites * 0.2 * w_mask
        if blacks != 0:
          b_mask = np.clip(1.0 - l_norm, 0, 1)
          l_ch += blacks * 0.2 * b_mask

        l_ch = np.clip(l_ch, 0, 255)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        del l_ch, a_ch, b_ch, l_norm
        adjusted_bgr = (
            cv2.cvtColor(lab.astype("uint8"), cv2.COLOR_LAB2BGR).astype(
                "float32"
            )
            / 255.0
        )
        del lab

        if temp != 0:
          adjusted_bgr[:, :, 0] -= temp * 0.002
          adjusted_bgr[:, :, 2] += temp * 0.002
        if tint != 0:
          adjusted_bgr[:, :, 1] += tint * 0.002
        adjusted_bgr = np.clip(adjusted_bgr, 0, 1)

        hsv = cv2.cvtColor(
            (adjusted_bgr * 255).astype("uint8"), cv2.COLOR_BGR2HSV
        ).astype("float32")
        del adjusted_bgr
        if saturation != 0:
          sat_mult = 1.0 + (saturation / 100.0)
          hsv[:, :, 1] *= sat_mult
        if vibrance != 0:
          v_mask = 1.0 - (hsv[:, :, 1] / 255.0)
          hsv[:, :, 1] += vibrance * 0.5 * v_mask
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

        sat_adj_small = cv2.cvtColor(
            hsv.astype("uint8"), cv2.COLOR_HSV2BGR
        )
        del hsv
        gc.collect()

        new_w = max(1, int(w * scale_factor))
        new_h = max(1, int(h * scale_factor))
        upscaled = cv2.resize(
            sat_adj_small, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
        )
        del sat_adj_small
        gc.collect()

        sat_adj = upscaled.astype("float32") / 255.0
        del upscaled

        if clarity != 0 or dehaze != 0 or sharpen > 0:
          if dehaze != 0:
            dark_channel = cv2.min(
                cv2.min(sat_adj[:, :, 0], sat_adj[:, :, 1]), sat_adj[:, :, 2]
            )
            dehaze_mask = 1.0 - (dark_channel * (dehaze / 50.0))
            sat_adj = sat_adj * np.dstack(
                [dehaze_mask, dehaze_mask, dehaze_mask]
            )
            del dark_channel, dehaze_mask

          blur_radius = max(1, int(sat_adj.shape[0] / 200)) * 2 + 1
          gaussian = cv2.GaussianBlur(sat_adj, (blur_radius, blur_radius), 0)
          sharp_weight = (sharpen + abs(clarity)) / 40.0
          sat_adj = cv2.addWeighted(
              sat_adj, 1.0 + sharp_weight, gaussian, -sharp_weight, 0
          )
          del gaussian
          sat_adj = np.clip(sat_adj, 0, 1)

        if vignette > 0:
          rows, cols = sat_adj.shape[:2]
          kernel_x = cv2.getGaussianKernel(cols, cols / 1.5)
          kernel_y = cv2.getGaussianKernel(rows, rows / 1.5)
          kernel = kernel_y * kernel_x.T
          mask = kernel / kernel.max()
          v_factor = vignette / 100.0
          mask = np.power(mask, 1.0 - v_factor * 0.4)
          mask = np.dstack([mask, mask, mask])
          sat_adj = np.clip(sat_adj * mask, 0, 1)
          del kernel_x, kernel_y, kernel, mask

        final_bgr = (sat_adj * 255).astype("uint8")
        del sat_adj
        gc.collect()

        if smart_enhance > 0:
          sigma_s = 10 + (smart_enhance / 100.0) * 40
          sigma_r = 0.15 + (smart_enhance / 100.0) * 0.35
          final_bgr = cv2.detailEnhance(
              final_bgr, sigma_s=sigma_s, sigma_r=sigma_r
          )
          gc.collect()

        st.session_state["processed_img"] = cv2.cvtColor(
            final_bgr, cv2.COLOR_BGR2RGB
        )
        del final_bgr
        gc.collect()

        if REQUIRE_LOGIN:
          deduct_credit(current_user["id"])
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
  st.info("👆 Silakan unggah foto terlebih dahulu di atas.")
