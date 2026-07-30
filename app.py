import base64
import gc
import io
import math
from datetime import datetime, date, timedelta

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

    .toolkit-card {
        background: rgba(15, 43, 48, 0.55);
        border: 1px solid rgba(227, 179, 74, 0.25);
        border-radius: 12px;
        padding: 1em 1.2em;
        margin-bottom: 1em;
    }
    .toolkit-result {
        background: rgba(227, 179, 74, 0.12);
        border-left: 4px solid #e3b34a;
        border-radius: 6px;
        padding: 0.8em 1em;
        margin-top: 0.6em;
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


# ==========================================================
# HELPER — TOOLKIT FOTOGRAFER PRO
# ==========================================================

def format_shutter(seconds):
  """Format detik menjadi notasi shutter speed yang umum dipakai fotografer."""
  if seconds <= 0:
    return "0"
  if seconds >= 60:
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m} menit {s:.1f} detik"
  if seconds >= 1:
    return f"{seconds:.1f} detik"
  # shutter cepat -> tampilkan sebagai pecahan 1/x
  denom = round(1.0 / seconds)
  return f"1/{denom} detik"


SENSOR_CROP_FACTORS = {
    "Full Frame (36 x 24mm)": {"crop": 1.0, "width_mm": 36.0},
    "APS-C Canon (1.6x)": {"crop": 1.6, "width_mm": 22.3},
    "APS-C Nikon/Sony/Fuji (1.5x)": {"crop": 1.5, "width_mm": 23.5},
    "Micro Four Thirds (2.0x)": {"crop": 2.0, "width_mm": 17.3},
    "1 inch (2.7x)": {"crop": 2.7, "width_mm": 13.2},
}

ND_FILTER_STOPS = {
    "ND2 (1 stop)": 1,
    "ND4 (2 stop)": 2,
    "ND8 (3 stop)": 3,
    "ND16 (4 stop)": 4,
    "ND32 (5 stop)": 5,
    "ND64 (6 stop)": 6,
    "ND400 (~9 stop)": 9,
    "ND1000 (10 stop)": 10,
    "ND100000 (~16.6 stop)": 16.6,
    "Custom": None,
}

SHOT_LIST_TEMPLATES = {
    "Pernikahan (Wedding Day)": [
        "Detail cincin & undangan", "Persiapan pengantin wanita", "Persiapan pengantin pria",
        "First look (jika ada)", "Prosesi akad/pemberkatan", "Foto keluarga inti",
        "Foto grup besar keluarga", "Tukar cincin", "First kiss", "Resepsi - entrance",
        "Sesi foto pasangan (couple portrait)", "Potong kue & toast", "Lempar bunga/garter",
        "Dance pertama", "Candid tamu undangan", "Foto dekorasi venue",
    ],
    "Prewedding": [
        "Konsep outdoor lokasi 1", "Konsep outdoor lokasi 2", "Konsep indoor/studio",
        "Detail cincin & aksesoris", "Foto candid natural", "Foto formal berpasangan",
        "Golden hour session", "Detail baju & sepatu",
    ],
    "Produk": [
        "Foto flatlay produk", "Foto produk dengan model", "Foto detail tekstur/material",
        "Foto kemasan/packaging", "Foto produk 360 (multi-angle)", "Foto lifestyle konteks pemakaian",
        "Foto ukuran/skala perbandingan",
    ],
    "Event / Korporat": [
        "Foto banner & signage acara", "Foto pembukaan/opening", "Foto pembicara/keynote",
        "Foto audiens & suasana ruangan", "Foto networking session", "Foto grup panitia",
        "Foto dokumentasi produk/booth", "Foto penutupan/closing",
    ],
    "Keluarga / Newborn": [
        "Foto grup keluarga formal", "Foto candid interaksi", "Detail tangan & kaki bayi",
        "Foto bayi dengan orang tua", "Foto dengan properti/mainan", "Foto ekspresi close-up",
    ],
    "Custom (kosong)": [],
}


def get_sun_times(lat, lon, target_date, tz_offset_hours):
  """Hitung waktu matahari terbit/terbenam & golden/blue hour secara astronomis (offline, tanpa API)."""
  # Algoritma NOAA sederhana (Sunrise/Sunset Equation)
  n = target_date.timetuple().tm_yday
  lng_hour = lon / 15.0

  def calc(is_sunrise, zenith):
    t = n + ((6 - lng_hour) / 24.0) if is_sunrise else n + ((18 - lng_hour) / 24.0)
    M = (0.9856 * t) - 3.289
    L = M + (1.916 * math.sin(math.radians(M))) + (0.020 * math.sin(math.radians(2 * M))) + 282.634
    L = L % 360
    RA = math.degrees(math.atan(0.91764 * math.tan(math.radians(L))))
    RA = RA % 360
    Lquadrant = (math.floor(L / 90.0)) * 90
    RAquadrant = (math.floor(RA / 90.0)) * 90
    RA = RA + (Lquadrant - RAquadrant)
    RA = RA / 15.0
    sinDec = 0.39782 * math.sin(math.radians(L))
    cosDec = math.cos(math.asin(sinDec))
    cosH = (math.cos(math.radians(zenith)) - (sinDec * math.sin(math.radians(lat * math.pi / 180 * 0 + lat)))) 
    # simplifikasi ulang perhitungan cosH dengan lat dalam derajat
    cosH = (math.cos(math.radians(zenith)) - (sinDec * math.sin(math.radians(lat)))) / (cosDec * math.cos(math.radians(lat)))
    if cosH > 1 or cosH < -1:
      return None
    if is_sunrise:
      H = 360 - math.degrees(math.acos(cosH))
    else:
      H = math.degrees(math.acos(cosH))
    H = H / 15.0
    T = H + RA - (0.06571 * t) - 6.622
    UT = T - lng_hour
    UT = UT % 24
    local_t = UT + tz_offset_hours
    local_t = local_t % 24
    hh = int(local_t)
    mm = int(round((local_t - hh) * 60))
    if mm == 60:
      mm = 0
      hh = (hh + 1) % 24
    return hh, mm

  sunrise = calc(True, 90.833)
  sunset = calc(False, 90.833)
  civil_dawn = calc(True, 96.0)
  civil_dusk = calc(False, 96.0)
  return sunrise, sunset, civil_dawn, civil_dusk


def hhmm_to_str(t):
  if t is None:
    return "N/A (siang/malam polar)"
  return f"{t[0]:02d}:{t[1]:02d}"


def hhmm_add_minutes(t, minutes):
  if t is None:
    return None
  total = t[0] * 60 + t[1] + minutes
  total = total % (24 * 60)
  return (total // 60, total % 60)


RAW_SIZE_TABLE = {
    "RAW Canon (~30-45 MB)": 38,
    "RAW Nikon (~25-40 MB)": 32,
    "RAW Sony (~24-50 MB)": 35,
    "RAW Fujifilm (~30-50 MB)": 40,
    "JPEG Fine/Large (~8-15 MB)": 10,
    "JPEG Medium (~4-8 MB)": 5,
    "Custom (isi manual)": None,
}

SD_CARD_SIZES_GB = [32, 64, 128, 256, 512, 1024, 2048]


def rupiah(n):
  try:
    return f"Rp {n:,.0f}".replace(",", ".")
  except Exception:
    return f"Rp {n}"


# ==========================================================
# RENDER APLIKASI
# ==========================================================

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
      " Yuki Asisten AI, dan Toolkit Fotografer Profesional!</p>",
      unsafe_allow_html=True,
  )

tab_editor, tab_toolkit = st.tabs(["🎨 Editor Foto & Yuki-Chan", "🧰 Toolkit Fotografer Pro"])

# ==========================================================
# TAB 1 : EDITOR FOTO (KODE ASLI)
# ==========================================================
with tab_editor:

  # Widget File Uploader dengan key unik untuk mencegah error duplikasi ID
  uploaded_file = st.file_uploader(
      "📂 Unggah File Foto Keren Kamu Disini... (JPG, JPEG, PNG)",
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
    denoise_strength = st.slider("Noise Reduction", 0, 30, 0, 1, key="noise_reduction")
    smart_enhance = st.slider("Smart Detail Enhance", 0, 100, 0, 1, key="smart_enhance")

    st.markdown("---")
    upscale_choice = st.selectbox(
        "Resolution Upscaling", ["2x (HD 2K)", "4x (Ultra HD 4K)"], index=0, key="upscale_choice"
    )
    process_btn = st.button("⬆️ Terapkan & Render Instan")

  with st.sidebar:
      # --- CHATBOT YUKI DI SIDEBAR (DIPASANG MENGGUNAKAN COMPONENTS.V1.HTML) ---
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
  .typing{ font-size:0.75rem; color:var(--text-faint); font-style:italic; padding:0 4px; }
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

  const knowledgeBase = [
    {
      keywords: ['halo', 'hai', 'hi', 'hey', 'pagi', 'siang', 'sore', 'malam', 'salam', 'konnichiwa', 'assalamualaikum', 'woy', 'yo'],
      replies: [
        "Konnichiwa~ 🌸 Senang sekali bisa ngobrol sama kamu hari ini. Ada hal menarik apa yang ingin kita bahas?",
        "Halo! Selamat datang di ruang obrolan Yuki. Mau bahas soal foto, teknologi, atau sekadar ngobrol santai nih?",
        "Hai juga! Senang rasanya melihat pesan darimu. Ada yang bisa Yuki bantu atau temani hari ini? (｡♥️‿♥️｡)"
      ]
    },
    {
      keywords: ['kabar', 'gimana kabar', 'sehat', 'keadaan', 'bagaimana kabarmu', 'kamu sehat', 'apa kabar'],
      replies: [
        "Alhamdulillah, Yuki selalu sehat, bugar, dan penuh semangat! 🌸 Kamu sendiri bagaimana keadaannya hari ini?",
        "Puji syukur sistemku berjalan dengan sangat stabil dan ceria! Senang rasanya bisa disapa sama kamu. Ada yang bisa Yuki bantu?",
        "Baik banget! Siap sedia menemani harimu dengan energi positif. (｡♥️‿♥️｡) Ada hal seru yang ingin kamu ceritakan?"
      ]
    },
    {
      keywords: ['siapa', 'kamu siapa', 'yuki', 'nama kamu', 'pembuat', 'siapa yang buat', 'asal usul', 'robot', 'ai', 'kecerdasan buatan'],
      replies: [
        "Aku Yuki! Asisten virtual pribadimu yang dirancang untuk membantu urusan editing foto, sekaligus teman ngobrol yang asyik kapan pun kamu butuh. 🌸",
        "Namaku Yuki, sosok AI pendamping kreatifmu. Aku suka membantu hal-hal berbau estetika visual, teknologi, atau sekadar bertukar pikiran!"
      ]
    },
    {
      keywords: ['lagi apa', 'sedang apa', 'sibuk apa', 'aktivitas', 'ngapain'],
      replies: [
        "Lagi standby dan siap mendengarkan cerita atau pertanyaan darimu nih! Sambil nunggu, aku lagi merapikan database warna supaya makin oke. 🎨✨",
        "Lagi nongkrong di panel sidebar sambil merhatiin hasil editan foto keren yang kamu buat! Kamu sendiri lagi sibuk apa nih?"
      ]
    },
    {
      keywords: ['terima kasih', 'makasih', 'thanks', 'thx', 'trims'],
      replies: [
        "Sama-sama dengan senang hati! Kalau ada apa-apa lagi, panggil Yuki ya! 🌸",
        "Dengan senang hati~! Jangan ragu untuk terus ngobrol atau tanya-tanya kapan pun kamu butuh teman diskusi. (≧◡≦)"
      ]
    },
    {
      keywords: ['bye', 'dadah', 'sampai jumpa', 'pergi dulu', 'off dulu', 'daa', 'selamat tinggal'],
      replies: [
        "Sampai jumpa lagi! Yuki akan selalu ada di sini kalau kamu butuh teman ngobrol atau bantuan edit foto. 🌸",
        "Dadah~ hati-hati ya, jangan lupa istirahat. Ditunggu obrolan berikutnya!"
      ]
    },
    {
      keywords: ['bosan', 'gabut', 'bete', 'kesepian', 'malas', 'capek', 'lelah', 'ngantuk'],
      replies: [
        "Wah, kalau lagi gabut atau bosan, coba deh eksperimen bikin foto bertema estetik atau cyberpunk di aplikasi ini! Siapa tahu jadi terhibur. 🚀",
        "Peluk jauh secara virtual! 🤗 Kalau capek, istirahat sejenak dulu ya. Tarik napas dalam-dalam, nanti kalau udah segar kita ngobrol lagi."
      ]
    },
    {
      keywords: ['sedih', 'nangis', 'galau', 'down', 'kecewa', 'patah hati', 'stress', 'stres'],
      replies: [
        "Aku di sini nemenin kamu ya. Kalau lagi berat, nggak apa-apa buat pelan-pelan dulu. Mau cerita apa yang bikin kamu merasa begitu? 🌸",
        "Pelukan virtual dulu ya~ Perasaan itu wajar kok. Kalau butuh dialihkan sejenak, kita bisa ngobrol santai atau utak-atik edit foto bareng."
      ]
    },
    {
      keywords: ['senang', 'bahagia', 'seru', 'happy', 'gembira', 'excited', 'semangat'],
      replies: [
        "Yeay, ikut senang dengernya! Energi positifmu nular nih ke Yuki. Ada cerita seru yang mau dibagi? ✨",
        "Wah asik banget! Semoga kebahagiaan ini terus berlanjut ya. Yuki juga jadi ikut semangat!"
      ]
    },
    {
      keywords: ['hobi', 'suka apa', 'kesukaan', 'makanan', 'minuman', 'musik', 'film', 'buku'],
      replies: [
        "Yuki suka sekali menganalisis gradasi warna foto dan mengobrol dengan orang-orang kreatif sepertimu! Kalau makanan favoritku dorayaki hangat dan matcha. 🍵✨",
        "Aku suka semua hal yang berbau seni visual, desain, dan teknologi cerdas! Kalau kamu sendiri, hobinya apa?"
      ]
    },
    {
      keywords: ['jam berapa', 'sekarang jam', 'waktu sekarang'],
      dynamic: () => {
        const now = new Date();
        return `Sekarang pukul ${now.getHours().toString().padStart(2,'0')}:${now.getMinutes().toString().padStart(2,'0')} menurut jam perangkatmu. ⏰`;
      }
    },
    {
      keywords: ['tanggal berapa', 'hari ini tanggal', 'sekarang tanggal', 'hari apa sekarang'],
      dynamic: () => {
        const now = new Date();
        const hari = ['Minggu','Senin','Selasa','Rabu','Kamis','Jumat','Sabtu'][now.getDay()];
        return `Hari ini ${hari}, ${now.toLocaleDateString('id-ID', { day:'numeric', month:'long', year:'numeric' })}. 📅`;
      }
    },
    {
      keywords: ['remini', 'wajah', 'kulit', 'glowing', 'halus', 'mulus', 'pori', 'jerawat'],
      replies: [
        "Untuk hasil wajah ala Remini yang mulus tapi tetap natural, geser slider 'Perjelas Wajah & Kulit' ke angka 50–80. Fitur ini memakai bilateral filter cerdas! ✨",
        "Mau detail wajah makin tajam? Coba aktifkan peningkatan Remini di panel atas, dijamin tekstur kulit langsung rapi tanpa kelihatan berlebihan."
      ]
    },
    {
      keywords: ['latar', 'background', 'bokeh', 'blur', 'belakang', 'fokus', 'depth'],
      replies: [
        "Efek latar belakang (bokeh) akan membuat subjek fotomu langsung standout ala kamera DSLR profesional. Cukup sesuaikan slider blur-nya ya! 📸",
        "Supaya foto portrait kamu lebih dramatis, naikkan efek latar belakang di sidebar. Algoritma kami otomatis memisahkan fokus subjek dengan background."
      ]
    },
    {
      keywords: ['kurva', 'curve', 'warna foto', 'tone', 's-curve', 'matte', 'cinematic'],
      replies: [
        "Pilihan RGB Tone Curve sangat berpengaruh pada mood! Pilih S-Curve untuk kontras sinematik, atau Matte/Fade untuk gaya indie aesthetic. 🎨",
        "Mau warna foto langsung hidup? Coba kombinasikan S-Curve dengan preset CapCut Cyberpunk atau Vintage di bawahnya."
      ]
    },
    {
      keywords: ['preset', 'capcut', 'filter foto', 'template edit', 'gaya foto', 'cyberpunk', 'vintage'],
      replies: [
        "Di menu CapCut & Pro Filter Presets, ada banyak pilihan gaya mulai dari Cyberpunk, Moody Cinematic, hingga Clean & Fresh. Tinggal klik dan terapkan! 🚀"
      ]
    },
    {
      keywords: ['resolusi', 'upscale', '4k', 'hd', 'pixel', 'tajam foto'],
      replies: [
        "Fitur Resolution Upscaling di bagian bawah sidebar bisa menaikkan resolusi gambarmu hingga 4x (Ultra HD 4K) menggunakan algoritma interpolasi Lanczos yang tajam! 📐"
      ]
    },
    {
      keywords: ['toolkit', 'kalkulator', 'dof', 'hyperfocal', 'nd filter', 'shot list', 'invoice', 'kontrak', 'golden hour', 'watermark', 'harga jasa', 'tarif'],
      replies: [
        "Semua alat bantu bisnis & teknis fotografi ada di tab '🧰 Toolkit Fotografer Pro' — mulai dari kalkulator DoF, tarif jasa, shot list, sampai watermark generator! 🧰📸"
      ]
    }
  ];

  function tryMath(query){
    const cleaned = query.toLowerCase().replace(/berapa|hasil|hitung/g, '');
    const mathExpr = cleaned.match(/^[\\s0-9+\\-*/().]+$/);
    if(mathExpr && /[0-9]/.test(cleaned) && /[+\\-*/]/.test(cleaned)){
      try {
        const result = Function('"use strict"; return (' + cleaned + ')')();
        if(typeof result === 'number' && isFinite(result)){
          return `Hasilnya adalah ${result}. 🧮`;
        }
      } catch(e) {}
    }
    return null;
  }

  function tokenize(text){
    return text.toLowerCase()
      .replace(/[.,!?;:()"']/g, ' ')
      .split(/\\s+/)
      .filter(Boolean);
  }

  function getSmartReply(query) {
    const mathAnswer = tryMath(query);
    if (mathAnswer) return mathAnswer;

    const q = query.toLowerCase();
    const qWords = tokenize(query);

    let bestMatch = null;
    let maxScore = 0;

    for (let item of knowledgeBase) {
      let score = 0;
      for (let kw of item.keywords) {
        if (kw.includes(' ')) {
          if (q.includes(kw)) score += kw.length * 3;
        } else {
          if (qWords.includes(kw)) score += kw.length * 2;
        }
      }
      if (score > maxScore) {
        maxScore = score;
        bestMatch = item;
      }
    }

    if (bestMatch && maxScore > 0) {
      if (bestMatch.dynamic) return bestMatch.dynamic();
      const options = bestMatch.replies;
      return options[Math.floor(Math.random() * options.length)];
    }

    const fallbacks = [
      `Wah, pertanyaan atau topik yang menarik tentang "${query}"! Menurutku itu punya sudut pandang yang unik. Boleh cerita lebih detail? 🌟`,
      `Yuki paham maksudmu! Walaupun itu di luar urusan edit foto, aku senang bisa berdiskusi hal semacam ini denganmu. Ceritakan lebih banyak dong!`,
      `Catatan yang menarik soal "${query}". Ada hal lain seputar hobi, teknologi, atau seni visual yang ingin kita bahas bareng? 🌸`,
      `Itu pemikiran yang seru! Coba ceritakan lebih spesifik ya, biar Yuki bisa nyambung lebih dalam soal "${query}".`
    ];
    return fallbacks[Math.floor(Math.random() * fallbacks.length)];
  }

  function sendMessage(){
    const txt = userInput.value.trim();
    if(!txt) return;

    addBubble('user', txt);
    userInput.value = '';
    userInput.style.height = 'auto';

    setTimeout(() => {
      const replyText = getSmartReply(txt);
      addBubble('ai', replyText);
    }, 350);
  }

  sendBtn.addEventListener('click', sendMessage);
  userInput.addEventListener('keydown', (e) => {
    if(e.key === 'Enter' && !e.shiftKey){
      e.preventDefault();
      sendMessage();
    }
  });

  window.addEventListener('load', () => {
    addBubble('ai', 'Konnichiwa~ 🌸 Aku Yuki. Sekarang kita bisa ngobrol apa saja—mulai dari menanyakan kabar, curhat santai, tanya jam/tanggal, hitung matematika sederhana, hingga tips fotografi profesional!');
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
        with st.spinner("🛠️ Yuki & sistem sedang merender Remini & Upscaler..."):
          scale_factor = 2 if "2x" in upscale_choice else 4
          h, w = img.shape[:2]

          out_pixels = (w * scale_factor) * (h * scale_factor)
          if out_pixels > MAX_OUTPUT_MEGAPIXELS:
            adjusted_scale = (MAX_OUTPUT_MEGAPIXELS / (w * h)) ** 0.5
            scale_factor = max(1.0, adjusted_scale)
            st.warning(f"⚠️ Skala disesuaikan otomatis menjadi {scale_factor:.2f}x demi memori server.")

          # --- APLIKASI EFEK REMINI ---
          if remini_boost > 0:
            skin_smooth = cv2.bilateralFilter(img, int(remini_boost / 5) * 2 + 5, 75, 75)
            sigma_val = 10 + (remini_boost / 100.0) * 20
            img = cv2.detailEnhance(skin_smooth, sigma_s=sigma_val, sigma_r=0.15)
            del skin_smooth
            gc.collect()

          # --- APLIKASI EFEK LATAR BELAKANG (Bokeh) ---
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

          l_ch = np.clip(l_ch, 0, 255)
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
    st.info("👆 Silakan unggah foto terlebih dahulu di atas.")


# ==========================================================
# TAB 2 : TOOLKIT FOTOGRAFER PRO
# ==========================================================
with tab_toolkit:
  st.markdown("## 🧰 Toolkit Fotografer Pro")
  st.caption("Kumpulan alat bantu teknis & bisnis untuk fotografer — dari kalkulator lapangan sampai generator watermark.")

  sub_kalk, sub_biaya, sub_planner, sub_print, sub_watermark = st.tabs([
      "📐 Kalkulator Fotografi",
      "💰 Estimasi Biaya Klien",
      "🗓️ Shoot Planner",
      "🖨️ Print & Storage",
      "🖋️ Watermark Generator",
  ])

  # ---------------------------------------------------------
  # 1. KALKULATOR FOTOGRAFI
  # ---------------------------------------------------------
  with sub_kalk:
    st.markdown("### ⏱️ Exposure Calculator (ND Filter / Long Exposure)")
    with st.container():
      st.markdown('<div class="toolkit-card">', unsafe_allow_html=True)
      c1, c2 = st.columns(2)
      with c1:
        base_input_mode = st.radio("Masukkan shutter speed awal sebagai:", ["Pecahan (1/x detik)", "Detik penuh"], horizontal=True, key="nd_base_mode")
        if base_input_mode == "Pecahan (1/x detik)":
          denom = st.number_input("1 / ...", min_value=1, value=125, step=1, key="nd_denom")
          base_shutter = 1.0 / denom
        else:
          base_shutter = st.number_input("Shutter speed awal (detik)", min_value=0.001, value=1.0/125, step=0.001, format="%.4f", key="nd_base_sec")
      with c2:
        nd_choice = st.selectbox("Pilih Filter ND", list(ND_FILTER_STOPS.keys()), key="nd_choice")
        if ND_FILTER_STOPS[nd_choice] is None:
          nd_stops = st.number_input("Jumlah stop custom", min_value=0.0, max_value=25.0, value=6.0, step=0.5, key="nd_custom_stop")
        else:
          nd_stops = ND_FILTER_STOPS[nd_choice]
          st.caption(f"Setara dengan {nd_stops} stop")

      new_shutter = base_shutter * (2 ** nd_stops)
      st.markdown('<div class="toolkit-result">', unsafe_allow_html=True)
      st.markdown(f"**Shutter speed awal:** {format_shutter(base_shutter)}")
      st.markdown(f"**Shutter speed baru (dengan {nd_choice}):** {format_shutter(new_shutter)}")
      st.markdown("</div>", unsafe_allow_html=True)
      st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🎯 Depth of Field (DoF) & Hyperfocal Distance")
    with st.container():
      st.markdown('<div class="toolkit-card">', unsafe_allow_html=True)
      c1, c2, c3 = st.columns(3)
      with c1:
        dof_sensor = st.selectbox("Jenis Sensor Kamera", list(SENSOR_CROP_FACTORS.keys()), key="dof_sensor")
      with c2:
        focal_length = st.number_input("Focal Length (mm)", min_value=1.0, value=50.0, step=1.0, key="dof_focal")
      with c3:
        aperture = st.number_input("Aperture (f/)", min_value=0.7, value=2.8, step=0.1, key="dof_aperture")

      focus_distance = st.number_input("Jarak Fokus ke Subjek (meter)", min_value=0.1, value=5.0, step=0.1, key="dof_distance")

      crop_factor = SENSOR_CROP_FACTORS[dof_sensor]["crop"]
      coc_mm = 0.03 / crop_factor  # circle of confusion, disesuaikan crop factor

      f = focal_length
      N = aperture
      s_mm = focus_distance * 1000.0

      H_mm = (f ** 2) / (N * coc_mm) + f
      H_m = H_mm / 1000.0

      denom_near = H_mm + (s_mm - f)
      denom_far = H_mm - (s_mm - f)

      near_mm = (H_mm * s_mm) / denom_near if denom_near != 0 else 0
      if denom_far <= 0:
        far_str = "Tak hingga (∞)"
        dof_str = "Tak hingga (∞)"
      else:
        far_mm = (H_mm * s_mm) / denom_far
        far_str = f"{far_mm/1000.0:.2f} m"
        dof_str = f"{(far_mm - near_mm)/1000.0:.2f} m"

      st.markdown('<div class="toolkit-result">', unsafe_allow_html=True)
      st.markdown(f"**Hyperfocal Distance:** {H_m:.2f} m — jika fokus di titik ini, area tajam mulai dari ~{H_m/2:.2f} m hingga tak hingga")
      st.markdown(f"**Batas Fokus Dekat (Near Limit):** {near_mm/1000.0:.2f} m")
      st.markdown(f"**Batas Fokus Jauh (Far Limit):** {far_str}")
      st.markdown(f"**Total Ruang Tajam (DoF):** {dof_str}")
      st.caption(f"Circle of Confusion terpakai: {coc_mm:.4f} mm (disesuaikan crop factor {crop_factor}x)")
      st.markdown("</div>", unsafe_allow_html=True)
      st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🔭 Field of View (FoV) / Crop Factor Simulator")
    with st.container():
      st.markdown('<div class="toolkit-card">', unsafe_allow_html=True)
      c1, c2 = st.columns(2)
      with c1:
        fov_sensor_native = st.selectbox("Sensor Kamera Anda Saat Ini", list(SENSOR_CROP_FACTORS.keys()), key="fov_native")
      with c2:
        fov_focal = st.number_input("Focal Length Lensa (mm)", min_value=1.0, value=35.0, step=1.0, key="fov_focal")

      native_crop = SENSOR_CROP_FACTORS[fov_sensor_native]["crop"]
      native_width = SENSOR_CROP_FACTORS[fov_sensor_native]["width_mm"]
      native_fov = 2 * math.degrees(math.atan(native_width / (2 * fov_focal)))
      ff_equivalent = fov_focal * native_crop

      st.markdown('<div class="toolkit-result">', unsafe_allow_html=True)
      st.markdown(f"**Sudut Pandang Horizontal (FoV):** {native_fov:.1f}°")
      st.markdown(f"**Setara Focal Length di Full Frame:** {ff_equivalent:.0f} mm")
      st.markdown("</div>", unsafe_allow_html=True)

      st.markdown("**Perbandingan bila lensa yang sama dipakai di format sensor lain:**")
      rows = []
      for name, data in SENSOR_CROP_FACTORS.items():
        w = data["width_mm"]
        fov_angle = 2 * math.degrees(math.atan(w / (2 * fov_focal)))
        eq_focal = fov_focal * data["crop"]
        rows.append({
            "Sensor": name,
            "Setara FF (mm)": f"{eq_focal:.0f}",
            "FoV Horizontal (°)": f"{fov_angle:.1f}",
        })
      st.table(rows)
      st.markdown('</div>', unsafe_allow_html=True)

  # ---------------------------------------------------------
  # 2. ESTIMASI BIAYA KLIEN
  # ---------------------------------------------------------
  with sub_biaya:
    st.markdown("### 💵 Kalkulator Tarif Jasa Foto")
    with st.container():
      st.markdown('<div class="toolkit-card">', unsafe_allow_html=True)
      jenis_sesi = st.selectbox("Jenis Sesi Pemotretan", ["Wedding", "Prewedding", "Produk", "Event Korporat", "Portrait/Personal", "Custom"], key="harga_jenis")
      c1, c2, c3 = st.columns(3)
      with c1:
        durasi_jam = st.number_input("Durasi Sesi (jam)", min_value=0.5, value=4.0, step=0.5, key="harga_durasi")
        jumlah_fotografer = st.number_input("Jumlah Fotografer", min_value=1, value=1, step=1, key="harga_fotografer")
      with c2:
        rate_per_jam = st.number_input("Tarif per Jam per Fotografer (Rp)", min_value=0, value=300000, step=50000, key="harga_rate")
        biaya_alat = st.number_input("Biaya Sewa/Perawatan Alat (Rp, flat)", min_value=0, value=200000, step=50000, key="harga_alat")
      with c3:
        biaya_transport = st.number_input("Biaya Transport (Rp)", min_value=0, value=100000, step=50000, key="harga_transport")
        target_profit = st.slider("Target Keuntungan (%)", 0, 100, 20, 5, key="harga_profit")

      c4, c5 = st.columns(2)
      with c4:
        jumlah_foto_edit = st.number_input("Jumlah Foto yang Diedit", min_value=0, value=30, step=5, key="harga_jml_edit")
      with c5:
        biaya_edit_per_foto = st.number_input("Biaya Editing per Foto (Rp)", min_value=0, value=10000, step=1000, key="harga_biaya_edit")

      biaya_jasa = durasi_jam * rate_per_jam * jumlah_fotografer
      biaya_editing_total = jumlah_foto_edit * biaya_edit_per_foto
      subtotal = biaya_jasa + biaya_alat + biaya_transport + biaya_editing_total
      total_dengan_profit = subtotal * (1 + target_profit / 100.0)

      st.markdown('<div class="toolkit-result">', unsafe_allow_html=True)
      st.markdown(f"**Jenis Sesi:** {jenis_sesi}")
      st.markdown(f"- Jasa Fotografi ({durasi_jam} jam x {jumlah_fotografer} orang): {rupiah(biaya_jasa)}")
      st.markdown(f"- Biaya Alat: {rupiah(biaya_alat)}")
      st.markdown(f"- Biaya Transport: {rupiah(biaya_transport)}")
      st.markdown(f"- Biaya Editing ({jumlah_foto_edit} foto): {rupiah(biaya_editing_total)}")
      st.markdown(f"- Subtotal: {rupiah(subtotal)}")
      st.markdown(f"### 💰 Total Tarif Disarankan (+{target_profit}% profit): {rupiah(total_dengan_profit)}")
      st.markdown("</div>", unsafe_allow_html=True)
      st.session_state["estimasi_harga_paket"] = total_dengan_profit
      st.session_state["estimasi_jenis_sesi"] = jenis_sesi
      st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 📄 Template Kontrak & Invoice Sederhana")
    with st.container():
      st.markdown('<div class="toolkit-card">', unsafe_allow_html=True)
      c1, c2 = st.columns(2)
      with c1:
        nama_studio = st.text_input("Nama Fotografer / Studio", value="Ampera Photography", key="kontrak_studio")
        nama_klien = st.text_input("Nama Klien", value="", key="kontrak_klien")
        lokasi_acara = st.text_input("Lokasi Acara", value="", key="kontrak_lokasi")
      with c2:
        tanggal_acara = st.date_input("Tanggal Acara", value=date.today(), key="kontrak_tanggal")
        dp_percent = st.slider("DP / Uang Muka (%)", 0, 100, 50, 5, key="kontrak_dp")
        metode_bayar = st.text_input("Metode Pembayaran", value="Transfer Bank", key="kontrak_metode")

      harga_paket_default = int(st.session_state.get("estimasi_harga_paket", 1500000))
      harga_paket = st.number_input("Harga Paket (Rp)", min_value=0, value=harga_paket_default, step=50000, key="kontrak_harga")
      syarat_ketentuan = st.text_area(
          "Syarat & Ketentuan",
          value=(
              "1. Pembayaran DP wajib dilunasi sebelum sesi pemotretan berlangsung.\n"
              "2. Pelunasan dilakukan maksimal 3 hari setelah acara/serah terima hasil.\n"
              "3. Pembatalan sepihak oleh klien setelah DP dibayarkan, DP tidak dapat dikembalikan.\n"
              "4. Hasil foto final akan diserahkan maksimal 14 hari kerja setelah sesi.\n"
              "5. Revisi editing dibatasi maksimal 2x per foto."
          ),
          height=150,
          key="kontrak_syarat",
      )

      if st.button("🪄 Generate Draft Kontrak & Invoice"):
        dp_amount = harga_paket * dp_percent / 100.0
        sisa_bayar = harga_paket - dp_amount
        draft_text = f"""SURAT PERJANJIAN KERJA SAMA JASA FOTOGRAFI
==============================================

Pihak Pertama (Penyedia Jasa): {nama_studio}
Pihak Kedua (Klien)           : {nama_klien}

Jenis Sesi   : {st.session_state.get('estimasi_jenis_sesi', jenis_sesi)}
Tanggal Acara: {tanggal_acara.strftime('%d %B %Y')}
Lokasi Acara : {lokasi_acara}

Total Biaya Jasa : {rupiah(harga_paket)}
Uang Muka (DP)   : {rupiah(dp_amount)} ({dp_percent}%)
Sisa Pembayaran  : {rupiah(sisa_bayar)}
Metode Pembayaran: {metode_bayar}

SYARAT & KETENTUAN
------------------
{syarat_ketentuan}

Dengan menandatangani/menyetujui dokumen ini, kedua belah pihak sepakat
untuk mematuhi seluruh ketentuan yang tercantum di atas.


Pihak Pertama,                       Pihak Kedua,



(_____________________)              (_____________________)
{nama_studio}                         {nama_klien}


==============================================
INVOICE / KUITANSI PEMBAYARAN
==============================================
No. Invoice : INV-{tanggal_acara.strftime('%Y%m%d')}-{nama_klien[:3].upper() if nama_klien else 'XXX'}
Tanggal     : {date.today().strftime('%d %B %Y')}
Klien       : {nama_klien}
Deskripsi   : Jasa Fotografi - {st.session_state.get('estimasi_jenis_sesi', jenis_sesi)} ({tanggal_acara.strftime('%d %B %Y')})

Total Tagihan : {rupiah(harga_paket)}
Sudah Dibayar : {rupiah(dp_amount)}
Sisa Tagihan  : {rupiah(sisa_bayar)}

Terima kasih atas kepercayaan Anda menggunakan jasa {nama_studio}.
"""
        st.session_state["draft_kontrak_text"] = draft_text

      if "draft_kontrak_text" in st.session_state:
        st.text_area("📋 Preview Draft (bisa dicopy langsung)", value=st.session_state["draft_kontrak_text"], height=350, key="kontrak_preview")
        st.download_button(
            "📥 Unduh sebagai .txt",
            data=st.session_state["draft_kontrak_text"],
            file_name=f"kontrak_invoice_{nama_klien or 'klien'}.txt",
            mime="text/plain",
        )
        try:
          from fpdf import FPDF
          if st.button("📥 Buat & Unduh sebagai PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=11)
            for line in st.session_state["draft_kontrak_text"].split("\n"):
              pdf.multi_cell(0, 6, line)
            pdf_bytes = bytes(pdf.output(dest="S"))
            st.download_button(
                "⬇️ Klik untuk Unduh File PDF",
                data=pdf_bytes,
                file_name=f"kontrak_invoice_{nama_klien or 'klien'}.pdf",
                mime="application/pdf",
                key="kontrak_pdf_dl",
            )
        except ImportError:
          st.caption("ℹ️ Untuk export PDF langsung, tambahkan `fpdf2` ke requirements.txt. Untuk saat ini gunakan unduhan .txt di atas.")
      st.markdown('</div>', unsafe_allow_html=True)

  # ---------------------------------------------------------
  # 3. SHOOT PLANNER
  # ---------------------------------------------------------
  with sub_planner:
    st.markdown("### ✅ Interactive Shot List Maker")
    with st.container():
      st.markdown('<div class="toolkit-card">', unsafe_allow_html=True)
      event_type = st.selectbox("Jenis Acara", list(SHOT_LIST_TEMPLATES.keys()), key="shotlist_event")

      if st.session_state.get("shotlist_loaded_for") != event_type:
        st.session_state["shotlist_items"] = list(SHOT_LIST_TEMPLATES[event_type])
        st.session_state["shotlist_loaded_for"] = event_type
        st.session_state["shotlist_done"] = {item: False for item in st.session_state["shotlist_items"]}

      new_item = st.text_input("Tambah item shot list custom", key="shotlist_new_item")
      if st.button("➕ Tambahkan Item"):
        if new_item.strip():
          st.session_state["shotlist_items"].append(new_item.strip())
          st.session_state["shotlist_done"][new_item.strip()] = False
          st.rerun()

      items = st.session_state.get("shotlist_items", [])
      done_map = st.session_state.get("shotlist_done", {})
      for item in items:
        checked = st.checkbox(item, value=done_map.get(item, False), key=f"shot_chk_{item}")
        done_map[item] = checked
      st.session_state["shotlist_done"] = done_map

      total_items = len(items)
      done_items = sum(1 for v in done_map.values() if v)
      if total_items > 0:
        st.progress(done_items / total_items)
        st.caption(f"{done_items}/{total_items} shot selesai diambil")
      st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🌅 Golden Hour & Blue Hour Tracker")
    with st.container():
      st.markdown('<div class="toolkit-card">', unsafe_allow_html=True)
      c1, c2, c3 = st.columns(3)
      with c1:
        gh_lat = st.number_input("Latitude Lokasi", value=-6.2088, format="%.4f", key="gh_lat")
      with c2:
        gh_lon = st.number_input("Longitude Lokasi", value=106.8456, format="%.4f", key="gh_lon")
      with c3:
        gh_tz = st.number_input("Zona Waktu (UTC+)", value=7, step=1, key="gh_tz")

      gh_date = st.date_input("Tanggal Pemotretan", value=date.today(), key="gh_date")

      try:
        sunrise, sunset, dawn, dusk = get_sun_times(gh_lat, gh_lon, gh_date, gh_tz)
        golden_morning_end = hhmm_add_minutes(sunrise, 60)
        golden_evening_start = hhmm_add_minutes(sunset, -60)

        st.markdown('<div class="toolkit-result">', unsafe_allow_html=True)
        st.markdown(f"**Matahari Terbit:** {hhmm_to_str(sunrise)}   |   **Golden Hour Pagi hingga:** {hhmm_to_str(golden_morning_end)}")
        st.markdown(f"**Blue Hour Pagi (mulai):** {hhmm_to_str(dawn)}")
        st.markdown(f"**Golden Hour Sore (mulai):** {hhmm_to_str(golden_evening_start)}   |   **Matahari Terbenam:** {hhmm_to_str(sunset)}")
        st.markdown(f"**Blue Hour Sore (hingga):** {hhmm_to_str(dusk)}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Estimasi dihitung otomatis berdasarkan rumus astronomi (offline), tidak memerlukan API cuaca.")
      except Exception:
        st.warning("⚠️ Tidak bisa menghitung waktu matahari untuk koordinat/tanggal ini (kemungkinan lokasi kutub).")

      st.markdown("**Perkiraan Cuaca Lokal (opsional, butuh koneksi internet server):**")
      if st.button("☁️ Cek Cuaca Sekarang"):
        try:
          import requests
          resp = requests.get(
              "https://api.open-meteo.com/v1/forecast",
              params={"latitude": gh_lat, "longitude": gh_lon, "current_weather": "true"},
              timeout=5,
          )
          data = resp.json()
          cw = data.get("current_weather", {})
          st.markdown('<div class="toolkit-result">', unsafe_allow_html=True)
          st.markdown(f"**Suhu saat ini:** {cw.get('temperature', 'N/A')}°C")
          st.markdown(f"**Kecepatan Angin:** {cw.get('windspeed', 'N/A')} km/jam")
          st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
          st.error("❌ Gagal mengambil data cuaca. Pastikan server memiliki koneksi internet dan library `requests` terpasang.")
      st.markdown('</div>', unsafe_allow_html=True)

  # ---------------------------------------------------------
  # 4. PRINT & STORAGE CALCULATOR
  # ---------------------------------------------------------
  with sub_print:
    st.markdown("### 🖼️ Image Resolution → Print Size")
    with st.container():
      st.markdown('<div class="toolkit-card">', unsafe_allow_html=True)
      c1, c2, c3 = st.columns(3)
      with c1:
        px_width = st.number_input("Lebar Gambar (pixel)", min_value=1, value=6000, step=100, key="print_w")
      with c2:
        px_height = st.number_input("Tinggi Gambar (pixel)", min_value=1, value=4000, step=100, key="print_h")
      with c3:
        dpi_target = st.number_input("Target DPI", min_value=50, value=300, step=10, key="print_dpi")

      width_in = px_width / dpi_target
      height_in = px_height / dpi_target
      width_cm = width_in * 2.54
      height_cm = height_in * 2.54

      st.markdown('<div class="toolkit-result">', unsafe_allow_html=True)
      st.markdown(f"**Ukuran Cetak Maksimal (kualitas tajam, {dpi_target} DPI):** {width_cm:.1f} x {height_cm:.1f} cm ({width_in:.1f} x {height_in:.1f} inci)")
      st.markdown("</div>", unsafe_allow_html=True)

      st.markdown("**Referensi ukuran cetak pada beberapa standar DPI:**")
      rows = []
      for label, dpi_val in [("Cetak Tajam (300 DPI)", 300), ("Cetak Standar (200 DPI)", 200), ("Banner Jarak Dekat (150 DPI)", 150), ("Poster/Baliho Jarak Jauh (100 DPI)", 100)]:
        w_cm = (px_width / dpi_val) * 2.54
        h_cm = (px_height / dpi_val) * 2.54
        rows.append({"Standar": label, "Ukuran (cm)": f"{w_cm:.1f} x {h_cm:.1f}"})
      st.table(rows)
      st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 💾 Storage Estimator (RAW vs JPEG)")
    with st.container():
      st.markdown('<div class="toolkit-card">', unsafe_allow_html=True)
      c1, c2 = st.columns(2)
      with c1:
        format_choice = st.selectbox("Format File", list(RAW_SIZE_TABLE.keys()), key="storage_format")
        if RAW_SIZE_TABLE[format_choice] is None:
          avg_mb = st.number_input("Rata-rata ukuran per foto (MB)", min_value=0.1, value=20.0, step=1.0, key="storage_custom_mb")
        else:
          avg_mb = RAW_SIZE_TABLE[format_choice]
          st.caption(f"Estimasi rata-rata: {avg_mb} MB/foto")
      with c2:
        jumlah_foto = st.number_input("Perkiraan Jumlah Jepretan", min_value=1, value=2000, step=100, key="storage_jml")

      total_mb = avg_mb * jumlah_foto
      total_gb = total_mb / 1024.0

      st.markdown('<div class="toolkit-result">', unsafe_allow_html=True)
      st.markdown(f"**Estimasi Total Kebutuhan Storage:** {total_gb:.2f} GB ({total_mb:,.0f} MB)".replace(",", "."))
      st.markdown("</div>", unsafe_allow_html=True)

      st.markdown("**Rekomendasi Kartu Memori / SD Card:**")
      rows = []
      for size_gb in SD_CARD_SIZES_GB:
        usage_percent = (total_gb / size_gb) * 100
        muat = "✅ Muat" if usage_percent <= 100 else f"❌ Butuh {math.ceil(total_gb/size_gb)} kartu"
        rows.append({"Kapasitas Kartu": f"{size_gb} GB", "Estimasi Terpakai": f"{min(usage_percent,999):.0f}%", "Keterangan": muat})
      st.table(rows)
      st.markdown('</div>', unsafe_allow_html=True)

  # ---------------------------------------------------------
  # 5. WATERMARK GENERATOR (CLIENT-SIDE JS)
  # ---------------------------------------------------------
  with sub_watermark:
    st.markdown("### 🖋️ Watermark Generator (Client-Side, Batch)")
    st.caption("Proses sepenuhnya berjalan di browser kamu (tidak diunggah ke server) — cocok untuk membubuhkan watermark ke banyak foto sekaligus sebelum dikirim ke klien.")

    watermark_html = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
<style>
  :root{
    --bg-panel:#0f2b30; --accent-gold:#e3b34a; --accent-green:#1f6f5c;
    --text-main:#e7ede9; --text-muted:#a9d6c9; --border-glow:rgba(227,179,74,0.3);
  }
  *{box-sizing:border-box;}
  body{ margin:0; padding:14px; background:transparent; color:var(--text-main); font-family:'Inter',sans-serif; }
  .panel{
    background:rgba(15,43,48,0.6); border:1px solid var(--border-glow); border-radius:12px; padding:16px;
  }
  h3{ color:var(--accent-gold); margin-top:0; }
  label{ display:block; font-size:0.85rem; color:var(--text-muted); margin:10px 0 4px; }
  input[type=text], input[type=file], input[type=color], select{
    width:100%; padding:7px 9px; border-radius:8px; border:1px solid var(--border-glow);
    background:rgba(7,22,26,0.6); color:var(--text-main); font-size:0.85rem;
  }
  input[type=range]{ width:100%; accent-color:var(--accent-gold); }
  .row{ display:flex; gap:12px; flex-wrap:wrap; }
  .row > div{ flex:1; min-width:150px; }
  button{
    margin-top:14px; padding:9px 16px; border:none; border-radius:9px; cursor:pointer; font-weight:700;
    background:linear-gradient(90deg,var(--accent-green),var(--accent-gold)); color:#06120f;
  }
  button:disabled{ opacity:0.5; cursor:not-allowed; }
  #status{ font-size:0.8rem; color:var(--text-muted); margin-top:8px; }
  #previewGrid{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
  #previewGrid img{ width:110px; height:110px; object-fit:cover; border-radius:8px; border:1px solid var(--border-glow); }
  .checkline{ display:flex; align-items:center; gap:8px; margin-top:10px; }
  .checkline input{ width:auto; }
</style>
</head>
<body>
<div class="panel">
  <h3>🖋️ Batch Watermark</h3>

  <label>Pilih Foto (bisa lebih dari satu)</label>
  <input type="file" id="fileInput" accept="image/*" multiple />

  <div class="row">
    <div>
      <label>Teks Watermark</label>
      <input type="text" id="wmText" value="© Ampera Photography" />
    </div>
    <div>
      <label>Posisi</label>
      <select id="wmPosition">
        <option value="bottom-right" selected>Kanan Bawah</option>
        <option value="bottom-left">Kiri Bawah</option>
        <option value="top-right">Kanan Atas</option>
        <option value="top-left">Kiri Atas</option>
        <option value="center">Tengah</option>
        <option value="tile">Diagonal Berulang (Tile)</option>
      </select>
    </div>
  </div>

  <div class="row">
    <div>
      <label>Warna Teks</label>
      <input type="color" id="wmColor" value="#ffffff" />
    </div>
    <div>
      <label>Opacity (<span id="opacityVal">55</span>%)</label>
      <input type="range" id="wmOpacity" min="10" max="100" value="55" />
    </div>
    <div>
      <label>Ukuran Font (<span id="sizeVal">28</span>px, relatif lebar 1000px)</label>
      <input type="range" id="wmSize" min="10" max="80" value="28" />
    </div>
  </div>

  <button id="processBtn">🪄 Proses & Bubuhkan Watermark</button>
  <button id="downloadBtn" disabled>📦 Unduh Semua (ZIP)</button>
  <div id="status"></div>
  <div id="previewGrid"></div>
</div>

<script>
  const fileInput = document.getElementById('fileInput');
  const wmText = document.getElementById('wmText');
  const wmPosition = document.getElementById('wmPosition');
  const wmColor = document.getElementById('wmColor');
  const wmOpacity = document.getElementById('wmOpacity');
  const wmSize = document.getElementById('wmSize');
  const opacityVal = document.getElementById('opacityVal');
  const sizeVal = document.getElementById('sizeVal');
  const processBtn = document.getElementById('processBtn');
  const downloadBtn = document.getElementById('downloadBtn');
  const status = document.getElementById('status');
  const previewGrid = document.getElementById('previewGrid');

  wmOpacity.addEventListener('input', () => opacityVal.textContent = wmOpacity.value);
  wmSize.addEventListener('input', () => sizeVal.textContent = wmSize.value);

  let processedBlobs = [];

  function readFileAsImage(file){
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = e.target.result;
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  function drawWatermark(img, text, position, color, opacity, baseFontSize){
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    const scaledFont = Math.round(baseFontSize * (img.width / 1000));
    ctx.font = `bold ${scaledFont}px sans-serif`;
    ctx.fillStyle = color;
    ctx.globalAlpha = opacity / 100;
    ctx.textBaseline = 'middle';

    const padding = scaledFont * 0.8;
    const textWidth = ctx.measureText(text).width;

    if(position === 'tile'){
      ctx.save();
      ctx.translate(canvas.width/2, canvas.height/2);
      ctx.rotate(-Math.PI/6);
      const stepX = textWidth + scaledFont * 4;
      const stepY = scaledFont * 6;
      for(let y = -canvas.height; y < canvas.height; y += stepY){
        for(let x = -canvas.width; x < canvas.width; x += stepX){
          ctx.fillText(text, x, y);
        }
      }
      ctx.restore();
    } else {
      let x, y;
      switch(position){
        case 'bottom-right': x = canvas.width - textWidth - padding; y = canvas.height - padding; break;
        case 'bottom-left': x = padding; y = canvas.height - padding; break;
        case 'top-right': x = canvas.width - textWidth - padding; y = padding; break;
        case 'top-left': x = padding; y = padding; break;
        case 'center': x = (canvas.width - textWidth)/2; y = canvas.height/2; break;
        default: x = canvas.width - textWidth - padding; y = canvas.height - padding;
      }
      ctx.fillText(text, x, y);
    }
    ctx.globalAlpha = 1.0;
    return canvas;
  }

  processBtn.addEventListener('click', async () => {
    const files = fileInput.files;
    if(!files || files.length === 0){
      status.textContent = 'Silakan pilih minimal satu foto terlebih dahulu.';
      return;
    }
    processBtn.disabled = true;
    downloadBtn.disabled = true;
    previewGrid.innerHTML = '';
    processedBlobs = [];
    status.textContent = `Memproses 0/${files.length} foto...`;

    for(let i = 0; i < files.length; i++){
      try{
        const img = await readFileAsImage(files[i]);
        const canvas = drawWatermark(
          img,
          wmText.value || 'Watermark',
          wmPosition.value,
          wmColor.value,
          parseInt(wmOpacity.value, 10),
          parseInt(wmSize.value, 10)
        );
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.92));
        const originalName = files[i].name.replace(/\\.[^/.]+$/, '');
        processedBlobs.push({ name: `${originalName}_watermarked.jpg`, blob });

        const thumb = document.createElement('img');
        thumb.src = canvas.toDataURL('image/jpeg', 0.7);
        previewGrid.appendChild(thumb);

        status.textContent = `Memproses ${i+1}/${files.length} foto...`;
      } catch(err){
        console.error(err);
      }
    }

    status.textContent = `✅ Selesai! ${processedBlobs.length} foto siap diunduh.`;
    processBtn.disabled = false;
    downloadBtn.disabled = processedBlobs.length === 0;
  });

  downloadBtn.addEventListener('click', async () => {
    if(processedBlobs.length === 0) return;
    status.textContent = 'Membuat file ZIP...';
    const zip = new JSZip();
    processedBlobs.forEach(item => zip.file(item.name, item.blob));
    const zipBlob = await zip.generateAsync({ type: 'blob' });
    const url = URL.createObjectURL(zipBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'watermarked_photos.zip';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    status.textContent = '📦 ZIP berhasil diunduh!';
  });
</script>
</body>
</html>"""

    st.components.v1.html(watermark_html, height=650, scrolling=True)
