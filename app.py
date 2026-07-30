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

  st.markdown("### 🎬 10+ Pro Filter Presets")
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
      ],
  )

  if st.button("🪄 Terapkan Preset Pilihan"):
    if capcut_preset.startswith("✨ Cyberpunk"):
      st.session_state.update({"exposure": 0.2, "contrast": 25, "highlights": -10, "shadows": 15, "temp": -15, "tint": 15, "vibrance": 35, "saturation": 20, "clarity": 25, "vignette": 40})
    elif capcut_preset.startswith("🎞️ Vintage"):
      st.session_state.update({"exposure": 0.1, "contrast": 10, "highlights": -20, "shadows": 30, "temp": 25, "tint": -5, "vibrance": -10, "saturation": -5, "clarity": 10, "vignette": 50})
    elif capcut_preset.startswith("🎬 Moody"):
      st.session_state.update({"exposure": -0.3, "contrast": 35, "highlights": -40, "shadows": -20, "temp": -10, "tint": 5, "vibrance": 10, "saturation": 5, "clarity": 30, "vignette": 65})
    elif capcut_preset.startswith("🌟 Clean"):
      st.session_state.update({"exposure": 0.3, "contrast": 15, "highlights": 10, "shadows": 25, "temp": 0, "tint": 0, "vibrance": 20, "saturation": 15, "clarity": 15, "vignette": 10})
    elif capcut_preset.startswith("☕ Warm Portrait"):
      st.session_state.update({"exposure": 0.1, "contrast": 5, "highlights": 10, "shadows": 20, "temp": 15, "tint": 5, "vibrance": 15, "saturation": 5, "clarity": 5, "vignette": 15})
    elif capcut_preset.startswith("🖤 Dramatic"):
      st.session_state.update({"exposure": 0.0, "contrast": 40, "highlights": -30, "shadows": -30, "temp": 0, "tint": 0, "vibrance": -50, "saturation": -50, "clarity": 35, "vignette": 50})
    elif capcut_preset.startswith("🌅 Golden Hour"):
      st.session_state.update({"exposure": 0.2, "contrast": 15, "highlights": 5, "shadows": 20, "temp": 30, "tint": 10, "vibrance": 25, "saturation": 15, "clarity": 10, "vignette": 20})
    elif capcut_preset.startswith("🌲 Emerald Forest"):
      st.session_state.update({"exposure": -0.1, "contrast": 20, "highlights": -10, "shadows": 10, "temp": -10, "tint": -20, "vibrance": 30, "saturation": 20, "clarity": 20, "vignette": 30})
    elif capcut_preset.startswith("🧊 Arctic Frost"):
      st.session_state.update({"exposure": 0.1, "contrast": 10, "highlights": 15, "shadows": 10, "temp": -35, "tint": 10, "vibrance": 15, "saturation": 5, "clarity": 15, "vignette": 15})
    elif capcut_preset.startswith("🍑 Peach Blossom"):
      st.session_state.update({"exposure": 0.2, "contrast": 5, "highlights": 20, "shadows": 25, "temp": 10, "tint": 15, "vibrance": 20, "saturation": 10, "clarity": 5, "vignette": 10})
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
      keywords: ['kecewa', 'sedih', 'galau', 'stress'],
      replies: [
        "Cup cup, jangan bersedih ya... Hidup memang punya naik turunnya. Tarik napas dalam-dalam, tersenyumlah, dan ingat Yuki selalu ada buat nemenin kamu. 🫂💖"
      ]
    },
    {
      keywords: ['hebat', 'keren', 'mantap', 'cool', 'gokil'],
      replies: [
        "Terima kasih banyak! Dukungan dan apresiasi kamu adalah bahan bakar utama bagi Yuki untuk terus aktif melayani. 🚀✨"
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
    return "Wah, pertanyaan yang sangat menarik! Yuki akan ingat itu baik-baik. Ada lagi seputar editing foto atau hal lain yang ingin kita diskusikan? 🌸";
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
            <h4 style="color: #e3b34a; margin-top: 0;">⚡ Selective & Layer Pro</h4>
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
