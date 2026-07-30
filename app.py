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

# ==========================================================
# HEADER UTAMA & TOMBOL SIMPAN 4K DI POJOK KANAN ATAS
# ==========================================================
top_col1, top_col2, top_col3 = st.columns([0.6, 3.4, 2])

with top_col1:
  try:
    st.image(LOGO_PATH, use_container_width=True)
  except Exception:
    st.markdown("<h1 style='margin:0;'>👾</h1>", unsafe_allow_html=True)

with top_col2:
  st.title("AMPER.AI — Professional Suite")
  st.markdown(
      "<p style='color: #a9d6c9; font-size: 0.95em;'>Platform pengolahan foto pro lengkap dengan efek perjelas wajah & bokeh.</p>",
      unsafe_allow_html=True,
  )

with top_col3:
  st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
  st.markdown("<p style='color: #e3b34a; font-size: 0.85em; margin-bottom: 4px; font-weight: bold;'>✨ Quick Export 4K</p>", unsafe_allow_html=True)
  
  if "processed_img" in st.session_state:
    result_pil = Image.fromarray(st.session_state["processed_img"])
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG", quality=95)
    byte_im = buf.getvalue()
    
    st.download_button(
        label="📥 Unduh 4K / HD",
        data=byte_im,
        file_name="amper_ai_pro_4k.jpg",
        mime="image/jpeg",
        use_container_width=True,
    )
  else:
    st.button("📥 Unduh 4K (Belum Ada Hasil)", disabled=True, use_container_width=True)
  st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

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
# LAYOUT UTAMA: PISAHKAN EDITOR FOTO (KIRI) & YUKI AI (KANAN)
# ==========================================================
main_col_editor, main_col_yuki = st.columns([1.5, 1])

with main_col_editor:
  if uploaded_file is not None and img is not None:
    sub_c1, sub_c2 = st.columns(2)
    with sub_c1:
      st.subheader("🎆 Foto Asli")
      st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with sub_c2:
      st.subheader("🎇 Hasil Upscaller & Pro")
      if "processed_img" in st.session_state:
        st.image(st.session_state["processed_img"], use_container_width=True)
      else:
        st.info("Klik tombol render di bawah untuk melihat hasil.")
  else:
    st.info("👈 Silakan unggah foto di atas untuk mulai melihat perbandingan.")

with main_col_yuki:
  st.subheader("🌸 Yuki-Chan AI Companion")
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
    height:460px;
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
    padding:8px 12px;
    background:rgba(37,26,61,0.85);
    border-bottom:1px solid var(--border-glow);
  }
  .avatar{
    width:32px; height:32px; border-radius:50%; flex-shrink:0;
    background:radial-gradient(circle at 35% 30%, #ffe3ee 0%, #ff9dc0 45%, #7a4fa8 100%);
    position:relative; overflow:hidden;
  }
  .avatar svg{ width:100%; height:100%; display:block; }
  .id-name{ font-weight:700; font-size:0.85rem; color:var(--accent-pink-soft); }
  .id-role{ font-size:0.6rem; color:var(--text-muted); }

  main{
    flex:1; overflow-y:auto; padding:10px; display:flex; flex-direction:column; gap:10px;
  }
  main::-webkit-scrollbar{ width:4px; }
  main::-webkit-scrollbar-thumb{ background:rgba(255,122,168,0.3); border-radius:4px; }

  .row{ display:flex; gap:8px; max-width:100%; align-items:flex-end; }
  .row.user{ flex-direction:row-reverse; }
  .bubble{
    max-width:82%; padding:8px 12px; border-radius:12px; font-size:0.8rem; line-height:1.4;
    word-wrap:break-word; white-space:pre-wrap;
  }
  .row.ai .bubble{
    background:var(--bg-bubble-ai); border:1px solid rgba(255,202,107,0.18); border-bottom-left-radius:2px;
  }
  .row.user .bubble{
    background:var(--bg-bubble-user); border:1px solid rgba(127,233,220,0.2); border-bottom-right-radius:2px; color:#f3ecff;
  }
  .tag{ display:block; font-size:0.55rem; color:var(--accent-pink-soft); margin-bottom:2px; }

  .dialogue-wrap{ padding:6px 8px; background:rgba(28,19,48,0.9); border-top:1px solid var(--border-glow); }
  .dialogue-box{ display:flex; align-items:flex-end; gap:6px; background:rgba(37,26,61,0.9); border:1px solid var(--border-glow); border-radius:8px; padding:4px 6px; }
  #userInput{
    flex:1; resize:none; background:transparent; border:none; outline:none; color:var(--text-main); font-size:0.8rem; max-height:50px;
  }
  #userInput::placeholder{ color:var(--text-faint); }
  #sendBtn{
    width:30px; height:30px; border-radius:50%; border:none; cursor:pointer;
    background:linear-gradient(135deg, var(--accent-pink), #c85f92); color:#fff;
    display:flex; align-items:center; justify-content:center;
  }
  #sendBtn svg{ width:12px; height:12px; }
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
      <div class="id-name">Yuki AI</div>
      <div class="id-role">Teman Ngobrol & Asisten Kreatif</div>
    </div>
  </header>

  <main id="chatArea"></main>

  <div class="dialogue-wrap">
    <div class="dialogue-box">
      <textarea id="userInput" rows="1" placeholder="Tanya Yuki..."></textarea>
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
      keywords: ['halo', 'hai', 'hi', 'pagi', 'siang', 'sore', 'malam', 'konnichiwa'],
      replies: ["Konnichiwa~ 🌸 Ada yang bisa Yuki bantu untuk editan fotomu hari ini?"]
    },
    {
      keywords: ['siapa', 'kamu', 'yuki', 'nama'],
      replies: ["Aku Yuki! Asisten AI yang siap menemani kamu menggunakan AMPER.AI. ✨"]
    },
    {
      keywords: ['terima kasih', 'makasih', 'thanks', 'arigatou'],
      replies: ["Sama-sama! 🌸 Senang bisa membantu proses kreatifmu."]
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
    return "Wah menarik sekali! Ada hal lain seputar editing foto yang ingin kita diskusikan? 🌸";
  }

  function sendMessage(){
    const txt = userInput.value.trim();
    if(!txt) return;
    addBubble('user', txt);
    userInput.value = '';
    setTimeout(() => {
      addBubble('ai', findReply(txt));
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
    addBubble('ai', 'Konnichiwa~ 🌸 Silakan atur parameter foto di bawah atau ngobrol dengan Yuki di sini!');
  });
</script>
</body>
</html>"""
  
  # Perbaikan ada di baris ini (menghapus parameter scrolling yang tidak valid/bermasalah)
  components.html(yuki_html, height=485)
st.markdown("---")

# ==========================================================
# PANEL KONTROL & PENGATURAN FOTO DI BAGIAN BAWAH
# ==========================================================
st.markdown("### 🎛️ Panel Kontrol & Pengaturan Pro (Di Bawah)")

with st.expander("🎭 Face, Body & Background Retouch", expanded=True):
  col_eb1, col_eb2, col_eb3 = st.columns(3)
  with col_eb1:
    remini_boost = st.slider("Perjelas Wajah & Kulit (Remini Effect)", 0, 100, 0, 1)
  with col_eb2:
    body_slim = st.slider("Body Slimming & Contour Pro", 0, 100, 0, 1)
  with col_eb3:
    bg_blur = st.slider("Efek Latar Belakang (Bokeh / Blur)", 0, 100, 0, 2)

with st.expander("📍 Selective Edit & Layer Blending"):
  col_sl1, col_sl2 = st.columns(2)
  with col_sl1:
    enable_selective = st.checkbox("Aktifkan Selective Control Point")
    sel_x_pct = st.slider("Titik Kontrol X (%)", 0, 100, 50, 1)
    sel_y_pct = st.slider("Titik Kontrol Y (%)", 0, 100, 50, 1)
    sel_radius = st.slider("Radius Area", 20, 300, 100, 5)
    sel_exposure = st.slider("Exposure Khusus Area", -1.0, 1.0, 0.0, 0.1)
    sel_sat = st.slider("Saturasi Khusus Area", -50, 50, 0, 1)
  with col_sl2:
    enable_layer = st.checkbox("Aktifkan Gabung Layer Kedua")
    layer_file = st.file_uploader("Unggah Foto Layer Overlay", type=["jpg", "jpeg", "png"], key="layer_uploader")
    layer_opacity = st.slider("Opacity Layer", 0.0, 1.0, 0.5, 0.05)
    layer_mode = st.selectbox("Mode Blending", ["Normal", "Overlay", "Screen", "Multiply"])

with st.expander("🎨 10+ Pro Filter Presets & Tone Curve"):
  col_fc1, col_fc2 = st.columns(2)
  with col_fc1:
    curve_preset = st.selectbox(
        "Pilih Kurva Pencahayaan",
        [
            "Linear (Standard)",
            "S-Curve (Kontras Tinggi & Sinematik)",
            "Matte / Fade (Gaya Film Indie)",
            "Bright Pop (Terang & Segar)",
        ],
    )
  with col_fc2:
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
      st.session_state.update({"exposure": 0.2, "contrast": 25, "highlights": -10, "shadows": 15, "clarity": 25})
    elif capcut_preset.startswith("🎞️ Vintage"):
      st.session_state.update({"exposure": 0.1, "contrast": 10, "highlights": -20, "shadows": 30, "clarity": 10})
    elif capcut_preset.startswith("🎬 Moody"):
      st.session_state.update({"exposure": -0.3, "contrast": 35, "highlights": -40, "shadows": -20, "clarity": 30})
    st.rerun()

with st.expander("☀️ Light, Color & Detail Adjustments", expanded=True):
  col_adj1, col_adj2, col_adj3 = st.columns(3)
  with col_adj1:
    st.markdown("#### Light")
    exposure = st.slider("Exposure", -2.0, 2.0, 0.0, 0.1, key="exposure")
    contrast = st.slider("Contrast", -50, 50, 10, 1, key="contrast")
    highlights = st.slider("Highlights", -100, 100, -20, 1, key="highlights")
    shadows = st.slider("Shadows", -100, 100, 25, 1, key="shadows")
  with col_adj2:
    st.markdown("#### Color")
    temp = st.slider("Temperature", -50, 50, -5, 1, key="temp")
    tint = st.slider("Tint", -50, 50, 0, 1, key="tint")
    vibrance = st.slider("Vibrance", -50, 50, 15, 1, key="vibrance")
    saturation = st.slider("Saturation", -50, 50, 10, 1, key="saturation")
  with col_adj3:
    st.markdown("#### Detail & Upscale")
    clarity = st.slider("Clarity", -50, 50, 20, 1, key="clarity")
    sharpen = st.slider("Sharpening HD", 0, 100, 30, 1, key="sharpen")
    denoise_strength = st.slider("Noise Reduction", 0, 30, 0, 1, key="noise_reduction")
    upscale_choice = st.selectbox("Resolution Upscaling", ["2x (HD 2K)", "4x (Ultra HD 4K)"], index=0, key="upscale_choice")

st.markdown("<br>", unsafe_transactions=True if "unsafetransactions" in globals() else None)
process_btn = st.button("⬆️ Terapkan & Render Instan", use_container_width=True)

# ---------------- Proses Eksekusi Gambar ----------------
if uploaded_file is not None and img is not None and process_btn:
  try:
    with st.spinner("🛠️ Yuki & sistem sedang merender proses foto..."):
      scale_factor = 2 if "2x" in upscale_choice else 4
      h, w = img.shape[:2]

      out_pixels = (w * scale_factor) * (h * scale_factor)
      if out_pixels > MAX_OUTPUT_MEGAPIXELS:
        adjusted_scale = (MAX_OUTPUT_MEGAPIXELS / (w * h)) ** 0.5
        scale_factor = max(1.0, adjusted_scale)

      if body_slim > 0:
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

      if enable_layer and layer_file is not None:
        layer_bytes = np.asarray(bytearray(layer_file.read()), dtype=np.uint8)
        layer_img = cv2.imdecode(layer_bytes, cv2.IMREAD_COLOR)
        if layer_img is not None:
          layer_resized = cv2.resize(layer_img, (img.shape[1], img.shape[0]))
          if layer_mode == "Normal":
            img = cv2.addWeighted(img, 1.0 - layer_opacity, layer_resized, layer_opacity, 0)
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
      st.success("✅ Berhasil merender foto! Silakan unduh melalui tombol di pojok kanan atas.")
      st.rerun()

  except Exception as e:
      st.error("❌ Opss..Terjadi kesalahan saat memproses gambar.")
      with st.expander("Detail teknis error"):
        st.exception(e)
