import base64
import gc
import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Untuk uji coba pertama: sistem login & kredit dimatikan dulu.
# Kalau sudah siap ditawarkan/dijual, ganti jadi True lagi — dan pastikan
# auth.py sudah diupload ke repo + requirements.txt sudah berisi
# supabase & bcrypt, baru redeploy.
REQUIRE_LOGIN = False

if REQUIRE_LOGIN:
  from auth import render_auth_sidebar, get_credits, deduct_credit

st.set_page_config(
    page_title="AMPER.AI - Pro Suite", page_icon="😈", layout="wide"
)

# ==========================================================
# Nama file aset — letakkan logo_amper.png & bg_amper.jpg
# di folder yang SAMA dengan app.py di repo GitHub kamu
# ==========================================================
LOGO_PATH = "logo_amper.png"
BG_PATH = "bg_amper.jpg"

# Sisi terpanjang foto ASLI akan diturunkan ke ukuran ini dulu
# sebelum diedit, supaya tidak membebani server sejak awal
MAX_INPUT_DIM = 3000

# Batas aman total pixel HASIL AKHIR (setelah upscaling)
MAX_OUTPUT_MEGAPIXELS = 20_000_000

# Untuk uji coba pertama: sistem login & kredit dimatikan dulu.
# Kalau sudah siap ditawarkan/dijual, ganti jadi True lagi.
REQUIRE_LOGIN = False


def get_base64_of_bin_file(path):
  with open(path, "rb") as f:
    return base64.b64encode(f.read()).decode()


def set_background(image_path):
  """Pasang bg_amper.jpg sebagai background. Jika file belum ada,
  aplikasi tetap jalan normal dengan gradient default (tidak crash)."""
  try:
    bin_str = get_base64_of_bin_file(image_path)
    css = f"""
      <style>
      .stApp {{
          background-image: linear-gradient(160deg, rgba(6,17,20,0.93), rgba(6,17,20,0.93)),
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
  """Analisis histogram sederhana untuk menyarankan nilai awal slider
  pencahayaan & ketajaman, supaya foto kurang terang/kurang jelas bisa
  otomatis dikoreksi tanpa perlu diatur manual dulu."""
  gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
  mean_brightness = float(np.mean(gray))
  contrast_std = float(np.std(gray))
  laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

  # --- Exposure: target kecerahan rata-rata sekitar 125 dari 255 ---
  target_brightness = 125.0
  diff = target_brightness - mean_brightness
  suggested_exposure = float(np.clip(diff / 90.0, -1.2, 1.2))

  # --- Contrast: kalau std rendah (foto flat/berkabut), naikkan contrast ---
  suggested_contrast = int(np.clip((45 - contrast_std) * 1.1, 0, 40))

  # --- Highlights/Shadows: deteksi area yang ke-clip gelap/terang ---
  hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
  total_px = gray.size
  shadow_clip_ratio = hist[:15].sum() / total_px
  highlight_clip_ratio = hist[240:].sum() / total_px
  suggested_shadows = int(np.clip(shadow_clip_ratio * 400, 0, 60))
  suggested_highlights = int(np.clip(-highlight_clip_ratio * 400, -60, 0))

  # --- Clarity/Sharpen: laplacian variance rendah = foto kurang tajam ---
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


# =====================================================================
# ======================  FITUR AI PHOTO TOOLS  ======================
# Semua fungsi di bawah ini SENGAJA dibuat tanpa model AI berat
# (tidak ada download GFPGAN/Stable Diffusion dkk). Alasannya: server
# gratis Streamlit Cloud biasanya cuma punya RAM terbatas & tanpa GPU,
# jadi model besar begitu sering bikin app crash / OOM saat deploy.
# Semua fungsi ini pakai algoritma bawaan OpenCV (photo module) yang
# ringan tapi hasilnya tetap terlihat "AI-ish" & profesional.
# =====================================================================


def apply_ai_photo_enhancer(img_bgr):
  """'AI Photo Enhancer' 1-klik: gabungan auto white balance (gray-world),
  auto tone dari compute_auto_suggestions, dan sedikit local contrast,
  supaya foto langsung terlihat lebih hidup tanpa perlu atur slider."""
  # Auto white balance sederhana (gray-world assumption)
  result = img_bgr.astype("float32")
  avg_b, avg_g, avg_r = [result[:, :, i].mean() for i in range(3)]
  avg_gray = (avg_b + avg_g + avg_r) / 3.0
  avg_b, avg_g, avg_r = max(avg_b, 1e-3), max(avg_g, 1e-3), max(avg_r, 1e-3)
  result[:, :, 0] *= avg_gray / avg_b
  result[:, :, 1] *= avg_gray / avg_g
  result[:, :, 2] *= avg_gray / avg_r
  result = np.clip(result, 0, 255).astype("uint8")

  # Local contrast (CLAHE) di channel L supaya detail midtone lebih pop
  lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
  l_ch, a_ch, b_ch = cv2.split(lab)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  l_ch = clahe.apply(l_ch)
  result = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

  # Sedikit sharpening akhir supaya terlihat "diproses AI"
  gaussian = cv2.GaussianBlur(result, (0, 0), 2)
  result = cv2.addWeighted(result, 1.25, gaussian, -0.25, 0)
  return result


def apply_face_enhancer(img_bgr):
  """Deteksi wajah (Haar Cascade bawaan OpenCV), lalu perhalus kulit
  dengan bilateral filter (menjaga tepi mata/hidung/bibir) dan
  pertajam detail wajah dengan unsharp mask lokal."""
  face_cascade = cv2.CascadeClassifier(
      cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
  )
  gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(
      gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50)
  )

  result = img_bgr.copy()
  for (x, y, w, h) in faces:
    pad = int(0.2 * w)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img_bgr.shape[1], x + w + pad), min(img_bgr.shape[0], y + h + pad)
    face_roi = result[y1:y2, x1:x2]

    # Perhalus kulit (menjaga tepi tajam mata/bibir)
    smooth = cv2.bilateralFilter(face_roi, d=9, sigmaColor=55, sigmaSpace=55)
    # Pertajam detail halus (mata, alis, bibir) via unsharp mask
    blur = cv2.GaussianBlur(smooth, (0, 0), 3)
    sharp = cv2.addWeighted(smooth, 1.4, blur, -0.4, 0)
    # Blend lembut supaya transisi ke area luar wajah tidak terlihat kotak
    mask = np.zeros(face_roi.shape[:2], dtype="float32")
    cv2.ellipse(
        mask,
        (mask.shape[1] // 2, mask.shape[0] // 2),
        (mask.shape[1] // 2, mask.shape[0] // 2),
        0, 0, 360, 1.0, -1,
    )
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask_3 = np.dstack([mask, mask, mask])
    blended = (sharp.astype("float32") * mask_3 + face_roi.astype("float32") * (1 - mask_3))
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype("uint8")

  return result, len(faces)


def apply_old_photo_restoration(img_bgr):
  """Restorasi foto lama: denoise kuat, hilangkan goresan/scratch halus
  lewat deteksi+inpainting, koreksi warna pudar (gray-world white
  balance), lalu naikkan kontras & ketajaman."""
  # 1. Denoise kuat (foto lama biasanya banyak grain/film noise)
  restored = cv2.fastNlMeansDenoisingColored(img_bgr, None, 12, 12, 7, 21)

  # 2. Deteksi goresan/scratch tipis lalu tambal dengan inpainting
  gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
  diff = cv2.absdiff(gray, cv2.medianBlur(gray, 7))
  _, scratch_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
  scratch_mask = cv2.dilate(scratch_mask, np.ones((3, 3), np.uint8), iterations=1)
  restored = cv2.inpaint(restored, scratch_mask, 3, cv2.INPAINT_TELEA)

  # 3. Koreksi warna pudar/menguning (gray-world auto white balance)
  result_f = restored.astype("float32")
  avg_b, avg_g, avg_r = [result_f[:, :, i].mean() for i in range(3)]
  avg_gray = (avg_b + avg_g + avg_r) / 3.0
  avg_b, avg_g, avg_r = max(avg_b, 1e-3), max(avg_g, 1e-3), max(avg_r, 1e-3)
  result_f[:, :, 0] *= avg_gray / avg_b
  result_f[:, :, 1] *= avg_gray / avg_g
  result_f[:, :, 2] *= avg_gray / avg_r
  restored = np.clip(result_f, 0, 255).astype("uint8")

  # 4. Naikkan sedikit kontras & ketajaman supaya detail muncul kembali
  restored = cv2.convertScaleAbs(restored, alpha=1.12, beta=8)
  blur = cv2.GaussianBlur(restored, (0, 0), 3)
  restored = cv2.addWeighted(restored, 1.3, blur, -0.3, 0)
  return restored


def compute_foreground_mask(img_bgr):
  """Segmentasi objek utama vs background pakai GrabCut (bawaan OpenCV,
  tanpa model AI eksternal). Dijalankan di resolusi kecil supaya cepat,
  hasil mask di-resize belakangan ke resolusi final."""
  small = img_bgr
  max_dim = 500
  h, w = small.shape[:2]
  if max(h, w) > max_dim:
    scale = max_dim / max(h, w)
    small = cv2.resize(small, (int(w * scale), int(h * scale)))

  mask = np.zeros(small.shape[:2], np.uint8)
  bgd_model = np.zeros((1, 65), np.float64)
  fgd_model = np.zeros((1, 65), np.float64)
  sh, sw = small.shape[:2]
  rect = (int(sw * 0.04), int(sh * 0.04), int(sw * 0.92), int(sh * 0.92))
  try:
    cv2.grabCut(small, mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("float32")
  except cv2.error:
    # Kalau GrabCut gagal (foto polos/kontras rendah), anggap semua foreground
    fg_mask = np.ones(small.shape[:2], dtype="float32")

  fg_mask = cv2.GaussianBlur(fg_mask, (9, 9), 0)  # tepi lebih halus
  return fg_mask  # nilai 0..1, 1 = objek utama


def apply_background_enhancer(img_bgr, fg_mask_small, mode):
  """Terapkan efek ke area BACKGROUND saja (objek utama tetap tajam),
  berdasarkan mask hasil compute_foreground_mask."""
  h, w = img_bgr.shape[:2]
  fg_mask = cv2.resize(fg_mask_small, (w, h))
  fg_mask_3 = np.dstack([fg_mask, fg_mask, fg_mask])

  if mode == "Blur Background":
    bg_version = cv2.GaussianBlur(img_bgr, (0, 0), 15)
  elif mode == "Cerahkan Background":
    bg_version = cv2.convertScaleAbs(img_bgr, alpha=1.15, beta=25)
  elif mode == "Studio Gelap (Ganti Warna)":
    bg_version = np.full_like(img_bgr, (35, 32, 30))  # abu gelap ala studio
  else:
    return img_bgr

  blended = img_bgr.astype("float32") * fg_mask_3 + bg_version.astype("float32") * (1 - fg_mask_3)
  return np.clip(blended, 0, 255).astype("uint8")


def apply_ai_avatar_style(img_bgr, style):
  """Filter stylization ala 'avatar/kartun' pakai modul photo bawaan
  OpenCV (stylization & pencilSketch) — tanpa model generative AI."""
  if style == "Kartun (Cartoon)":
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(img_bgr, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)
  elif style == "Sketsa Pensil":
    _, color_sketch = cv2.pencilSketch(
        img_bgr, sigma_s=60, sigma_r=0.07, shade_factor=0.05
    )
    return color_sketch
  elif style == "Lukisan Cat (Oil Paint)":
    return cv2.stylization(img_bgr, sigma_s=60, sigma_r=0.45)
  return img_bgr


set_custom_theme()
set_background(BG_PATH)

# ---------------- Gerbang login (opsional, lihat REQUIRE_LOGIN) ----------------
current_user = None
if REQUIRE_LOGIN:
  is_logged_in = render_auth_sidebar()
  if not is_logged_in:
    st.title("😈 AMPER.AI — Professional Editing & 4K Upscaler Suite")
    st.info(
        "Silakan **Masuk** atau **Daftar** dulu lewat panel di sebelah kiri ya.."
        " untuk mulai memakai Amper-AI PRO Setiap akun baru otomatis dapat"
        " kredit gratis untuk dicoba."
    )
    st.stop()
  current_user = st.session_state["user"]

# ---------------- Header dengan logo ----------------
header_col1, header_col2 = st.columns([1, 6])
with header_col1:
  try:
    st.image(LOGO_PATH, use_container_width=True)
  except Exception:
    st.markdown("<h1 style='margin:0;'>😈</h1>", unsafe_allow_html=True)

with header_col2:
  st.title("AMPER.AI — Professional Editing & 4K Upscaler Suite")
  st.markdown(
      "<p style='color: #a9d6c9; font-size: 1.05em;'>Platform pengolahan"
      " foto pintar berstandar industri dengan kontrol parameter lengkap "
      " & AI Upscaling.</p>",
      unsafe_allow_html=True,
  )

# ---------------- Upload foto (dipindah ke atas supaya bisa dianalisis
# dulu sebelum sidebar dibuat, untuk fitur Auto Enhance) ----------------
uploaded_file = st.file_uploader(
    "📂 Unggah File Foto Keren Kamu Kesini..(JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
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
    st.error("❌ Oopss..Gagal membaca file gambar. Coba unggah file JPG/PNG yang lain.")
    st.stop()

  # Turunkan foto asli dulu kalau memang sangat besar (mis. hasil kamera HP),
  # supaya proses edit di resolusi asli tetap ringan
  h0, w0 = img.shape[:2]
  if max(h0, w0) > MAX_INPUT_DIM:
    input_scale = MAX_INPUT_DIM / max(h0, w0)
    img = cv2.resize(
        img,
        (int(w0 * input_scale), int(h0 * input_scale)),
        interpolation=cv2.INTER_AREA,
    )
    st.info(
        "ℹ️ Maaf ya..Foto asli diturunkan sementara ke resolusi lebih kecil sebelum"
        " diproses agar server tidak kehabisan memori."
    )

  auto_suggestions = compute_auto_suggestions(img)

  # Otomatis terapkan saran sekali setiap ada foto baru diupload,
  # tapi slider tetap bisa digeser manual sesudahnya
  if st.session_state.get("auto_applied_for") != file_signature:
    for slider_key, val in auto_suggestions.items():
      st.session_state[slider_key] = val
    st.session_state["auto_applied_for"] = file_signature

# ---------------- Sidebar kontrol ----------------
with st.sidebar:
  st.markdown("## 🎛️ Ampera-AI PRO Control Panel")

  if auto_suggestions is not None:
    st.success(
        "🪄 Pencahayaan & ketajaman foto ini sudah disesuaikan otomatis"
        " berdasarkan analisis foto. Geser slider di bawah kalau mau"
        " diubah manual."
    )
    if st.button("🪄 Sesuaikan Ulang Otomatis", key="auto_enhance_btn"):
      for slider_key, val in auto_suggestions.items():
        st.session_state[slider_key] = val
      st.rerun()
    st.markdown("---")

  st.markdown("### 1. Light & Exposure")
  exposure = st.slider(
      "Exposure", -2.0, 2.0, 0.0, 0.1,
      help="Menyesuaikan keseluruhan pencahayaan (otomatis disarankan jika foto diunggah)",
      key="exposure",
  )
  contrast = st.slider(
      "Contrast", -50, 50, 10, 1, help="Mempertajam perbedaan terang dan gelap",
      key="contrast",
  )
  highlights = st.slider(
      "Highlights", -100, 100, -20, 1, help="Mengatur area paling terang",
      key="highlights",
  )
  shadows = st.slider(
      "Shadows", -100, 100, 25, 1, help="Mengangkat detail pada area gelap",
      key="shadows",
  )
  whites = st.slider("Whites", -50, 50, 0, 1)
  blacks = st.slider("Blacks", -50, 50, 0, 1)

  st.markdown("### 2. Color & White Balance")
  temp = st.slider(
      "Temperature (Kelvin/Tint)",
      -50, 50, -5, 1,
      help="Nuansa warna Hangat (Kuning) ke Dingin (Biru)",
  )
  tint = st.slider("Tint", -50, 50, 0, 1, help="Nuansa Hijau ke Magenta")
  vibrance = st.slider(
      "Vibrance", -50, 50, 15, 1, help="Menaikkan warna yang belum jenuh"
  )
  saturation = st.slider("Saturation", -50, 50, 10, 1, help="Kepadatan warna")

  st.markdown("### 3. Detail, Clarity & Effects")
  clarity = st.slider(
      "Clarity / Texture", -50, 50, 20, 1,
      help="Mempertegas kontras midtone/tekstur (otomatis disarankan jika foto diunggah)",
      key="clarity",
  )
  dehaze = st.slider("Dehaze", -50, 50, 10, 1, help="Menghilangkan kabut/asap tipis")
  sharpen = st.slider(
      "Sharpening HD", 0, 100, 30, 1,
      help="Mempertajam detail tepi objek (otomatis disarankan jika foto diunggah)",
      key="sharpen",
  )
  vignette = st.slider(
      "Vignette (Cinematic Edge)", 0, 100, 25, 1,
      help="Memberikan bayangan artistik di tepi foto",
  )

  st.markdown("### 4. Quality Boost (Non-AI Berat)")
  denoise_strength = st.slider(
      "Noise Reduction (sebelum upscale)", 0, 30, 0, 1,
      help="Mengurangi noise/grain sebelum di-upscale, supaya noise tidak"
      " ikut diperbesar. Semakin tinggi, semakin halus tapi detail bisa"
      " sedikit berkurang.",
  )
  smart_enhance = st.slider(
      "Smart Detail Enhance (setelah upscale)", 0, 100, 0, 1,
      help="Filter edge-aware yang menguatkan detail & tekstur supaya hasil"
      " upscaling terlihat lebih 'pop', tanpa memakai model AI berat.",
  )

  st.markdown("### 5. 🤖 AI Photo Tools")
  ai_enhancer_on = st.checkbox(
      "🪄 AI Photo Enhancer (Auto Enhance Pro)",
      value=False,
      help="Auto white balance + local contrast (CLAHE) + sharpen, 1 klik langsung nendang.",
  )
  face_enhancer_on = st.checkbox(
      "🧑 Face Enhancer (Halus & Tajamkan Wajah)",
      value=False,
      help="Deteksi wajah otomatis lalu perhalus kulit & pertajam detail mata/bibir.",
  )
  old_restore_on = st.checkbox(
      "🕰️ Restorasi Foto Lama",
      value=False,
      help="Untuk foto jadul: hilangkan noise & goresan, perbaiki warna pudar/menguning.",
  )
  bg_enhancer_mode = st.selectbox(
      "🖼️ Background Enhancer",
      ["Tidak Aktif", "Blur Background", "Cerahkan Background", "Studio Gelap (Ganti Warna)"],
      index=0,
      help="Objek utama dideteksi otomatis (GrabCut) & tetap tajam, hanya background yang diubah.",
  )
  avatar_style = st.selectbox(
      "🎭 AI Avatar / Style Filter",
      ["Tidak Aktif", "Kartun (Cartoon)", "Sketsa Pensil", "Lukisan Cat (Oil Paint)"],
      index=0,
      help="Filter stylization untuk bikin avatar/PP unik. Ini override tampilan akhir foto.",
  )
  st.caption(
      "ℹ️ Fitur di atas pakai algoritma computer-vision ringan bawaan OpenCV"
      " (bukan model generative-AI berat), supaya tetap stabil & cepat di"
      " Streamlit Cloud gratis."
  )

  st.markdown("---")
  upscale_choice = st.selectbox(
      "Resolution Upscaling", ["2x (HD 2K)", "4x (Ultra HD 4K)"], index=0
  )
  process_btn = st.button("⚒️ Terapkan & Render Instan")

# ---------------- Tampilkan foto & proses ----------------
if uploaded_file is not None and img is not None:
  col_orig, col_res = st.columns(2)
  with col_orig:
    st.subheader("🎆 Foto Asli")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

  if process_btn or "processed_img" not in st.session_state:
    if REQUIRE_LOGIN:
      user_credits = get_credits(current_user["id"])
      if user_credits <= 0:
        st.error(
            "💳 Kredit kamu sudah habis. Silakan top up dulu untuk lanjut"
            " memakai Amper.AI."
        )
        st.stop()
    try:
      with st.spinner("✨ Sedang merender mesin AI-PRO & Upscaler AI..."):
        scale_factor = 2 if "2x" in upscale_choice else 4
        h, w = img.shape[:2]

        # Guard anti-crash tambahan untuk hasil akhir
        out_pixels = (w * scale_factor) * (h * scale_factor)
        if out_pixels > MAX_OUTPUT_MEGAPIXELS:
          adjusted_scale = (MAX_OUTPUT_MEGAPIXELS / (w * h)) ** 0.5
          scale_factor = max(1.0, adjusted_scale)
          st.warning(
              "⚠️ MAAF YA..Resolusi hasil upscaling terlalu besar dan berisiko"
              f" membuat server kehabisan memori. Skala diturunkan otomatis"
              f" menjadi {scale_factor:.2f}x agar tetap aman."
          )

        # =========================================================
        # TAHAP -1 — Restorasi foto lama (opsional), dijalankan
        # paling awal di resolusi asli, sebelum efek lain menumpuk
        # =========================================================
        if old_restore_on:
          img = apply_old_photo_restoration(img)

        # =========================================================
        # TAHAP 0 — noise reduction (opsional) di resolusi ASLI,
        # sebelum di-upscale, supaya noise tidak ikut diperbesar
        # =========================================================
        if denoise_strength > 0:
          img = cv2.fastNlMeansDenoisingColored(
              img, None, float(denoise_strength), float(denoise_strength), 7, 21
          )

        # =========================================================
        # TAHAP 1 — semua koreksi tone & warna dilakukan di resolusi
        # ASLI (kecil) dulu. Ini jauh lebih hemat memori daripada
        # meng-upscale dulu baru mengedit.
        # =========================================================
        img_f = img.astype("float32") / 255.0

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

        sat_adj_small = (
            cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
        )
        del hsv
        gc.collect()

        # =========================================================
        # TAHAP 1.5 — Face Enhancer & persiapan mask Background
        # Enhancer, dijalankan di resolusi KECIL (sebelum upscale)
        # supaya deteksi wajah/objek lebih cepat & hemat memori.
        # =========================================================
        if face_enhancer_on:
          sat_adj_small, n_faces = apply_face_enhancer(sat_adj_small)
          if n_faces == 0:
            st.info("ℹ️ Face Enhancer aktif, tapi tidak ada wajah yang terdeteksi di foto ini.")

        bg_fg_mask_small = None
        if bg_enhancer_mode != "Tidak Aktif":
          bg_fg_mask_small = compute_foreground_mask(sat_adj_small)

        # =========================================================
        # TAHAP 2 — upscaling dilakukan SETELAH koreksi warna,
        # jadi cuma satu kali proses resize di gambar besar.
        # =========================================================
        new_w = max(1, int(w * scale_factor))
        new_h = max(1, int(h * scale_factor))
        upscaled = cv2.resize(
            sat_adj_small, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
        )
        del sat_adj_small
        gc.collect()

        sat_adj = upscaled.astype("float32") / 255.0
        del upscaled

        # =========================================================
        # TAHAP 3 — clarity, dehaze, sharpen, vignette di resolusi
        # akhir (besar). Hanya sedikit array besar yang aktif.
        # =========================================================
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

        # =========================================================
        # TAHAP 4 — Smart Detail Enhance (opsional): filter edge-aware
        # bawaan OpenCV, memberi kesan "AI upscaler" tanpa model berat
        # =========================================================
        if smart_enhance > 0:
          sigma_s = 10 + (smart_enhance / 100.0) * 40  # ~10–50
          sigma_r = 0.15 + (smart_enhance / 100.0) * 0.35  # ~0.15–0.5
          final_bgr = cv2.detailEnhance(
              final_bgr, sigma_s=sigma_s, sigma_r=sigma_r
          )
          gc.collect()

        # =========================================================
        # TAHAP 5 — AI Photo Enhancer 1-klik (opsional), diterapkan
        # di resolusi akhir supaya white balance & CLAHE ikut sinkron
        # dengan hasil upscaling & sharpening di atas.
        # =========================================================
        if ai_enhancer_on:
          final_bgr = apply_ai_photo_enhancer(final_bgr)
          gc.collect()

        # =========================================================
        # TAHAP 6 — Background Enhancer: mask dari resolusi kecil
        # di-resize ke resolusi final, lalu efek diterapkan HANYA
        # ke area background (objek utama tetap tajam & natural).
        # =========================================================
        if bg_enhancer_mode != "Tidak Aktif" and bg_fg_mask_small is not None:
          final_bgr = apply_background_enhancer(
              final_bgr, bg_fg_mask_small, bg_enhancer_mode
          )
          gc.collect()

        # =========================================================
        # TAHAP 7 — AI Avatar / Style Filter (opsional). Filter ini
        # bersifat "final look" jadi sengaja diterapkan PALING akhir,
        # menimpa hasil editing tone/detail di atasnya.
        # =========================================================
        if avatar_style != "Tidak Aktif":
          final_bgr = apply_ai_avatar_style(final_bgr, avatar_style)
          gc.collect()

        st.session_state["processed_img"] = cv2.cvtColor(
            final_bgr, cv2.COLOR_BGR2RGB
        )
        del final_bgr
        gc.collect()

        if REQUIRE_LOGIN:
          deduct_credit(current_user["id"])
    except Exception as e:
      st.error(
          "❌ Oppss..Terjadi kesalahan saat memproses gambar (kemungkinan foto"
          " terlalu besar untuk skala upscaling yang dipilih, atau kombinasi"
          " fitur AI Photo Tools terlalu berat). Coba unggah foto dengan"
          " resolusi lebih kecil, nonaktifkan salah satu fitur AI Photo"
          " Tools, atau pilih 2x HD Standard dulu."
      )
      with st.expander("Detail teknis error"):
        st.exception(e)
      st.session_state.pop("processed_img", None)

  with col_res:
    st.subheader("🎇 Hasil AI Pro & Upscaled")
    if "processed_img" in st.session_state:
      st.image(st.session_state["processed_img"], use_container_width=True)

      result_pil = Image.fromarray(st.session_state["processed_img"])
      buf = io.BytesIO()
      result_pil.save(buf, format="JPEG", quality=95)
      byte_im = buf.getvalue()

      st.download_button(
          label="📥 Unduh Foto HD Pro (JPEG)",
          data=byte_im,
          file_name="amper_ai_pro.jpg",
          mime="image/jpeg",
          use_container_width=True,
      )
else:
  st.info(
      "👆 Silakan unggah foto terlebih dahulu melalui tombol di atas untuk"
      " mulai menggunakan suite lengkap Ampera-Ai & Upscaler."
  )
