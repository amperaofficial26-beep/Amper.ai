import base64
import gc
import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="AMPER.AI - Lightroom Pro Suite", page_icon="⚡", layout="wide"
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


set_custom_theme()
set_background(BG_PATH)

# ---------------- Header dengan logo ----------------
header_col1, header_col2 = st.columns([1, 6])
with header_col1:
  try:
    st.image(LOGO_PATH, use_container_width=True)
  except Exception:
    st.markdown("<h1 style='margin:0;'>⚡</h1>", unsafe_allow_html=True)

with header_col2:
  st.title("AMPER.AI — Professional Lightroom & 4K Upscaler Suite")
  st.markdown(
      "<p style='color: #a9d6c9; font-size: 1.05em;'>Platform pengolahan"
      " foto pintar berstandar industri dengan kontrol parameter lengkap ala"
      " Lightroom & AI Upscaling.</p>",
      unsafe_allow_html=True,
  )

# ---------------- Sidebar kontrol ----------------
with st.sidebar:
  st.markdown("## 🎛️ Ampera-AI PRO Control Panel")

  st.markdown("### 1. Light & Exposure")
  exposure = st.slider(
      "Exposure", -2.0, 2.0, 0.0, 0.1, help="Menyesuaikan keseluruhan pencahayaan"
  )
  contrast = st.slider(
      "Contrast", -50, 50, 10, 1, help="Mempertajam perbedaan terang dan gelap"
  )
  highlights = st.slider(
      "Highlights", -100, 100, -20, 1, help="Mengatur area paling terang"
  )
  shadows = st.slider(
      "Shadows", -100, 100, 25, 1, help="Mengangkat detail pada area gelap"
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
      "Clarity / Texture", -50, 50, 20, 1, help="Mempertegas kontras midtone/tekstur"
  )
  dehaze = st.slider("Dehaze", -50, 50, 10, 1, help="Menghilangkan kabut/asap tipis")
  sharpen = st.slider(
      "Sharpening HD", 0, 100, 30, 1, help="Mempertajam detail tepi objek"
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

  st.markdown("---")
  upscale_choice = st.selectbox(
      "Resolution Upscaling", ["2x (HD Standard)", "4x (Ultra HD 4K)"], index=0
  )
  process_btn = st.button("🚀 Terapkan & Render Instan")

# ---------------- Upload foto ----------------
uploaded_file = st.file_uploader(
    "📂 Unggah File Foto Anda (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  file_signature = f"{uploaded_file.name}_{uploaded_file.size}"
  if st.session_state.get("last_file_signature") != file_signature:
    st.session_state["last_file_signature"] = file_signature
    st.session_state.pop("processed_img", None)

  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

  if img is None:
    st.error("❌ Gagal membaca file gambar. Coba unggah file JPG/PNG yang lain.")
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
        "ℹ️ Foto asli diturunkan sementara ke resolusi lebih kecil sebelum"
        " diproses agar server tidak kehabisan memori."
    )

  col_orig, col_res = st.columns(2)
  with col_orig:
    st.subheader("📷 Foto Asli")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

  if process_btn or "processed_img" not in st.session_state:
    try:
      with st.spinner("✨ Sedang merender mesin Lightroom & Upscaler AI..."):
        scale_factor = 2 if "2x" in upscale_choice else 4
        h, w = img.shape[:2]

        # Guard anti-crash tambahan untuk hasil akhir
        out_pixels = (w * scale_factor) * (h * scale_factor)
        if out_pixels > MAX_OUTPUT_MEGAPIXELS:
          adjusted_scale = (MAX_OUTPUT_MEGAPIXELS / (w * h)) ** 0.5
          scale_factor = max(1.0, adjusted_scale)
          st.warning(
              "⚠️ Resolusi hasil upscaling terlalu besar dan berisiko"
              f" membuat server kehabisan memori. Skala diturunkan otomatis"
              f" menjadi {scale_factor:.2f}x agar tetap aman."
          )

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

        st.session_state["processed_img"] = cv2.cvtColor(
            final_bgr, cv2.COLOR_BGR2RGB
        )
        del final_bgr
        gc.collect()
    except Exception as e:
      st.error(
          "❌ Terjadi kesalahan saat memproses gambar (kemungkinan foto"
          " terlalu besar untuk skala upscaling yang dipilih). Coba unggah"
          " foto dengan resolusi lebih kecil, atau pilih 2x HD Standard"
          " dulu."
      )
      with st.expander("Detail teknis error"):
        st.exception(e)
      st.session_state.pop("processed_img", None)

  with col_res:
    st.subheader("✨ Hasil Lightroom Pro & Upscaled")
    if "processed_img" in st.session_state:
      st.image(st.session_state["processed_img"], use_container_width=True)

      result_pil = Image.fromarray(st.session_state["processed_img"])
      buf = io.BytesIO()
      result_pil.save(buf, format="JPEG", quality=95)
      byte_im = buf.getvalue()

      st.download_button(
          label="📥 Unduh Foto HD Pro (JPEG)",
          data=byte_im,
          file_name="amper_ai_lightroom_pro.jpg",
          mime="image/jpeg",
          use_container_width=True,
      )
else:
  st.info(
      "👆 Silakan unggah foto terlebih dahulu melalui tombol di atas untuk"
      " mulai menggunakan suite lengkap Lightroom & Upscaler."
  )
