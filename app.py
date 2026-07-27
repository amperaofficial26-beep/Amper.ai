import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(page_title="AMPER.AI", page_icon="⚡", layout="centered")

# Estetika Dark Luxury Gold
st.markdown(
    """
    <style>
    .main { background-color: #0e0e0e; color: #d4af37; }
    h1, h2, h3 { color: #ffd700 !important; font-family: 'serif'; }
    .stButton>button {
        background: linear-gradient(90deg, #b8860b, #ffd700);
        color: #000000;
        font-weight: bold;
        border-radius: 8px;
        border: none;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("⚡ AMPER.AI")
st.markdown(
    "<p style='color: #a3a3a3;'>Next-Gen Local Image & Video Upscaler</p>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Pilih Foto (JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  # Baca gambar menggunakan OpenCV
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

  st.image(
      cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
      caption="Foto Asli",
      use_column_width=True,
  )

  if st.button("🚀 Proses Upscaling ke 4K"):
    with st.spinner("Sedang memproses detail AI & penghalusan 4K..."):
      # 1. Resize menggunakan Lanczos Interpolation
      height, width = img.shape[:2]
      scale_factor = 2  # Bisa dinaikkan ke 4 jika ingin perbesaran ekstra
      new_width = width * scale_factor
      new_height = height * scale_factor

      upscaled = cv2.resize(
          img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
      )

      # 2. Efek Halus (Denoise ringan agar mulus tanpa ngeblur)
      smoothed = cv2.bilateralFilter(upscaled, d=9, sigmaColor=75, sigmaSpace=75)

      # 3. Efek Ketajaman 4K (Unsharp Masking untuk menajamkan detail)
      gaussian = cv2.GaussianBlur(smoothed, (0, 0), 3.0)
      sharpened = cv2.addWeighted(smoothed, 1.5, gaussian, -0.5, 0)

      # Konversi kembali ke RGB untuk Streamlit
      final_image = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

    st.success("✨ Upscaling Selesai!")
    st.image(
        final_image, caption="Hasil Upscaled 4K (Dihaluskan & Ditegaskan)"
    )

    # Tombol Download
    result_pil = Image.fromarray(final_image)
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG", quality=95)
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Download Foto 4K",
        data=byte_im,
        file_name="amper_ai_4k.jpg",
        mime="image/jpeg",
    )
