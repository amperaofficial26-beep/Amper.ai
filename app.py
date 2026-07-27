import base64
import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(page_title="AMPER.AI", page_icon="⚡", layout="centered")


# Fungsi untuk mengubah gambar lokal agar bisa dibaca CSS sebagai Background Base64
def set_background(image_file):
  with open(image_file, "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

  css = f"""
    <style>
    .stApp {{
        background-image: linear-gradient( rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8) ), url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: #d4af37;
    }
    h1, h2, h3 {{ color: #ffd700 !important; font-family: 'serif'; }}
    .stButton>button {{
        background: linear-gradient(90deg, #b8860b, #ffd700);
        color: #000000;
        font-weight: bold;
        border-radius: 8px;
        border: none;
    }
    </style>
    """
  st.markdown(css, unsafe_allow_html=True)


# Panggil fungsi background (Pastikan nama file gambarnya sesuai di folder proyekmu)
# Tips: Ubah nama file gambar yang kamu upload menjadi 'bg_amper.jpg' dan simpan sefolder dengan app.py
try:
  set_background("bg_amper.jpg")
except:
  pass  # Fallback jika gambar belum dimasukkan

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
      use_container_width=True,
  )

  if st.button("🚀 Proses Upscaling ke 4K"):
    with st.spinner(
        "Sedang memproses (Saturasi: 45, Suhu: 15, Ketajaman: 30)..."
    ):
      # 1. Resize gambar (Skala 2x)
      height, width = img.shape[:2]
      scale_factor = 2
      new_width = width * scale_factor
      new_height = height * scale_factor

      upscaled = cv2.resize(
          img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
      )

      # 2. Atur Saturasi (Nilai tetap 45%)
      hsv = cv2.cvtColor(upscaled, cv2.COLOR_BGR2HSV).astype("float32")
      hsv[:, :, 1] = hsv[:, :, 1] * 0.45
      hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
      saturated = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

      # 3. Atur Suhu Warna / Temperature (Nilai tetap 15)
      temp_val = 15
      lab = cv2.cvtColor(saturated, cv2.COLOR_BGR2LAB).astype("float32")
      lab[:, :, 2] += temp_val
      lab = np.clip(lab, 0, 255)
      temp_adjusted = cv2.cvtColor(lab.astype("uint8"), cv2.COLOR_LAB2BGR)

      # 4. Efek Halus (Denoise)
      smoothed = cv2.bilateralFilter(
          temp_adjusted, d=9, sigmaColor=75, sigmaSpace=75
      )

      # 5. Efek Ketajaman (Nilai tetap 30)
      sharp_weight = 3.0
      gaussian = cv2.GaussianBlur(smoothed, (0, 0), 3.0)
      sharpened = cv2.addWeighted(
          smoothed, 1.0 + sharp_weight, gaussian, -sharp_weight, 0
      )

      # Konversi kembali ke RGB untuk Streamlit
      final_image = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

    st.success("✨ Selesai diproses dengan parameter kustom!")
    st.image(final_image, caption="Hasil Upscaled 4K")

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
