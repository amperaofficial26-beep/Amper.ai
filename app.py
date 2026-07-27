Import base64
import cv2
import io
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(page_title="AMPER.AI", page_icon="📸", layout="centered")


# Fungsi Background
def set_background(image_file):
  try:
    with open(image_file, "rb") as f:
      encoded = base64.b64encode(f.read()).decode()

    css = (
        """
        <style>
        .stApp {
            background-image: linear-gradient( rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.85) ), url("data:image/jpeg;base64,"""
        + encoded
        + """");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: #d4af37;
        }
        h1, h2, h3 { color: #ffd700 !important; font-family: 'serif'; }
        .stButton>button {
            background: linear-gradient(90deg, #b8860b, #ffd700);
            color: #000000;
            font-weight: bold;
            border-radius: 8px;
            border: none;
        }
        </style>
        """
    )
    st.markdown(css, unsafe_allow_html=True)
  except:
    css = """
        <style>
        .stApp {
            background-color: #0e0e0e;
            color: #d4af37;
        }
        h1, h2, h3 { color: #ffd700 !important; font-family: 'serif'; }
        .stButton>button {
            background: linear-gradient(90deg, #b8860b, #ffd700);
            color: #000000;
            font-weight: bold;
            border-radius: 8px;
            border: none;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


set_background("bg_amper.jpg")

# Tampilkan Logo PNG jika ada
try:
  st.image("logo_amper.png", width=120)
except:
  pass

st.title("📷 AMPER.AI")
st.markdown(
    "<p style='color: #a3a3a3;'>Next-Gen Local Image & Video Upscaler</p>",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Letak Foto Kamu Disini (JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

  st.image(
      cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
      caption="Foto Asli",
      use_column_width=True,
  )

  if st.button("🪛🪛Proses Upscaling & Auto Preset HD"):
    with st.spinner("Sedang menerapkan preset otomatis dan upscaling 4K..."):
      height, width = img.shape[:2]
      scale_factor = 2
      new_width = width * scale_factor
      new_height = height * scale_factor

      # 1. Upscale Lanczos (4K Simulation)
      upscaled = cv2.resize(
          img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
      )

      # ==========================================
      # TAMBAHAN: PENYESUAIAN OTOMATIS (AUTO-ENHANCE)
      # ==========================================
      # Menerapkan CLAHE secara otomatis untuk menyeimbangkan kecerahan & detail bayangan
      lab_auto = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
      l_auto, a_auto, b_auto = cv2.split(lab_auto)
      clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
      l_auto = clahe.apply(l_auto)
      lab_auto = cv2.merge((l_auto, a_auto, b_auto))
      upscaled = cv2.cvtColor(lab_auto, cv2.COLOR_LAB2BGR)
      # ==========================================

      # 2. Nilai Preset Otomatis (Hardcoded sesuai resep)
      exposure_val = -12
      brightness_val = -23
      contrast_val = 7
      saturation_val = 15
      temp_val = -16
      sharpen_val = 16
      clarity_val = 13
      structure_val = -13

      # 3. Exposure, Brightness & Contrast (LAB L-Channel)
      lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB).astype("float32")
      l_channel, a_channel, b_channel = cv2.split(lab)

      l_channel += exposure_val + brightness_val
      if contrast_val != 0:
        factor = (259 * (contrast_val + 255)) / (255 * (259 - contrast_val))
        l_channel = factor * (l_channel - 128) + 128

      l_channel = np.clip(l_channel, 0, 255)
      lab = cv2.merge([l_channel, a_channel, b_channel])
      adjusted = cv2.cvtColor(lab.astype("uint8"), cv2.COLOR_LAB2BGR)

      # 4. Temperature (Temp)
      lab_temp = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB).astype("float32")
      lab_temp[:, :, 2] += temp_val * 0.5
      lab_temp = np.clip(lab_temp, 0, 255)
      temp_adjusted = cv2.cvtColor(lab_temp.astype("uint8"), cv2.COLOR_LAB2BGR)

      # 5. Saturation
      hsv = cv2.cvtColor(temp_adjusted, cv2.COLOR_BGR2HSV).astype("float32")
      sat_multiplier = 1.0 + (saturation_val / 100.0)
      hsv[:, :, 1] = hsv[:, :, 1] * sat_multiplier
      hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
      color_adjusted = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

      # 6. Structure, Clarity & Sharpen (Unsharp Masking)
      gaussian = cv2.GaussianBlur(
          color_adjusted, (0, 0), max(1.0, abs(clarity_val) / 3.0)
      )
      sharp_weight = (sharpen_val + abs(clarity_val)) / 30.0
      sharpened = cv2.addWeighted(
          color_adjusted, 1.0 + sharp_weight, gaussian, -sharp_weight, 0
      )

      final_image = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

    st.success("✨ Selesai,Selamat Fotonya udah jadi..🥳🥳!")
    st.image(final_image, caption="(AFTER)Hasil Peningkatan Gambar..."

    result_pil = Image.fromarray(final_image)
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG", quality=95)
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Mau Di-Downlaod Fotonya,Silahkan....",
        data=byte_im,
        file_name="amper_ai_4k.jpg",
        mime="image/jpeg",
    )
