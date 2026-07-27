import os
import cv2
import tempfile
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance

# ==========================================
# 1. KONFIGURASI BRANDING AMPER.AI
# ==========================================
st.set_page_config(
    page_title="AMPER.AI - Local AI Upscaler",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CUSTOM CSS (DARK LUXURY GOLD)
# ==========================================
st.markdown("""
<style>
.stApp {
    background-color: #1a1c1e;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='4' height='4' viewBox='0 0 4 4'%3E%3Cpath fill='%2322252a' fill-opacity='0.4' d='M1 3h1v1H1V3zm2-2h1v1H3V1z'%3E%3C/path%3E%3C/svg%3E");
}

[data-testid="stSidebar"] {
    background-color: #121315 !important;
    border-right: 1px solid #d4af37;
}

.stMarkdown, p, h1, h2, h3, h4, label, span {
    color: #e0e0e0 !important;
}

.stButton>button {
    background: linear-gradient(90deg, #b8860b 0%, #e6ca65 50%, #d4af37 100%) !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.3s ease !important;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(212, 175, 55, 0.4) !important;
}

.gold-text {
    background: linear-gradient(90deg, #f3e5ab 0%, #d4af37 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. FUNGSI UPSCALE LOKAL (OPENCV & PIL)
# ==========================================
def upscale_image_local(image, scale=2):
    # Ubah ukuran dengan Bicubic Interpolation
    width, height = image.size
    new_size = (width * scale, height * scale)
    upscaled = image.resize(new_size, Image.Resampling.BICUBIC)
    
    # Pertajam Gambar (Sharpening)
    enhancer = ImageEnhance.Sharpness(upscaled)
    upscaled = enhancer.enhance(1.8)
    
    # Penyesuaian Kontras
    contrast = ImageEnhance.Contrast(upscaled)
    upscaled = contrast.enhance(1.1)
    
    return upscaled

def upscale_video_local(input_path, scale=2):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    new_width, new_height = width * scale, height * scale
    
    # Temporary File untuk Output Video
    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_temp.name, fourcc, fps, (new_width, new_height))
    
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Upscale frame dengan OpenCV Interpolation
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Filter Penajaman Frame
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened_frame = cv2.filter2D(resized_frame, -1, kernel)
        
        out.write(sharpened_frame)
        current_frame += 1
        if total_frames > 0:
            progress_bar.progress(min(current_frame / total_frames, 1.0))

    cap.release()
    out.release()
    return output_temp.name

# ==========================================
# 4. TAMPILAN UTAMA
# ==========================================
st.markdown("<h1 class='gold-text'>AMPER.AI - 100% Free Local Upscaler</h1>", unsafe_allow_html=True)
st.caption("Memproses Media Secara Offline & Gratis Selamanya Langsung di Perangkat Kamu.")

uploaded_file = st.file_uploader("Unggah Video atau Gambar", type=["mp4", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    is_video = uploaded_file.name.lower().endswith('.mp4')
    
    if is_video:
        st.video(uploaded_file)
        if st.button("✨ Upscale Video (2x HD Local)"):
            with st.spinner("Memproses frame video secara lokal... Harap tunggu..."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                
                output_video_path = upscale_video_local(tfile.name)
                
                st.success("Video Upscaling Selesai!")
                st.video(output_video_path)
    else:
        col1, col2 = st.columns(2)
        original_img = Image.open(uploaded_file)
        
        with col1:
            st.subheader("Gambar Original")
            st.image(original_img, use_container_width=True)
            
        if st.button("✨ Upscale Gambar (2x High Res)"):
            with st.spinner("Meningkatkan kualitas gambar..."):
                result_img = upscale_image_local(original_img)
                
                with col2:
                    st.subheader("Hasil Upscale AI Local")
                    st.image(result_img, use_container_width=True)
                    st.success("Upscaling Berhasil!")