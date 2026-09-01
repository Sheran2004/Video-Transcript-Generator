import os
import tempfile
from pathlib import Path

import streamlit as st
import whisper
from docx import Document


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Video Transcript Generator",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Video Transcript Generator")
st.write(
    "Upload a video and automatically convert its spoken content "
    "into a complete text transcript."
)

st.info(
    "Your video is processed locally on this computer. "
    "No paid API is required."
)


# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your video",
    type=["mp4", "mov", "avi", "mkv", "webm", "mpeg", "mpg"]
)


# -----------------------------
# Device Detection
# -----------------------------

def get_device():
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"

        return "cpu", "int8"

    except Exception:
        return "cpu", "int8"

@st.cache_resource
def load_whisper_model(device):
    with st.spinner("Loading speech recognition model..."):
        return whisper.load_model("tiny", device=device)

# -----------------------------
# Create TXT
# -----------------------------
def create_txt(transcript):
    return transcript.encode("utf-8")


# -----------------------------
# Create DOCX
# -----------------------------

def create_docx(transcript):
    document = Document()
    document.add_heading("Video Transcript", level=1)

    for paragraph in transcript.split("\n"):
        if paragraph.strip():
            document.add_paragraph(paragraph)

    # Windows-safe temporary file handling
    fd, temp_path = tempfile.mkstemp(suffix=".docx")

    try:
        os.close(fd)

        document.save(temp_path)

        with open(temp_path, "rb") as file:
            data = file.read()

        return data

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# -----------------------------
# Main Processing
# -----------------------------
if uploaded_file:

    st.success(f"Selected: {uploaded_file.name}")

    if st.button("🚀 Generate Transcript", type="primary"):

        temp_video = None

        try:
            # Save uploaded video temporarily
            suffix = Path(uploaded_file.name).suffix

            temp_video = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            )

            temp_video.write(uploaded_file.getbuffer())
            temp_video.close()

            # Detect device
            device, _ = get_device()
            st.write(f"Processing on: **{device.upper()}**")

            # Load Whisper model
            model = load_whisper_model(device)

            # Transcribe
            with st.spinner(
                "Transcribing video... This may take some time."
            ):
                result = model.transcribe(
                    temp_video.name,
                    fp16=(device == "cuda")
                )

            segments = result.get("segments", [])

            if not segments:
                st.error(
                    "No spoken content could be detected in this video."
                )
            else:

                # Build transcript
                transcript_parts = []

                for segment in segments:

                    text = segment.get("text", "").strip()

                    if text:
                        start = segment.get("start", 0)
                        end = segment.get("end", 0)

                        minutes_start = int(start // 60)
                        seconds_start = int(start % 60)

                        minutes_end = int(end // 60)
                        seconds_end = int(end % 60)

                        transcript_parts.append(
                            f"[{minutes_start:02d}:{seconds_start:02d} - "
                            f"{minutes_end:02d}:{seconds_end:02d}] "
                            f"{text}"
                        )

                transcript = "\n".join(transcript_parts)

                st.success("✅ Transcript generated successfully!")

                # -----------------------------
                # Preview
                # -----------------------------
                st.subheader("📄 Transcript Preview")

                st.text_area(
                    "Generated Transcript",
                    transcript,
                    height=400
                )

                # -----------------------------
                # Downloads
                # -----------------------------
                txt_data = create_txt(transcript)
                docx_data = create_docx(transcript)

                st.subheader("⬇️ Download Transcript")

                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        label="📄 Download TXT",
                        data=txt_data,
                        file_name="transcript.txt",
                        mime="text/plain"
                    )

                with col2:
                    st.download_button(
                        label="📝 Download DOCX",
                        data=docx_data,
                        file_name="transcript.docx",
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "wordprocessingml.document"
                        )
                    )

        except Exception as e:

            st.error("❌ Something went wrong.")

            st.exception(e)

        finally:

            if temp_video and os.path.exists(temp_video.name):
                os.unlink(temp_video.name)