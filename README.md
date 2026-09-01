# 🎙️ Video Transcript Generator

An AI-powered local video transcription tool built with **Python, Streamlit, and OpenAI Whisper**.

Upload a video and automatically convert its spoken content into a complete text transcript. The generated transcript can be previewed and downloaded as a **TXT** or **DOCX** file.

## ✨ Features

- 🎥 Upload video files
- 🤖 AI-powered speech-to-text using OpenAI Whisper
- ⚡ CUDA/GPU support when available
- 📝 Timestamped transcript
- 👀 Transcript preview
- 📄 Download transcript as TXT
- 📝 Download transcript as DOCX
- 🔒 Processes videos locally
- 💰 No paid API required

## 🛠️ Tech Stack

- Python
- Streamlit
- OpenAI Whisper
- PyTorch
- python-docx
- FFmpeg

## 📁 Project Structure

```text
Video-Transcript-Generator/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── venv/

🚀 Installation

1. Clone the repository
git clone https://github.com/Sheran2004/Video-Transcript-Generator.git
cd Video-Transcript-Generator
2. Create a virtual environment
python -m venv venv
3. Activate the virtual environment
Windows PowerShell:
.\venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
5. Install FFmpeg
FFmpeg is required by Whisper for audio/video processing.
Make sure FFmpeg is installed and available in your system PATH.

▶️ Run the Application
streamlit run app.py
The application will open in your browser.

🎯 How to Use
1. Open the application.
2. Upload a supported video.
3. Click Generate Transcript.
4. Wait for Whisper to process the video.
5. Preview the generated transcript.
6. Download the transcript as TXT or DOCX.

📦 Supported Video Formats
- MP4
- MOV
- AVI
- MKV
- WEBM
- MPEG
- MPG

⚡ GPU Support
The application automatically detects CUDA-compatible NVIDIA GPUs.
If CUDA is available, transcription is processed using the GPU for improved performance. Otherwise, the application falls back to CPU processing.

🔐 Privacy
Videos are processed locally on the user's computer. No paid external transcription API is required.

⚠️ Accuracy Note
The tool uses OpenAI Whisper for automatic speech recognition. Transcription accuracy can vary depending on:
- Audio quality
- Background noise
- Speaker accents
- Multiple speakers
- Speech clarity
- Language

Therefore, generated transcripts may require manual review for critical use cases.

📄 License
This project is intended for educational and development purposes.
