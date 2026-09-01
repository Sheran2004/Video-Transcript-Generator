# 🎙️ AI Video Transcription Tool

An AI-powered local video transcription application built with **Python, Streamlit, OpenAI Whisper and Pyannote Audio**.

The application converts spoken content from videos into timestamped transcripts, detects speakers, supports translation, editing, searching and multiple export formats.

## ✨ Features

- 🎥 Upload video files and generate transcripts
- 🤖 OpenAI Whisper transcription
- ⚡ CUDA GPU acceleration when NVIDIA GPU is available
- 🎚️ Multiple Whisper models:
  - Tiny — Fastest
  - Base — Balanced
  - Small — Better Accuracy
  - Medium — High Accuracy
- 🌐 Automatic language detection
- 🗣️ Speaker diarization using Pyannote
- 👥 Automatic speaker labels such as Speaker 1, Speaker 2, etc.
- 📊 Speaker-wise statistics
  - Speaking duration
  - Number of transcript segments
- ✏️ Editable transcript
- 🔎 Search transcript
- 📋 Copy transcript
- 🌍 Translate transcript into multiple languages
- 📥 Download transcript as:
  - TXT
  - DOCX
  - PDF
  - SRT
  - VTT
  - JSON
- 🔒 Video processing is performed locally
- 💰 No paid transcription API required

## 🛠️ Tech Stack

- Python
- Streamlit
- OpenAI Whisper
- Pyannote Audio
- PyTorch
- CUDA
- FFmpeg
- SoundFile
- Deep Translator
- Python-Docx
- ReportLab

## 📁 Project Structure

```text
Video-Transcript-Generator/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── venv/

⚙️ Installation

1. Clone the repository
git clone https://github.com/Sheran2004/Video-Transcript-Generator.git
cd Video-Transcript-Generator
2. Create virtual environment
python -m venv venv
3. Activate virtual environment
Windows PowerShell
.\venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt

🤗 Hugging Face Setup

Speaker diarization uses:
pyannote/speaker-diarization-community-1
You need a Hugging Face account and access to the model.
After accepting the model terms, authenticate using:
hf auth login
Enter your Hugging Face access token when prompted.

▶️ Run the Application

streamlit run app.py
The application will open in your browser.

🎯 How to Use

1. Upload a video.
2. Select the Whisper model.
3. Select the video language or use Auto Detect.
4. Enable translation if required.
5. Select the desired translation language.
6. Select Speaker Count or use Auto Detect.
7. Click Generate Transcript.
8. View the timestamped transcript.
9. Review speaker labels and speaker statistics.
10. Edit or search the transcript if required.
11. Copy the transcript or download it in the required format.

⚡ GPU Acceleration

If CUDA is available, the application automatically uses the NVIDIA GPU for Whisper and speaker diarization.
Otherwise, processing falls back to CPU.

🔒 Privacy

Videos are processed locally on the user's computer.
The application does not require a paid transcription API.

📄 Supported Export Formats

Format	Description
TXT	Plain text transcript
DOCX	Microsoft Word document
PDF	PDF transcript
SRT	Subtitle format
VTT	Web subtitle format
JSON	Structured transcript data

👥 Speaker Diarization

The application identifies different speakers and associates them with transcript segments.
Example:
[00:00 - 00:04] Speaker 1: Hello, welcome to the meeting.
[00:04 - 00:08] Speaker 2: Thank you for having me.
[00:08 - 00:12] Speaker 1: Let's get started.
Speaker statistics provide an overview of each detected speaker's participation.

🚀 Future Improvements

- Real-time transcription
- Better speaker name customization
- Audio-only input support
- Batch video processing
- Improved subtitle styling
- Advanced transcript summarization
- AI-powered meeting insights

👨‍💻 Author

Mohammad Sheran Asgar