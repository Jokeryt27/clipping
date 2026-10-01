# AI Shorts Clipper

Free/local-first Windows tool for turning **videos you are authorized to use** into vertical Shorts.

## V1 pipeline
1. Provide a local video file.
2. Extract audio and transcribe with faster-whisper.
3. Detect candidate moments from transcript segments.
4. Export 9:16 clips with FFmpeg.
5. Generate SRT captions for each clip.

## Requirements
- Windows 10/11
- Python 3.10+
- FFmpeg available on PATH
- Optional NVIDIA GPU for faster Whisper inference

## Install
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```powershell
python app.py
```

Open the local GUI shown by the app.

> Use only videos you own or have permission to download/edit. YouTube's terms and copyright rules still apply.
