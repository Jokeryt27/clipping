from pathlib import Path
import json
import re
import subprocess
import gradio as gr
from faster_whisper import WhisperModel
try:
    import cv2
except ImportError:
    cv2 = None

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "outputs"
DOWNLOADS = BASE / "downloads"
OUTPUT.mkdir(exist_ok=True)
DOWNLOADS.mkdir(exist_ok=True)

_MODEL = None

def get_model(model_size: str):
    global _MODEL
    if _MODEL is None or getattr(_MODEL, "_clip_model_size", None) != model_size:
        try:
            _MODEL = WhisperModel(model_size, device="cuda", compute_type="float16")
        except Exception:
            _MODEL = WhisperModel(model_size, device="cpu", compute_type="int8")
        _MODEL._clip_model_size = model_size
    return _MODEL

def ffmpeg_ok():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, check=True)

def download_video(url):
    if not url or not url.strip():
        raise ValueError("YouTube URL डालें.")
    if not re.match(r"^https?://", url.strip(), re.I):
        raise ValueError("Valid http/https URL डालें.")

    out_template = str(DOWNLOADS / "%(id)s.%(ext)s")
    cmd = ["yt-dlp", "--no-playlist",
           "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
           "--merge-output-format", "mp4", "-o", out_template, url.strip()]
    result = run(cmd)
    files = sorted(DOWNLOADS.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    videos = [p for p in files if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
    if not videos:
        raise RuntimeError("Video download नहीं हो पाया. yt-dlp output: " + result.stderr[-1200:])
    return str(videos[0])

def transcribe(video_path, model_size):
    model = get_model(model_size)
    segments, info = model.transcribe(video_path, vad_filter=True, word_timestamps=True)
    rows = []
    for s in segments:
        text = (s.text or "").strip()
        if text:
            rows.append({"start": float(s.start), "end": float(s.end), "text": text})
    return rows, info

def score_segment(text):
    t = text.lower()
    hooks = ["but","however","because","secret","mistake","truth","why","how",
             "never","always","important","crazy","surprising","problem","tip",
             "best","worst","first","finally","actually"]
    score = sum(2 for w in hooks if re.search(r"\b" + re.escape(w) + r"\b", t))
    score += min(len(text) / 80, 5)
    score += 2 if "?" in text else 0
    score += 1 if "!" in text else 0
    return score

def candidates(segments, clip_len=45, max_clips=5):
    results = []
    for s in segments:
        start = max(0, s["start"] - 4)
        end = start + clip_len
        nearby = [x for x in segments if x["start"] < end and x["end"] > start]
        joined = " ".join(x["text"] for x in nearby)
        if len(joined) < 35:
            continue
        results.append({"start": start, "end": end,
                        "score": round(score_segment(joined[:1200]), 2),
                        "text": joined[:1200]})
    results.sort(key=lambda x: x["score"], reverse=True)
    selected = []
    for c in results:
        if all(abs(c["start"] - x["start"]) > clip_len * 0.65 for x in selected):
            selected.append(c)
        if len(selected) >= max_clips:
            break
    return selected

def detect_face_center(video_path, start, duration):
    if cv2 is None:
        return 0.5
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.5
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    video_duration = total / fps if total else duration
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    centers = []
    for i in range(8):
        t = min(start + duration * (i + 0.5) / 8, max(0, video_duration - 0.1))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            centers.append((x + w / 2) / max(1, frame.shape[1]))
    cap.release()
    if not centers:
        return 0.5
    return max(0.15, min(0.85, sum(centers) / len(centers)))

def make_srt(segments, clip_start, clip_end, out_file):
    lines, idx = [], 1
    def ts(sec):
        ms = int(round(sec * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        sec2, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{sec2:02d},{ms:03d}"
    for s in segments:
        a, b = max(s["start"], clip_start), min(s["end"], clip_end)
        if b <= a:
            continue
        lines += [str(idx), f"{ts(a-clip_start)} --> {ts(b-clip_start)}", s["text"], ""]
        idx += 1
    Path(out_file).write_text("\n".join(lines), encoding="utf-8")

def render_clip(video, clip, index, segments):
    start, end = clip["start"], clip["end"]
    duration = end - start
    out = OUTPUT / f"short_{index:02d}.mp4"
    srt = OUTPUT / f"short_{index:02d}.srt"
    make_srt(segments, start, end, srt)

    face_x = detect_face_center(video, start, duration)
    crop_expr = f"crop=ih*9/16:ih:iw*{face_x}-ih*9/32:0"
    subtitle_file = f"outputs/short_{index:02d}.srt"
    vf = (
        crop_expr + ","
        "scale=1080:1920,"
        f"subtitles='{subtitle_file}':force_style="
        "'FontName=Arial,FontSize=18,Bold=1,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,"
        "Alignment=2,MarginV=120'"
    )
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", video, "-t", str(duration),
           "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-c:a", "aac", "-b:a", "128k", str(out)]
    run(cmd)
    return str(out), str(srt)

def process(video, url, model_size, clip_len, count):
    if not ffmpeg_ok():
        return "FFmpeg नहीं मिला। FFmpeg install करके PATH में add करें.", [], ""
    try:
        if url and url.strip():
            video = download_video(url)
        if not video:
            return "Video file या YouTube URL डालें.", [], ""

        segments, _ = transcribe(video, model_size)
        clips = candidates(segments, int(clip_len), int(count))
        if not clips:
            return "कोई पर्याप्त candidate moment नहीं मिला.", [], ""

        outputs, report = [], []
        for i, clip in enumerate(clips, 1):
            mp4, srt = render_clip(video, clip, i, segments)
            outputs.append(mp4)
            report.append({
                "clip": i, "start": round(clip["start"], 2),
                "end": round(clip["end"], 2), "score": clip["score"],
                "transcript_preview": clip["text"][:300],
                "video": mp4, "captions": srt
            })

        report_file = OUTPUT / "clips.json"
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return f"Done: {len(outputs)} clips generated with burned captions.", outputs, str(report_file)
    except Exception as e:
        return f"Error: {e}", [], ""

with gr.Blocks(title="AI Shorts Clipper") as demo:
    gr.Markdown("# 🎬 AI Shorts Clipper\n**Free/local-first V1.2** — use only videos you own or have permission to edit.")
    url = gr.Textbox(label="YouTube URL", placeholder="Paste an authorized YouTube video URL here")
    video = gr.Video(label="Or choose a local video file", type="filepath")
    with gr.Row():
        model = gr.Dropdown(["tiny","base","small","medium"], value="small", label="Whisper model")
        length = gr.Slider(20, 60, value=45, step=5, label="Clip length (seconds)")
        count = gr.Slider(1, 10, value=5, step=1, label="Number of clips")
    button = gr.Button("🔥 FIND BEST CLIPS", variant="primary")
    status = gr.Textbox(label="Status")
    gallery = gr.File(label="Generated Shorts")
    report = gr.File(label="Clip report")
    button.click(process, [video, url, model, length, count], [status, gallery, report])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
