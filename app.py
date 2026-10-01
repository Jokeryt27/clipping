from pathlib import Path
import json
import re
import subprocess
import tempfile
import gradio as gr
from faster_whisper import WhisperModel

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "outputs"
OUTPUT.mkdir(exist_ok=True)

_MODEL = None

def get_model(model_size: str):
    global _MODEL
    if _MODEL is None or getattr(_MODEL, "_clip_model_size", None) != model_size:
        device = "cuda"
        compute = "float16"
        try:
            _MODEL = WhisperModel(model_size, device=device, compute_type=compute)
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
    score = 0
    hooks = [
        "but", "however", "because", "secret", "mistake", "truth", "why",
        "how", "never", "always", "important", "crazy", "surprising",
        "problem", "tip", "best", "worst", "first", "finally", "actually"
    ]
    score += sum(1 for w in hooks if re.search(r"\b" + re.escape(w) + r"\b", t)) * 2
    score += min(len(text) / 80, 5)
    if "?" in text:
        score += 2
    if "!" in text:
        score += 1
    return score

def candidates(segments, clip_len=45, max_clips=5):
    results = []
    for i, s in enumerate(segments):
        start = max(0, s["start"] - 4)
        end = start + clip_len
        nearby = [x for x in segments if x["start"] < end and x["end"] > start]
        joined = " ".join(x["text"] for x in nearby)
        if len(joined) < 35:
            continue
        score = score_segment(joined[:1200])
        results.append({
            "start": start,
            "end": end,
            "score": round(score, 2),
            "text": joined[:1200]
        })
    results.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    for c in results:
        if all(abs(c["start"] - x["start"]) > clip_len * 0.65 for x in selected):
            selected.append(c)
        if len(selected) >= max_clips:
            break
    return selected

def make_srt(segments, clip_start, clip_end, out_file):
    lines = []
    idx = 1
    for s in segments:
        a = max(s["start"], clip_start)
        b = min(s["end"], clip_end)
        if b <= a:
            continue
        a -= clip_start
        b -= clip_start

        def ts(sec):
            ms = int(round(sec * 1000))
            h, ms = divmod(ms, 3600000)
            m, ms = divmod(ms, 60000)
            sec2, ms = divmod(ms, 1000)
            return f"{h:02d}:{m:02d}:{sec2:02d},{ms:03d}"

        lines += [str(idx), f"{ts(a)} --> {ts(b)}", s["text"], ""]
        idx += 1
    Path(out_file).write_text("\n".join(lines), encoding="utf-8")

def render_clip(video, clip, index, segments):
    start, end = clip["start"], clip["end"]
    duration = end - start
    out = OUTPUT / f"short_{index:02d}.mp4"
    srt = OUTPUT / f"short_{index:02d}.srt"
    make_srt(segments, start, end, srt)

    # Center crop to 9:16. Captions are produced separately as SRT in V1.
    vf = "crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920"
    cmd = [
        "ffmpeg", "-y", "-ss", str(start), "-i", video,
        "-t", str(duration), "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k", str(out)
    ]
    run(cmd)
    return str(out), str(srt)

def process(video, model_size, clip_len, count):
    if not video:
        return "Please select a video.", [], ""
    if not ffmpeg_ok():
        return "FFmpeg नहीं मिला। FFmpeg install करके PATH में add करें.", [], ""

    try:
        segments, info = transcribe(video, model_size)
        clips = candidates(segments, int(clip_len), int(count))
        if not clips:
            return "कोई पर्याप्त candidate moment नहीं मिला.", [], ""

        outputs = []
        report = []
        for i, clip in enumerate(clips, 1):
            mp4, srt = render_clip(video, clip, i, segments)
            outputs.append(mp4)
            report.append({
                "clip": i,
                "start": round(clip["start"], 2),
                "end": round(clip["end"], 2),
                "score": clip["score"],
                "transcript_preview": clip["text"][:300],
                "video": mp4,
                "captions": srt
            })

        report_file = OUTPUT / "clips.json"
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return f"Done: {len(outputs)} clips generated.", outputs, str(report_file)
    except Exception as e:
        return f"Error: {e}", [], ""

with gr.Blocks(title="AI Shorts Clipper") as demo:
    gr.Markdown("# 🎬 AI Shorts Clipper\nLocal/free-first V1")
    video = gr.Video(label="Authorized video file", type="filepath")
    with gr.Row():
        model = gr.Dropdown(["tiny", "base", "small", "medium"], value="small", label="Whisper model")
        length = gr.Slider(20, 60, value=45, step=5, label="Clip length (seconds)")
        count = gr.Slider(1, 10, value=5, step=1, label="Number of clips")
    button = gr.Button("🔥 FIND BEST CLIPS", variant="primary")
    status = gr.Textbox(label="Status")
    gallery = gr.File(label="Generated Shorts")
    report = gr.File(label="Clip report")

    button.click(process, [video, model, length, count], [status, gallery, report])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
