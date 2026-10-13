import json
import os
import re
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import boto3
import requests
from botocore.config import Config
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi

API_BASE = os.environ["CLIPFARM_API_BASE"].rstrip("/")
WORKER_TOKEN = os.environ["CLIPFARM_WORKER_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET = os.environ["R2_BUCKET"]
R2_PUBLIC_BASE_URL = os.environ["R2_PUBLIC_BASE_URL"].rstrip("/")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "4"))
MAX_CLIPS = int(os.getenv("MAX_CLIPS", "5"))
HEADERS = {"Authorization": "Bearer " + WORKER_TOKEN, "Content-Type": "application/json"}
client = OpenAI(api_key=OPENAI_API_KEY)
s3 = boto3.client("s3", endpoint_url="https://" + R2_ACCOUNT_ID + ".r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY_ID, aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto", config=Config(signature_version="s3v4"))

def api(method, path, payload=None):
    r = requests.request(method, API_BASE + path, headers=HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def stage(job_id, status, progress, message, **extra):
    api("POST", "/api/worker/jobs/" + job_id + "/progress",
        {"status": status, "progress": progress, "stage_message": message, **extra})

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError("Command failed: " + " ".join(cmd) + "\n" + p.stderr[-6000:])
    return p.stdout

def video_metadata(path):
    data = json.loads(run(["ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,width,height,r_frame_rate", "-of", "json", str(path)]))
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    return {"duration": float(data.get("format", {}).get("duration") or 0),
            "width": video.get("width"), "height": video.get("height"),
            "framerate": video.get("r_frame_rate")}

def youtube_id(url):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None

def get_caption_transcript(url):
    vid = youtube_id(url)
    if not vid:
        return None
    try:
        rows = YouTubeTranscriptApi().fetch(vid, languages=["en", "en-US", "en-GB"])
        return [{"start": float(x.start), "end": float(x.start+x.duration), "text": str(x.text)} for x in rows] or None
    except Exception:
        return None

def transcript_with_whisper(video_path, work):
    audio = work / "audio.mp3"
    run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(audio)])
    with audio.open("rb") as f:
        result = client.audio.transcriptions.create(model="whisper-1", file=f,
            response_format="verbose_json", timestamp_granularities=["word", "segment"])
    segments = [{"start": float(x.start), "end": float(x.end), "text": str(x.text).strip()}
                for x in (getattr(result, "segments", None) or [])]
    words = [{"start": float(x.start), "end": float(x.end), "text": str(x.word)}
             for x in (getattr(result, "words", None) or [])]
    return segments, words

def ass_time(seconds):
    cs = max(0, int(round(seconds*100)))
    return "%d:%02d:%02d.%02d" % (cs//360000, (cs//6000)%60, (cs//100)%60, cs%100)

def build_ass(words, offset, duration, out_path):
    lines = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920",
        "ScaledBorderAndShadow: yes", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,64,&H00FFFFFF,&H0000D7FF,&H00101010,&H90000000,-1,0,1,5,1,2,80,80,240,1",
        "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for w in words:
        start, end = float(w["start"])-offset, float(w["end"])-offset
        if end <= 0 or start >= duration:
            continue
        txt = str(w["text"]).replace("{", "(").replace("}", ")").replace("\\", "")
        lines.append("Dialogue: 0,"+ass_time(max(0,start))+","+ass_time(min(duration,end))+
                     ",Default,,0,0,0,,{\\fad(70,70)}"+txt)
    out_path.write_text("\n".join(lines), encoding="utf-8")

def render_clip(video, work, clip, index, words, metadata, job_id):
    start, end = max(0.0, float(clip["start_time"])), float(clip["end_time"])
    if end <= start or end-start > 90:
        raise ValueError("Invalid clip interval")
    end = min(end, float(metadata["duration"]) or end)
    duration = end-start
    ass, output = work/("clip-"+str(index)+".ass"), work/("clip-"+str(index)+".mp4")
    build_ass(words, start, duration, ass)
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,subtitles='"+str(ass).replace("'", "\\'")+"'"
    run(["ffmpeg", "-y", "-ss", str(start), "-i", str(video), "-t", str(duration), "-vf", vf,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-c:a", "aac", "-b:a", "128k",
         "-movflags", "+faststart", str(output)])
    key = "jobs/"+job_id+"/clip-"+str(index)+".mp4"
    s3.upload_file(str(output), R2_BUCKET, key, ExtraArgs={"ContentType":"video/mp4","CacheControl":"public, max-age=3600"})
    stored = s3.head_object(Bucket=R2_BUCKET, Key=key)
    if int(stored.get("ContentLength", 0)) <= 0:
        raise RuntimeError("Uploaded MP4 is empty")
    url = R2_PUBLIC_BASE_URL+"/"+key
    check = requests.get(url, headers={"Range":"bytes=0-0"}, timeout=30, stream=True)
    try:
        if check.status_code not in (200, 206):
            raise RuntimeError("Public R2 URL verification failed: HTTP "+str(check.status_code))
    finally:
        check.close()
    return url

def process(job):
    job_id = job["id"]
    if job.get("source_type") == "upload":
        raise RuntimeError("Direct uploads are not implemented; use a YouTube URL.")
    if job.get("source_type") != "youtube" or not job.get("source_url"):
        raise RuntimeError("Worker requires a YouTube URL.")
    with tempfile.TemporaryDirectory(prefix="clipfarm-") as tmp:
        work = Path(tmp)
        stage(job_id, "downloading", 15, "Downloading source and reading metadata.")
        run(["yt-dlp", "--no-playlist", "--restrict-filenames", "-f", "bv*+ba/b",
             "--merge-output-format", "mp4", "-o", str(work/"source.%(ext)s"), job["source_url"]])
        videos = [p for p in work.glob("source.*") if p.suffix.lower() in (".mp4",".mkv",".webm",".mov")]
        if not videos:
            raise RuntimeError("No supported downloaded video found")
        video, metadata = videos[0], None
        metadata = video_metadata(video)
        if metadata["duration"] <= 0:
            raise RuntimeError("Could not determine video duration")
        stage(job_id, "transcribing", 40, "Checking captions, then Whisper if needed.")
        segments, words = get_caption_transcript(job["source_url"]), []
        if segments:
            for row in segments:
                tokens = str(row["text"]).split()
                if tokens:
                    span = (float(row["end"])-float(row["start"]))/len(tokens)
                    words.extend({"start":float(row["start"])+i*span,
                                  "end":float(row["start"])+(i+1)*span, "text":token}
                                 for i,token in enumerate(tokens))
        else:
            segments, words = transcript_with_whisper(video, work)
        if not segments:
            raise RuntimeError("No usable transcript returned")
        transcript = "\n".join("["+str(round(x["start"],2))+"-"+str(round(x["end"],2))+"] "+x["text"] for x in segments)
        stage(job_id, "analyzing_hooks", 65, "Scoring transcript moments.", transcript=transcript[:18000])
        r = requests.post(API_BASE+"/api/analyze-transcript", headers=HEADERS,
                          json={"transcript":transcript[:18000]}, timeout=180)
        r.raise_for_status()
        candidates = sorted(r.json().get("clips", []), key=lambda c:float(c.get("hook_score",0)), reverse=True)[:MAX_CLIPS]
        if not candidates:
            raise RuntimeError("Hook analysis returned no candidates")
        stage(job_id, "cutting_rendering", 85, "Rendering vertical clips and captions.", clips=candidates)
        rendered = []
        for i, clip in enumerate(candidates, 1):
            url = render_clip(video, work, clip, i, words, metadata, job_id)
            rendered.append({**clip, "rendered_url":url, "download_url":url,
                             "status":"Finished", "source_metadata":metadata})
            stage(job_id, "cutting_rendering", min(98,85+int(13*i/len(candidates))),
                  "Rendered clip "+str(i)+" of "+str(len(candidates))+".",
                  clips=rendered+candidates[i:])
        api("POST", "/api/worker/jobs/"+job_id+"/complete",
            {"clips":rendered, "transcript":transcript[:18000], "source_metadata":metadata})

def main():
    print("ClipFarm worker started; polling "+API_BASE, flush=True)
    while True:
        job = None
        try:
            job = api("POST", "/api/worker/claim-next", {}).get("job")
            if not job:
                time.sleep(POLL_SECONDS)
                continue
            print("Processing job "+job["id"], flush=True)
            process(job)
            print("Completed job "+job["id"], flush=True)
        except Exception as exc:
            detail = str(exc)+"\n"+traceback.format_exc()
            print(detail, flush=True)
            if job and job.get("id"):
                try:
                    api("POST", "/api/worker/jobs/"+job["id"]+"/fail",
                        {"error":detail[-4000:], "stage_message":"Worker failed; see job error details."})
                except Exception as report_exc:
                    print("Could not report failure: "+str(report_exc), flush=True)
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
