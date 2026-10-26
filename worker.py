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

API_BASE = os.getenv("CLIPFARM_API_BASE", "https://placeholder-api.com").rstrip("/")
WORKER_TOKEN = os.getenv("CLIPFARM_WORKER_TOKEN", "dummy-token")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "sk-proj-dummyplaceholderkey"
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "dummy")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "dummy")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "dummy")
R2_BUCKET = os.getenv("R2_BUCKET", "dummy")
R2_PUBLIC_BASE_URL = os.getenv("R2_PUBLIC_BASE_URL", "https://placeholder.com").rstrip("/")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "4"))
MAX_CLIPS = int(os.getenv("MAX_CLIPS", "5"))

HEADERS = {
    "Authorization": "Bearer " + WORKER_TOKEN,
    "Content-Type": "application/json",
}

client = OpenAI(api_key=OPENAI_API_KEY)

s3 = boto3.client(
    "s3",
    endpoint_url="https://" + R2_ACCOUNT_ID + ".r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
    config=Config(signature_version="s3v4"),
)


def api(method, path, payload=None):
    response = requests.request(
        method,
        API_BASE + path,
        headers=HEADERS,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def stage(job_id, status, progress, message, **extra):
    try:
        api(
            "POST",
            "/api/worker/jobs/" + job_id + "/progress",
            {
                "status": status,
                "progress": progress,
                "stage_message": message,
                **extra,
            },
        )
    except Exception:
        pass


def run(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed (" + str(result.returncode) + "): "
            + " ".join(cmd) + "\n" + result.stderr[-6000:]
        )
    return result.stdout


def video_metadata(path):
    data = json.loads(run([
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height,r_frame_rate",
        "-of", "json",
        str(path),
    ]))

    video = next(
        (s for s in data.get("streams", [])
         if s.get("codec_type") == "video"),
        {},
    )

    return {
        "duration": float(data.get("format", {}).get("duration") or 0),
        "width": video.get("width"),
        "height": video.get("height"),
        "framerate": video.get("r_frame_rate"),
    }


def youtube_id(url):
    match = re.search(
        r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})",
        url,
    )
    return match.group(1) if match else None


def get_caption_transcript(url):
    video_id = youtube_id(url)
    if not video_id:
        return None

    try:
        fetched = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en", "en-US", "en-GB"],
        )
        rows = [
            {
                "start": float(item.start),
                "end": float(item.start + item.duration),
                "text": str(item.text),
            }
            for item in fetched
        ]
        return rows or None
    except Exception:
        return None


def transcript_with_whisper(video_path, work):
    audio = work / "audio.mp3"

    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "64k",
        str(audio),
    ])

    with audio.open("rb") as file:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=file,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

    segments = [
        {
            "start": float(item.start),
            "end": float(item.end),
            "text": str(item.text).strip(),
        }
        for item in (getattr(result, "segments", None) or [])
    ]

    words = [
        {
            "start": float(item.start),
            "end": float(item.end),
            "text": str(item.word),
        }
        for item in (getattr(result, "words", None) or [])
    ]

    return segments, words


def ass_time(seconds):
    centiseconds = max(0, int(round(seconds * 100)))
    return "%d:%02d:%02d.%02d" % (
        centiseconds // 360000,
        (centiseconds // 6000) % 60,
        (centiseconds // 100) % 60,
        centiseconds % 100,
    )


def build_ass(words, offset, duration, out_path):
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "ScaledBorderAndShadow: yes",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, "
            "SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
            "MarginV, Encoding"
        ),
        (
            "Style: Default,Arial,64,&H00FFFFFF,&H0000D7FF,"
            "&H00101010,&H90000000,-1,0,1,5,1,2,80,80,240,1"
        ),
        "[Events]",
        (
            "Format: Layer, Start, End, Style, Name, MarginL, "
            "MarginR, MarginV, Effect, Text"
        ),
    ]

    for word in words:
        start = float(word["start"]) - offset
        end = float(word["end"]) - offset

        if end <= 0 or start >= duration:
            continue

        text = (
            str(word["text"])
            .replace("{", "(")
            .replace("}", ")")
            .replace("\\", "")
        )

        lines.append(
            "Dialogue: 0,"
            + ass_time(max(0, start))
            + ","
            + ass_time(min(duration, end))
            + ",Default,,0,0,0,,{\\fad(70,70)}"
            + text
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def render_clip(video, work, clip, index, words, metadata, job_id):
    start = max(0.0, float(clip["start_time"]))
    end = float(clip["end_time"])

    if end <= start or end - start > 90:
        raise ValueError("Invalid clip interval: " + str(start) + "-" + str(end))

    end = min(end, float(metadata["duration"]) or end)
    duration = end - start

    subtitle_file = work / ("clip-" + str(index) + ".ass")
    output = work / ("clip-" + str(index) + ".mp4")

    build_ass(words, start, duration, subtitle_file)

    video_filter = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,setsar=1,subtitles='"
        + str(subtitle_file).replace("'", "\\'")
        + "'"
    )

    run([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video),
        "-t", str(duration),
        "-vf", video_filter,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output),
    ])

    key = "jobs/" + job_id + "/clip-" + str(index) + ".mp4"

    s3.upload_file(
        str(output),
        R2_BUCKET,
        key,
        ExtraArgs={
            "ContentType": "video/mp4",
            "CacheControl": "public, max-age=3600",
        },
    )

    stored = s3.head_object(Bucket=R2_BUCKET, Key=key)
    if int(stored.get("ContentLength", 0)) <= 0:
        raise RuntimeError("Uploaded MP4 is empty.")

    url = R2_PUBLIC_BASE_URL + "/" + key
    return url


def process(job):
    job_id = job["id"]

    if job.get("source_type") == "upload":
        raise RuntimeError("Direct MP4 uploads are not implemented. Use a YouTube URL.")

    if job.get("source_type") != "youtube" or not job.get("source_url"):
        raise RuntimeError("Worker requires a YouTube URL source.")

    with tempfile.TemporaryDirectory(prefix="clipfarm-") as temp:
        work = Path(temp)

        stage(job_id, "downloading", 15, "Downloading source and reading metadata.")

        run([
            "yt-dlp",
            "--no-playlist",
            "--restrict-filenames",
            "-f", "bv*+ba/b",
            "--merge-output-format", "mp4",
            "-o", str(work / "source.%(ext)s"),
            job["source_url"],
        ])

        videos = [
            path for path in work.glob("source.*")
            if path.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")
        ]

        if not videos:
            raise RuntimeError("No supported downloaded video was found.")

        video = videos[0]
        metadata = video_metadata(video)

        if metadata["duration"] <= 0:
            raise RuntimeError("Could not determine source video duration.")

        stage(job_id, "transcribing", 40, "Checking captions or using Whisper.")

        segments = get_caption_transcript(job["source_url"])
        words = []

        if segments:
            for segment in segments:
                tokens = str(segment["text"]).split()
                if not tokens:
                    continue

                span = (float(segment["end"]) - float(segment["start"])) / len(tokens)

                for index, token in enumerate(tokens):
                    words.append({
                        "start": float(segment["start"]) + index * span,
                        "end": float(segment["start"]) + (index + 1) * span,
                        "text": token,
                    })
        else:
            segments, words = transcript_with_whisper(video, work)

        if not segments:
            raise RuntimeError("No usable transcript was returned.")

        transcript_text = "\n".join(
            "[" + str(round(item["start"], 2)) + "-" + str(round(item["end"], 2)) + "] " + item["text"]
            for item in segments
        )

        stage(job_id, "analyzing_hooks", 65, "Scoring transcript moments.", transcript=transcript_text[:18000])

        response = requests.post(
            API_BASE + "/api/analyze-transcript",
            headers=HEADERS,
            json={"transcript": transcript_text[:18000]},
            timeout=180,
        )
        response.raise_for_status()

        candidates = sorted(
            response.json().get("clips", []),
            key=lambda item: float(item.get("hook_score", 0)),
            reverse=True,
        )[:MAX_CLIPS]

        if not candidates:
            raise RuntimeError("Hook analysis returned no clip candidates.")

        stage(job_id, "cutting_rendering", 85, "Rendering vertical clips and captions.", clips=candidates)

        rendered = []

        for index, clip in enumerate(candidates, 1):
            url = render_clip(video, work, clip, index, words, metadata, job_id)

            rendered.append({
                **clip,
                "rendered_url": url,
                "download_url": url,
                "status": "Finished",
                "source_metadata": metadata,
            })

            stage(
                job_id,
                "cutting_rendering",
                min(98, 85 + int(13 * index / len(candidates))),
                "Rendered clip " + str(index) + " of " + str(len(candidates)) + ".",
                clips=rendered + candidates[index:],
            )

        api(
            "POST",
            "/api/worker/jobs/" + job_id + "/complete",
            {
                "clips": rendered,
                "transcript": transcript_text[:18000],
                "source_metadata": metadata,
            },
        )


def main():
    print("ClipFarm worker started; polling " + API_BASE, flush=True)

    while True:
        job = None

        try:
            job_resp = api("POST", "/api/worker/claim-next", {})
            job = job_resp.get("job") if job_resp else None

            if not job:
                time.sleep(POLL_SECONDS)
                continue

            print("Processing job " + job["id"], flush=True)
            process(job)
            print("Completed job " + job["id"], flush=True)

        except Exception as exc:
            detail = str(exc) + "\n" + traceback.format_exc()
            print(detail, flush=True)

            if job and job.get("id"):
                try:
                    api(
                        "POST",
                        "/api/worker/jobs/" + job["id"] + "/fail",
                        {
                            "error": detail[-4000:],
                            "stage_message": "Worker failed; see job error details.",
                        },
                    )
                except Exception:
                    pass

            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
