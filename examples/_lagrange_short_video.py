"""Video encoding and subtitle helpers for the narrated Lagrange example."""

import shutil
import subprocess
from pathlib import Path
from time import monotonic


def timestamp(seconds):
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def write_subtitles(path, narration, starts, durations):
    with Path(path).open("w", encoding="utf-8") as subtitles:
        for index, (text, start, duration) in enumerate(
            zip(narration, starts, durations), start=1
        ):
            subtitles.write(
                f"{index}\n{timestamp(start)} --> "
                f"{timestamp(start + duration)}\n{text}\n\n"
            )


def select_video_encoder():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the video")
    encoders = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    available = {
        line.split()[1]
        for line in encoders.splitlines()
        if len(line.split()) >= 2 and line.split()[0][:1] in {"V", "."}
    }
    codec = next(
        (
            candidate
            for candidate in ("libx264", "libopenh264", "mpeg4")
            if candidate in available
        ),
        None,
    )
    if codec is None:
        raise RuntimeError(
            "This FFmpeg build has no supported video encoder "
            "(tried libx264, libopenh264, and mpeg4). Install an FFmpeg build "
            "with one of these encoders and rerun the render."
        )
    return ffmpeg, codec


def encode_silent_video(frame_directory, output, ffmpeg, codec, fps, log):
    log(f"Encoding {fps} fps silent MP4 from {frame_directory} with {codec}")
    encode_started = monotonic()
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(frame_directory / "frame_%06d.png"),
                "-c:v",
                codec,
                "-b:v",
                "5000k",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() if error.stderr else str(error)
        raise RuntimeError(f"FFmpeg failed to encode the silent MP4: {detail}") from error
    log(f"Silent MP4 encoded in {monotonic() - encode_started:.1f}s: {output}")


def mux_audio(video_path, audio_path, output_path, duration, log):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to mux the narration into the short")
    log(f"Muxing narration into {output_path}")
    mux_started = monotonic()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-t",
            str(duration),
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )
    log(f"Audio mux completed in {monotonic() - mux_started:.1f}s")
