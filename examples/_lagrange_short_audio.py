"""Audio synthesis helpers for the narrated Lagrange example."""

import shutil
import subprocess
import wave


def _synthesize_clip(executable, directory, index, sentence, speed):
    path = directory / f"voice_{index:02d}.wav"
    subprocess.run(
        [
            executable,
            "-v",
            "en-us+m3",
            "-s",
            str(speed),
            "-p",
            "44",
            "-a",
            "165",
            "-w",
            str(path),
            sentence,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    with wave.open(str(path), "rb") as clip:
        params = clip.getparams()
        samples = clip.readframes(clip.getnframes())
    duration = len(samples) / (params.framerate * params.nchannels * params.sampwidth)
    return samples, params, duration


def _validate_wave_format(current, expected):
    if current is not None and (
        current.nchannels,
        current.sampwidth,
        current.framerate,
    ) != (expected.nchannels, expected.sampwidth, expected.framerate):
        raise RuntimeError("speech clips were synthesized with mismatched formats")


def _assemble_narration(directory, clips, durations, params, total_duration, log):
    speech_duration = sum(durations)
    if speech_duration >= total_duration - 1.0:
        raise RuntimeError(
            f"Narration is too long ({speech_duration:.1f}s); increase speech speed"
        )
    pause = (total_duration - speech_duration) / len(clips)
    log(
        f"Narration totals {speech_duration:.1f}s; inserting {pause:.1f}s "
        f"pauses to fill {total_duration:.0f}s"
    )
    silence = bytes(
        round(pause * params.framerate) * params.nchannels * params.sampwidth
    )
    starts, audio, elapsed = [], bytearray(), 0.0
    for samples, duration in zip(clips, durations):
        starts.append(elapsed)
        audio.extend(samples)
        audio.extend(silence)
        elapsed += duration + pause
    path = directory / "lagrange_intuition_narration.wav"
    with wave.open(str(path), "wb") as output:
        output.setparams(params._replace(nframes=0))
        output.writeframes(audio)
    return path, starts, pause


def speak_clips(directory, narration, total_duration, log, speed=150):
    """Synthesize narration clips and return their timings and sample format."""
    executable = shutil.which("espeak-ng")
    if executable is None:
        raise RuntimeError("espeak-ng is required to render the narrated short")
    clips, durations, wave_params = [], [], None
    log(f"Synthesizing {len(narration)} narration clips with espeak-ng")
    for index, sentence in enumerate(narration):
        log(f"Narration {index + 1}/{len(narration)}: starting synthesis")
        samples, params, duration = _synthesize_clip(
            executable, directory, index, sentence, speed
        )
        _validate_wave_format(wave_params, params)
        wave_params = params if wave_params is None else wave_params
        clips.append(samples)
        durations.append(duration)
        log(f"Narration {index + 1}/{len(narration)}: {duration:.1f}s audio ready")
    narration_path, starts, pause = _assemble_narration(
        directory, clips, durations, wave_params, total_duration, log
    )
    return narration_path, starts, durations, pause
