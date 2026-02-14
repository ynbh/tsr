import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from faster_whisper import WhisperModel
from pydantic import BaseModel, ConfigDict, Field


class Timestamps(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(alias="from")
    to: str


class Offsets(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: int = Field(alias="from")
    to: int


class Segment(BaseModel):
    timestamps: Timestamps
    offsets: Offsets
    text: str


class TranscriptionResult(BaseModel):
    language: str
    segments: list[Segment]


AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm"})


def extract_audio(video_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def get_media_duration_seconds(path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    value = result.stdout.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def transcribe(
    audio_path: str | Path,
    model_name: str,
    model_cache_dir: str | Path | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> TranscriptionResult:
    source_path = Path(audio_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        if source_path.suffix.lower() in VIDEO_EXTENSIONS:
            wav_path = tmpdir_path / "audio.wav"
            extract_audio(source_path, wav_path)
            source_path = wav_path

        duration = get_media_duration_seconds(source_path)
        model = WhisperModel(
            model_name,
            device="auto",
            compute_type="int8",
            download_root=str(model_cache_dir) if model_cache_dir else None,
        )

        if on_progress:
            on_progress(1)

        segments_iter, info = model.transcribe(str(source_path), vad_filter=True)
        segments: list[Segment] = []
        char_offset = 0
        last_progress = 1

        for raw_segment in segments_iter:
            text = raw_segment.text.strip()
            start = float(raw_segment.start)
            end = float(raw_segment.end)
            if end < start:
                end = start

            from_offset = char_offset
            to_offset = from_offset + len(text)
            char_offset = to_offset
            if text:
                char_offset += 1

            segments.append(
                Segment(
                    timestamps=Timestamps(
                        from_=format_timestamp(start),
                        to=format_timestamp(end),
                    ),
                    offsets=Offsets(from_=from_offset, to=to_offset),
                    text=text,
                )
            )

            if on_progress and duration and duration > 0:
                progress = int(min(99, max(1, (end / duration) * 100)))
                if progress > last_progress:
                    on_progress(progress)
                    last_progress = progress

    if on_progress:
        on_progress(100)

    return TranscriptionResult(
        language=(info.language or "unknown"),
        segments=segments,
    )
