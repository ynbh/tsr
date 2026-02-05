import json
import subprocess
import tempfile
from pathlib import Path

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


def transcribe(
    audio_path: str | Path,
    model_path: str | Path,
    whisper_bin: str | Path = "whisper-cli",
) -> TranscriptionResult:
    audio_path = Path(audio_path)
    model_path = Path(model_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        if audio_path.suffix.lower() in VIDEO_EXTENSIONS:
            wav_path = tmpdir / "audio.wav"
            extract_audio(audio_path, wav_path)
            audio_path = wav_path

        output_base = tmpdir / "output"

        subprocess.run(
            [
                str(whisper_bin),
                "-m",
                str(model_path),
                "-f",
                str(audio_path),
                "-oj",
                "-of",
                str(output_base),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        json_file = Path(f"{output_base}.json")
        data = json.loads(json_file.read_text(encoding="utf-8"))

    return TranscriptionResult(
        language=data["result"]["language"],
        segments=[Segment(**seg) for seg in data["transcription"]],
    )
