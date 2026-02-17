import json
import queue
import re
import subprocess
import sys
import tempfile
import wave
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Prompt
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, OptionList, Static

from tsr.whisper import (
    AUDIO_EXTENSIONS,
    Offsets,
    Segment,
    Timestamps,
    VIDEO_EXTENSIONS,
    TranscriptionResult,
    format_timestamp,
    transcribe,
    transcribe_with_model,
)

app = typer.Typer(name="tsr", help="transcribe audio/video using faster-whisper")
console = Console()

CONFIG_DIR = Path.home() / ".config" / "tsr"
MODELS_DIR = CONFIG_DIR / "models"
CONFIG_FILE = CONFIG_DIR / "config.json"

URL_PATTERN = re.compile(r"^https?://")
TIMESTAMP_PATTERN = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$")


class ModelSize(str, Enum):
    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    large = "large"


class OutputFormat(str, Enum):
    json = "json"
    srt = "srt"


class ModelPickerApp(App[ModelSize | None]):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit"),
    ]

    def __init__(self, current: ModelSize, downloaded: dict[ModelSize, bool]) -> None:
        super().__init__()
        self.current = current
        self.downloaded = downloaded
        self.models = list(ModelSize)

    def compose(self) -> ComposeResult:
        yield Static("Select default model (arrow keys + Enter)")
        yield OptionList(*[self._label(model) for model in self.models], id="models")
        yield Footer()

    def on_mount(self) -> None:
        option_list = self.query_one(OptionList)
        option_list.highlighted = self.models.index(self.current)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.exit(self.models[event.option_index])

    def _label(self, model: ModelSize) -> str:
        marker = "→" if model is self.current else " "
        downloaded = "✓" if self.downloaded[model] else " "
        return f"{marker} {downloaded} {get_model_name(model)}"


@dataclass(frozen=True)
class InputSource:
    path: Path
    output_base: Path


def is_url(value: str) -> bool:
    return bool(URL_PATTERN.match(value))


def get_model_name(size: ModelSize) -> str:
    if size is ModelSize.large:
        return "large-v3"
    return size.value


def parse_model_size(value: str | None) -> ModelSize:
    if value is None:
        return ModelSize.base
    if value == "large-v3":
        return ModelSize.large
    return ModelSize(value)


def is_model_cached(size: ModelSize) -> bool:
    token = f"faster-whisper-{get_model_name(size)}"
    return any(MODELS_DIR.glob(f"**/*{token}*"))


def load_config() -> dict:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return {"model": "base"}


def save_config(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")


def detect_input_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    raise typer.BadParameter(f"unsupported file type: {suffix}")


def result_to_srt(result: TranscriptionResult) -> str:
    lines = []
    for i, seg in enumerate(result.segments, 1):
        lines.append(f"{i}")
        lines.append(f"{seg.timestamps.from_} --> {seg.timestamps.to}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def result_to_plaintext(result: TranscriptionResult) -> str:
    return "\n".join(seg.text.strip() for seg in result.segments if seg.text.strip())


def parse_timestamp_to_seconds(value: str) -> float:
    match = TIMESTAMP_PATTERN.match(value)
    if not match:
        raise ValueError(f"invalid timestamp format: {value}")
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    return (hours * 3600) + (minutes * 60) + seconds + (millis / 1000)


def shift_timestamp(value: str, offset_seconds: float) -> str:
    return format_timestamp(parse_timestamp_to_seconds(value) + offset_seconds)


def write_transcription_output(
    result: TranscriptionResult,
    output_path: Path,
    output_format: OutputFormat,
) -> None:
    if output_format is OutputFormat.json:
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return
    output_path.write_text(result_to_srt(result), encoding="utf-8")


def build_shifted_segments(
    result: TranscriptionResult,
    offset_seconds: float,
    char_offset: int,
) -> tuple[list[Segment], int]:
    shifted: list[Segment] = []
    current_offset = char_offset
    for seg in result.segments:
        text = seg.text.strip()
        from_offset = current_offset
        to_offset = from_offset + len(text)
        current_offset = to_offset + (1 if text else 0)

        shifted.append(
            Segment(
                timestamps=Timestamps(
                    from_=shift_timestamp(seg.timestamps.from_, offset_seconds),
                    to=shift_timestamp(seg.timestamps.to, offset_seconds),
                ),
                offsets=Offsets(from_=from_offset, to=to_offset),
                text=text,
            )
        )
    return shifted, current_offset


def write_pcm_wav(path: Path, pcm_data: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)


def sanitize_stem(value: str) -> str:
    stem = re.sub(r"[^\w\s-]", "", value).strip().replace(" ", "_")
    return stem[:50] or "transcript"


def get_downloaded_audio_path(directory: Path) -> Path:
    audio_files = sorted(
        path
        for path in directory.glob("audio.*")
        if path.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        raise typer.BadParameter("yt-dlp did not produce a supported audio file")
    return audio_files[0]


def download_with_ytdlp(url: str, output_path: Path) -> str:
    title_result = subprocess.run(
        ["yt-dlp", "--print", "title", "--no-playlist", url],
        capture_output=True,
        text=True,
        check=True,
    )
    title = (
        title_result.stdout.strip().split("\n")[0]
        if title_result.stdout.strip()
        else "audio"
    )

    subprocess.run(
        [
            "yt-dlp",
            "--extract-audio",
            "--audio-format",
            "wav",
            "--audio-quality",
            "0",
            "--no-playlist",
            "-o",
            str(output_path),
            url,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return title


def resolve_local_input(path: Path) -> InputSource:
    if not path.exists():
        raise typer.BadParameter(f"file not found: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"not a file: {path}")

    input_type = detect_input_type(path)
    console.print(f"[blue]detected {input_type}: {path.name}[/]")
    return InputSource(path=path, output_base=path.with_suffix(""))


def resolve_url_input(url: str, stack: ExitStack) -> InputSource:
    console.print("[blue]downloading audio from URL...[/]")
    temp_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="tsr_")))
    output_template = temp_dir / "audio.%(ext)s"

    with Progress(
        SpinnerColumn(finished_text="✔"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("fetching audio", total=1)
        title = download_with_ytdlp(url, output_template)
        progress.update(task, completed=1)

    audio_path = get_downloaded_audio_path(temp_dir)
    input_type = detect_input_type(audio_path)
    console.print(f"[green]downloaded:[/] {title}")
    console.print(f"[blue]detected {input_type}: {audio_path.name}[/]")
    return InputSource(path=audio_path, output_base=Path(sanitize_stem(title)))


def resolve_input_source(value: str, stack: ExitStack) -> InputSource:
    if is_url(value):
        return resolve_url_input(value, stack)
    return resolve_local_input(Path(value))


def ensure_model_cached(size: ModelSize) -> str:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = get_model_name(size)

    if is_model_cached(size):
        console.print(f"[yellow]model {model_name} is already cached[/]")
        return model_name

    console.print(f"[blue]downloading {model_name} model...[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"downloading {model_name}", total=None)
        WhisperModel(
            model_name,
            device="auto",
            compute_type="int8",
            download_root=str(MODELS_DIR),
        )

        progress.update(task, completed=True)

    console.print(f"[green]cached model {model_name}[/]")
    return model_name


def pick_model_interactive(current: ModelSize) -> ModelSize:
    downloaded = {model: is_model_cached(model) for model in ModelSize}

    if sys.stdin.isatty() and sys.stdout.isatty():
        selected = ModelPickerApp(current, downloaded).run()
        return selected or current

    console.print("available models:")
    for model in ModelSize:
        marker = "→" if model is current else " "
        mark_downloaded = "✓" if downloaded[model] else " "
        console.print(f"  {marker} {mark_downloaded} {get_model_name(model)}")
    selected = Prompt.ask(
        "\nselect model",
        choices=[model.value for model in ModelSize],
        default=current.value,
        console=console,
    )
    return ModelSize(selected)


@app.command()
def download(
    size: Annotated[
        ModelSize,
        typer.Argument(help="model size to download"),
    ] = ModelSize.base,
):
    ensure_model_cached(size)


@app.command()
def model(
    size: Annotated[
        ModelSize | None,
        typer.Argument(help="set default model size"),
    ] = None,
):
    config = load_config()
    current = parse_model_size(config.get("model"))

    if size is None:
        selected = pick_model_interactive(current)
        config["model"] = selected.value
        save_config(config)
        console.print(f"[green]default model set to {selected.value}[/]")
        return

    config["model"] = size.value
    save_config(config)
    console.print(f"[green]default model set to {size.value}[/]")


def transcribe_with_progress(input_path: Path, model_name: str) -> TranscriptionResult:
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("transcribing", total=100)

        def on_progress(pct: int) -> None:
            progress.update(task, completed=pct)

        result = transcribe(
            input_path,
            model_name=model_name,
            model_cache_dir=MODELS_DIR,
            on_progress=on_progress,
        )
        progress.update(task, completed=100)
        return result


def get_default_output_path(source: InputSource, output_format: OutputFormat) -> Path:
    extension = ".json" if output_format is OutputFormat.json else ".srt"
    return source.output_base.with_suffix(extension)


@app.command()
def run(
    input_source: Annotated[
        str,
        typer.Argument(help="audio/video file or URL to transcribe"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="output file path"),
    ] = None,
    format: Annotated[
        OutputFormat,
        typer.Option("-f", "--format", help="output format: json or srt"),
    ] = OutputFormat.srt,
    model_size: Annotated[
        ModelSize | None,
        typer.Option("-m", "--model", help="model size to use"),
    ] = None,
    plain: Annotated[
        bool,
        typer.Option(
            "--plain",
            help="print plaintext transcript to stdout and do not write output file",
        ),
    ] = False,
):
    with ExitStack() as stack:
        source = resolve_input_source(input_source, stack)
        config = load_config()
        size = model_size or parse_model_size(config.get("model"))
        result = transcribe_with_progress(source.path, get_model_name(size))

    if plain:
        console.print(result_to_plaintext(result), markup=False)
        return

    output_path = output or get_default_output_path(source, format)
    write_transcription_output(result, output_path, format)
    console.print(f"[green]saved transcript to {output_path}[/]")


@app.command()
def watch(
    wav_output: Annotated[
        Path,
        typer.Option(
            "--wav-output",
            help="where to write the live microphone recording (.wav)",
        ),
    ] = Path("watch.wav"),
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="transcript output path"),
    ] = None,
    format: Annotated[
        OutputFormat,
        typer.Option("-f", "--format", help="output format: json or srt"),
    ] = OutputFormat.srt,
    model_size: Annotated[
        ModelSize | None,
        typer.Option("-m", "--model", help="model size to use"),
    ] = None,
    chunk_seconds: Annotated[
        float,
        typer.Option(
            "--chunk-seconds",
            min=1.0,
            help="transcribe every N seconds while recording",
        ),
    ] = 6.0,
    sample_rate: Annotated[
        int,
        typer.Option(
            "--sample-rate",
            min=8000,
            help="recording sample rate (Hz)",
        ),
    ] = 16000,
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            help="sounddevice input device name or numeric id",
        ),
    ] = None,
    plain: Annotated[
        bool,
        typer.Option(
            "--plain/--no-plain",
            help="print each transcribed chunk to stdout",
        ),
    ] = True,
):
    import sounddevice as sd

    config = load_config()
    size = model_size or parse_model_size(config.get("model"))
    model_name = ensure_model_cached(size)
    model = WhisperModel(
        model_name,
        device="auto",
        compute_type="int8",
        download_root=str(MODELS_DIR),
    )

    wav_output.parent.mkdir(parents=True, exist_ok=True)
    output_path = output or wav_output.with_suffix(
        ".json" if format is OutputFormat.json else ".srt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bytes_per_frame = 2
    target_chunk_frames = max(1, int(chunk_seconds * sample_rate))
    audio_queue: queue.Queue[bytes] = queue.Queue()

    selected_device: int | str | None = device
    if isinstance(selected_device, str) and selected_device.isdigit():
        selected_device = int(selected_device)

    def callback(
        indata: object, frames: int, time_info: object, status: object
    ) -> None:
        del frames, time_info, status
        audio_queue.put(bytes(indata))

    all_segments: list[Segment] = []
    language = "unknown"
    char_offset = 0
    timeline_offset_seconds = 0.0
    chunk_index = 0
    chunk_buffer = bytearray()
    chunk_frames = 0

    console.print(
        "[blue]watch started. Recording from microphone... press Ctrl+C to stop.[/]"
    )
    console.print(f"[blue]audio file:[/] {wav_output}")
    console.print(
        f"[blue]chunk interval:[/] {chunk_seconds:.1f}s, [blue]model:[/] {model_name}"
    )

    with tempfile.TemporaryDirectory(prefix="tsr_watch_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        def transcribe_chunk(pcm_data: bytes) -> None:
            nonlocal language, char_offset, timeline_offset_seconds, chunk_index

            chunk_index += 1
            chunk_path = temp_dir / f"chunk_{chunk_index:05d}.wav"
            write_pcm_wav(chunk_path, pcm_data, sample_rate)
            chunk_result = transcribe_with_model(chunk_path, model=model)

            language = chunk_result.language

            shifted_segments, next_offset = build_shifted_segments(
                chunk_result,
                offset_seconds=timeline_offset_seconds,
                char_offset=char_offset,
            )
            all_segments.extend(shifted_segments)
            char_offset = next_offset

            if plain:
                text = result_to_plaintext(chunk_result)
                if text:
                    console.print(text, markup=False)

            timeline_offset_seconds += len(pcm_data) / (bytes_per_frame * sample_rate)

            current_result = TranscriptionResult(
                language=language, segments=all_segments
            )
            write_transcription_output(current_result, output_path, format)

        with open(wav_output, "wb") as wav_raw_file:
            with wave.open(wav_raw_file, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)

                try:
                    with sd.RawInputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype="int16",
                        callback=callback,
                        device=selected_device,
                    ):
                        while True:
                            pcm_data = audio_queue.get()

                            wav_file.writeframes(pcm_data)
                            wav_raw_file.flush()

                            chunk_buffer.extend(pcm_data)
                            chunk_frames += len(pcm_data) // bytes_per_frame

                            while chunk_frames >= target_chunk_frames:
                                split_bytes = target_chunk_frames * bytes_per_frame
                                chunk_pcm = bytes(chunk_buffer[:split_bytes])
                                del chunk_buffer[:split_bytes]
                                chunk_frames -= target_chunk_frames
                                transcribe_chunk(chunk_pcm)

                except KeyboardInterrupt:
                    console.print("\n[blue]stopping watch...[/]")

                if chunk_buffer:
                    transcribe_chunk(bytes(chunk_buffer))

    final_result = TranscriptionResult(language=language, segments=all_segments)
    write_transcription_output(final_result, output_path, format)
    console.print(f"[green]saved recording to {wav_output}[/]")
    console.print(f"[green]saved transcript to {output_path}[/]")


def main():
    app()


if __name__ == "__main__":
    main()
