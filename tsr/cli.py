import json
import re
import subprocess
import sys
import tempfile
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
    VIDEO_EXTENSIONS,
    TranscriptionResult,
    transcribe,
)

app = typer.Typer(name="tsr", help="transcribe audio/video using faster-whisper")
console = Console()

CONFIG_DIR = Path.home() / ".config" / "tsr"
MODELS_DIR = CONFIG_DIR / "models"
CONFIG_FILE = CONFIG_DIR / "config.json"

URL_PATTERN = re.compile(r"^https?://")


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
    if value == "large-v3":
        return ModelSize.large
    if value is None:
        return ModelSize.base
    try:
        return ModelSize(value)
    except ValueError:
        return ModelSize.base


def is_model_cached(size: ModelSize) -> bool:
    if not MODELS_DIR.exists():
        return False
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
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("fetching audio", total=None)
        title = download_with_ytdlp(url, output_template)

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
        if not is_model_cached(size):
            resolved_name = get_model_name(size)
            console.print(
                f"[yellow]model {resolved_name} not in cache yet; "
                "it will be downloaded on first run[/]"
            )
        result = transcribe_with_progress(source.path, get_model_name(size))

    if plain:
        console.print(result_to_plaintext(result), markup=False)
        return

    output_path = output or get_default_output_path(source, format)
    if format is OutputFormat.json:
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    else:
        output_path.write_text(result_to_srt(result), encoding="utf-8")
    console.print(f"[green]saved transcript to {output_path}[/]")


def main():
    app()


if __name__ == "__main__":
    main()
