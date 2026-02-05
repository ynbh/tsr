import json
from enum import Enum
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

from tsr.whisper import (
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    TranscriptionResult,
    transcribe,
)

app = typer.Typer(name="tsr", help="transcribe audio/video using whisper.cpp")
console = Console()

CONFIG_DIR = Path.home() / ".config" / "tsr"
MODELS_DIR = CONFIG_DIR / "models"
CONFIG_FILE = CONFIG_DIR / "config.json"

HUGGINGFACE_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"


class ModelSize(str, Enum):
    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    large = "large"


class OutputFormat(str, Enum):
    json = "json"
    srt = "srt"


def get_model_url(size: ModelSize) -> str:
    return f"{HUGGINGFACE_BASE}/ggml-{size.value}.bin"


def get_model_path(size: ModelSize) -> Path:
    return MODELS_DIR / f"ggml-{size.value}.bin"


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


def ensure_model_downloaded(size: ModelSize) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = get_model_path(size)

    if model_path.exists():
        return model_path

    url = get_model_url(size)
    console.print(f"[blue]downloading {size.value} model...[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"downloading ggml-{size.value}.bin", total=None)

        with httpx.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)

        progress.update(task, completed=True)

    console.print(f"[green]saved to {model_path}[/]")
    return model_path


@app.command()
def download(
    size: Annotated[
        ModelSize,
        typer.Argument(help="model size to download"),
    ] = ModelSize.base,
):
    model_path = get_model_path(size)
    if model_path.exists():
        console.print(f"[yellow]model {size.value} already exists at {model_path}[/]")
        return

    ensure_model_downloaded(size)


@app.command()
def model(
    size: Annotated[
        ModelSize | None,
        typer.Argument(help="set default model size"),
    ] = None,
):
    config = load_config()

    if size is None:
        current = config.get("model", "base")
        console.print("available models:")
        for m in ModelSize:
            marker = "→" if m.value == current else " "
            downloaded = "✓" if get_model_path(m).exists() else " "
            console.print(f"  {marker} {downloaded} {m.value}")
        selected = Prompt.ask(
            "\nselect model",
            choices=[m.value for m in ModelSize],
            default=current,
            console=console,
        )
        config["model"] = selected
        save_config(config)
        console.print(f"[green]default model set to {selected}[/]")
        return

    config["model"] = size.value
    save_config(config)
    console.print(f"[green]default model set to {size.value}[/]")


@app.command()
def run(
    input_path: Annotated[
        Path,
        typer.Argument(
            help="audio or video file to transcribe",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
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
    input_type = detect_input_type(input_path)
    console.print(f"[blue]detected {input_type}: {input_path.name}[/]")

    config = load_config()
    size = model_size or ModelSize(config.get("model", "base"))
    if not get_model_path(size).exists():
        console.print(f"[yellow]model {size.value} not found, downloading now...[/]")
    model_path = ensure_model_downloaded(size)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("transcribing...", total=None)
        result = transcribe(input_path, model_path)

    if plain:
        console.print(result_to_plaintext(result), markup=False)
        return

    if output is None:
        ext = ".json" if format is OutputFormat.json else ".srt"
        output = input_path.with_suffix(ext)

    if format is OutputFormat.json:
        output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    else:
        output.write_text(result_to_srt(result), encoding="utf-8")

    console.print(f"[green]saved transcript to {output}[/]")


def main():
    app()


if __name__ == "__main__":
    main()
