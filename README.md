# tsr

CLI tool to transcribe audio or video files using `whisper.cpp`.

## Requirements

- Python `>=3.13`
- `ffmpeg` available on `PATH` (required for video inputs)
- `whisper-cli` available on `PATH` (from `whisper.cpp`)

## Install

```bash
uv sync
```

Run with:

```bash
uv run tsr --help
```

## Quick Start

1. Download a model:

```bash
uv run tsr download base
```

2. (Optional) Set default model:

```bash
uv run tsr model base
```

3. Transcribe a file:

```bash
uv run tsr run path/to/audio_or_video.mp4
```

This writes an `.srt` file by default next to the input.

## Commands

`tsr download [tiny|base|small|medium|large]`
- Downloads `ggml-<size>.bin` into `~/.config/tsr/models`.

`tsr model [size]`
- Without `size`, shows a selectable model list in terminal and saves your selection as default.
- With `size`, updates default model in `~/.config/tsr/config.json`.

`tsr run <input> [--output <path>] [--format srt|json] [--model <size>] [--plain]`
- Transcribes supported audio/video input.
- If the selected model is missing locally, it is downloaded automatically.
- If `--output` is omitted, output path is inferred from input extension.
- `--plain` prints plaintext transcript to terminal and skips writing a file.

## Supported Input Types

- Audio: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.opus`
- Video: `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`

## Notes

- Video files are converted to mono 16kHz WAV with `ffmpeg` before transcription.
- Missing or invalid model files will cause command failure.
- Subprocess failures from `ffmpeg` or `whisper-cli` are surfaced directly.
