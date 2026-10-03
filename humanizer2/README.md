# Humanizer 2 (Humanize.io Backend)

A fast, single-pass local humanizer service powered by [humanize.io](https://humanize.io/).

## Features
- **Direct Async Workflow**: Sends text via `/api/trpc/byPass.generate?batch=1` to obtain a task ID, then polls `/api/trpc/byPass.retrieval?batch=1` every 4 seconds until the humanized output is returned.
- **Single-Pass Simplicity**: Clean, direct execution without multi-pass loops or complex manipulations.
- **Browser Session Management**: Runs Chromium (headless by default or `--head` for visible mode) to maintain valid NextAuth session tokens automatically.
- **Non-Conflicting Port**: Runs on port `8001` by default (allowing `humanizer` and `humanizer2` to run concurrently).

## Quick Start

### 1. Start the Server
```bash
cd humanizer2
uv run python main.py
```
*(Optionally pass `--port 8001` or `--head` to display the browser window).*

### 2. Send Text via CLI Client
```bash
# Direct text
uv run python client.py "Your AI generated text here..."

# From file to output file
uv run python client.py -f input.txt -o output.txt

# Piped via stdin
cat input.txt | uv run python client.py
```

### 3. API Usage
```bash
curl -X POST http://127.0.0.1:8001/humanize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here...", "modelType": "latest"}'
```
