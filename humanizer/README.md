# Humanizer Playwright Service

A FastAPI web service powered by Playwright running in **head mode (visible browser UI)** that interacts with `https://www.humanizeai.pro/` via internal page JavaScript context execution.

## Features

- **Continuous Listening Route**: The FastAPI server keeps running indefinitely, listening for requests.
- **Persistent Tab in Head Mode**: Chromium opens with the tab visible and stays open across requests.
- **Automatic Page Refresh Every 20 Uses**: After a page is used 20 times, the tab is closed and reopened cleanly to avoid memory leaks while retaining stealth and performance.
- **Sequential Multi-Pass Humanization**: Supports multi-pass iterative humanization (`repeat`/`repeats` parameter and `-r` CLI flag) where each pass feeds into the next.
- **Simple Payload**: Accepts `{"text": "..."}` (or `{"message": "..."}`).
- **Internal JavaScript Context Execution**: Executes in-page `fetch('https://www.humanizeai.pro/api/process', ...)` inside the authenticated browser tab.
- **Interactive CAPTCHA / Challenge Handling**: If Cloudflare or a CAPTCHA appears, the browser window is brought to front and the service waits until you solve it in the browser window, then automatically retries and returns the result.

---

## Running the Server

Start the continuous server with custom repeat and page-limit flags:
```bash
uv run python main.py -r 2 --page-limit 20
```

### Server Command-Line Flags:
- `-r`, `--repeat`, `--repeats`: Number of sequential repeat passes per incoming request (default: `1`).
- `--page-limit`, `--page-iterations`, `--page-max-uses`: Number of iterations / uses before the page is closed and reopened (default: `20`).
- `--host`: Host interface to bind to (default: `0.0.0.0`).
- `--port`: Port to listen on (default: `8000`).

---

## API Usage

### Endpoint: `POST /humanize` (or `POST /` / `POST /process`)

#### Request Body
```json
{
  "text": "Human-Computer Interaction (HCI) has developed rapidly over the past few decades. Researchers and developers continue to look for natural, simple, and low-cost ways for people to interact with computers."
}
```

#### Response Format
```json
{
  "success": true,
  "text": "The field of Human-Computer Interaction (HCI) has grown significantly in recent decades...",
  "humanized_text": "The field of Human-Computer Interaction (HCI) has grown significantly in recent decades...",
  "passes_completed": 2,
  "raw_response": {
    "result": [
      {
        "text": "The field of Human-Computer Interaction (HCI) has grown significantly in recent decades..."
      }
    ],
    "request_ref": "dad02d5e-8d26-41a2-99b4-d3f9b691d2dc"
  },
  "error": null
}
```

#### Optional Parameters
- `repeat` / `repeats`: Override repeat count for this specific request (default: server configured value)
- `style`: Rewriting style (default: `"standard"`)
- `alg`: Algorithm mode (default: `0`)
- `sessionId`: Custom session ID (auto-generated if omitted)

---

## CLI Client (`client.py`)

You can call [`client.py`](file:///home/lahirukasunidilhara/Documents/university/research/humanizer/client.py) from the command line to humanize text via the local service:

### 1. Direct text argument:
```bash
uv run python client.py "Human-Computer Interaction (HCI) has developed rapidly over the past few decades."
```

### 2. Piped input (stdin):
```bash
echo "Monocular cameras allow cost effective solutions for vision computing." | uv run python client.py
```
Or pipe from a file:
```bash
cat input.txt | uv run python client.py
```

### 3. Read from file and save to output file:
```bash
uv run python client.py -f input.txt -o humanized.txt
```

### 4. Output as full JSON:
```bash
uv run python client.py "Your text here" --json
```

