# Humanizer Playwright Service

A FastAPI web service powered by Playwright running in **head mode (visible browser UI)** that interacts with `https://www.humanizeai.pro/` via internal page JavaScript context execution.

## Features

- **Continuous Listening Route**: The FastAPI server keeps running indefinitely, listening for requests.
- **Persistent Tab in Head Mode**: Chromium opens with the tab visible and **keeps the tab open** across all requests.
- **Simple Payload**: Accepts `{"text": "..."}` (or `{"message": "..."}`).
- **Internal JavaScript Context Execution**: Executes in-page `fetch('https://www.humanizeai.pro/api/process', ...)` inside the already open, authenticated browser tab.
- **Interactive CAPTCHA / Challenge Handling**: If Cloudflare or a CAPTCHA appears, the browser window is brought to front and the service waits until you solve it in the browser window, then automatically retries and returns the result.

---

## Running the Server

Start the continuous server:
```bash
uv run python main.py
```
*(Or `uv run uvicorn main:app --host 0.0.0.0 --port 8000`)*

The browser window will open in head mode and stay open at `https://www.humanizeai.pro/`.

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
- `style`: Rewriting style (default: `"standard"`)
- `alg`: Algorithm mode (default: `0`)
- `sessionId`: Custom session ID (auto-generated if omitted)

#### Example Response
```json
{
  "success": true,
  "humanized_text": "The field of Human-Computer Interaction (HCI) has grown significantly in recent decades...",
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

