import asyncio
import json
import logging
import os
import secrets
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import BrowserContext, Page, async_playwright
from pydantic import BaseModel, Field
import uvicorn

# Terminal ANSI Color Codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BLUE = "\033[94m"
RED = "\033[91m"
MAGENTA = "\033[95m"

MAX_WORDS = 200


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        timestamp = self.formatTime(record, self.datefmt)
        level = record.levelname
        msg = record.getMessage()
        if record.levelno >= logging.ERROR:
            return f"{BOLD}{RED}[{timestamp}] [ERROR]{RESET} {msg}"
        elif record.levelno >= logging.WARNING:
            return f"{BOLD}{YELLOW}[{timestamp}] [WARN]{RESET} {msg}"
        elif record.levelno == logging.INFO:
            return f"{BOLD}{CYAN}[{timestamp}] [INFO]{RESET} {msg}"
        return f"[{timestamp}] [{level}] {msg}"


logger = logging.getLogger("humanizer2")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter(datefmt="%H:%M:%S"))
logger.handlers = [console_handler]


class HumanizeRequest(BaseModel):
    text: Optional[str] = Field(None, description="Text to humanize")
    message: Optional[str] = Field(None, description="Alias for text")
    modelType: Optional[str] = Field("latest", description="Model type (e.g. latest)")


class HumanizeResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    humanized_text: Optional[str] = None
    task_id: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BrowserManager:
    def __init__(self, headless: bool = False):
        self._playwright = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._temp_dir: Optional[str] = None
        self._current_browser_id: Optional[str] = None
        self._lock = asyncio.Lock()
        self._request_count = 0
        self.headless = headless

    async def _destroy_current_session(self):
        """Closes browser context and completely deletes temporary user data directory from disk."""
        try:
            if self._page and not self._page.is_closed():
                await self._page.close()
            if self._context:
                await self._context.close()
        except Exception as e:
            logger.warning(f"Error closing browser context: {e}")
        finally:
            self._page = None
            self._context = None
            if self._temp_dir and os.path.exists(self._temp_dir):
                try:
                    shutil.rmtree(self._temp_dir, ignore_errors=True)
                    logger.info(f"{YELLOW}[DATA WIPE] Completely destroyed browser data directory: {self._temp_dir}{RESET}")
                except Exception as e:
                    logger.warning(f"Failed to delete {self._temp_dir}: {e}")
            self._temp_dir = None

    async def _open_fresh_session(self):
        """Launches a brand new browser instance with a random browserId fingerprint and clean profile."""
        await self._destroy_current_session()

        self._temp_dir = tempfile.mkdtemp(prefix="humanizer_clean_")
        self._current_browser_id = secrets.token_hex(16)
        mode_str = "Head Mode (visible window)" if not self.headless else "Headless Mode"
        logger.info(f"{CYAN}Launching fresh Chromium in {mode_str} with profile: {self._temp_dir} ...{RESET}")
        logger.info(f"{CYAN}Injected fresh anonymous deviceId: {self._current_browser_id}{RESET}")

        self._context = await self._playwright.chromium.launch_persistent_context(
            self._temp_dir,
            headless=self.headless,
            args=[
                "--no-sandbox",
                "--disable-infobars",
                "--start-maximized",
            ]
        )

        # Inject unique browserId into localStorage before any website script executes
        await self._context.add_init_script(f"""
            window.localStorage.setItem('browserId', '{self._current_browser_id}');
        """)

        self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()

        logger.info("Navigating fresh browser to https://humanize.io to initialize NextAuth session...")
        await self._page.goto("https://humanize.io", wait_until="networkidle", timeout=60000)
        await self._page.wait_for_timeout(1000)

        title = await self._page.title()
        logger.info(f"{BOLD}{GREEN}Fresh browser ready with new device session: '{title}'{RESET}")

    async def initialize(self):
        logger.info(f"{BOLD}{GREEN}Starting Playwright for Humanizer2 (Per-Request Restart & Dynamic Device ID)...{RESET}")
        self._playwright = await async_playwright().start()
        await self._open_fresh_session()

    async def generate_task(self, page: Page, text: str, model_type: str = "latest") -> str:
        """Calls /api/trpc/byPass.generate?batch=1 to create a task and return taskId."""
        js_code = """
        async (payload) => {
            const resp = await fetch('https://humanize.io/api/trpc/byPass.generate?batch=1', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    '0': {
                        'json': payload
                    }
                })
            });
            const status = resp.status;
            const data = await resp.json();
            return { status, ok: resp.ok, data };
        }
        """
        payload = {
            "text": text,
            "source": "web",
            "modelType": model_type,
        }
        res = await page.evaluate(js_code, payload)
        if not res.get("ok"):
            err_msg = str(res.get("data"))
            raise RuntimeError(f"byPass.generate failed with HTTP {res.get('status')}: {err_msg}")

        data = res.get("data", [])
        if not data or not isinstance(data, list):
            raise RuntimeError(f"Unexpected generate response format: {data}")

        task_id = data[0].get("result", {}).get("data", {}).get("json", {}).get("taskId")
        if not task_id:
            raise RuntimeError(f"No taskId found in generate response: {data}")

        return task_id

    async def poll_retrieval(self, page: Page, task_id: str, poll_interval: float = 4.0, max_attempts: int = 25) -> Dict[str, Any]:
        """Polls /api/trpc/byPass.retrieval?batch=1 every poll_interval seconds until finished."""
        js_code = """
        async (taskId) => {
            const resp = await fetch('https://humanize.io/api/trpc/byPass.retrieval?batch=1', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    '0': {
                        'json': {
                            'taskId': taskId
                        }
                    }
                })
            });
            const status = resp.status;
            const data = await resp.json();
            return { status, ok: resp.ok, data };
        }
        """

        for attempt in range(1, max_attempts + 1):
            logger.info(f"{CYAN}Waiting {poll_interval}s before retrieval check #{attempt}/{max_attempts}...{RESET}")
            await asyncio.sleep(poll_interval)

            res = await page.evaluate(js_code, task_id)
            if not res.get("ok"):
                logger.warning(f"Retrieval attempt #{attempt} HTTP error: {res.get('status')} - {res.get('data')}")
                continue

            data_list = res.get("data", [])
            if not data_list or not isinstance(data_list, list):
                continue

            json_data = data_list[0].get("result", {}).get("data", {}).get("json", {})
            output = json_data.get("output")
            finished = json_data.get("finished", False)
            status = json_data.get("bypassStatus", "unknown")

            logger.info(f"Retrieval #{attempt}: status='{status}', finished={finished}, output_length={len(output) if output else 0}")

            if finished and output:
                return json_data

        raise TimeoutError(f"Task {task_id} timed out after {max_attempts} attempts ({max_attempts * poll_interval}s).")

    async def humanize(self, text: str, model_type: str = "latest") -> Dict[str, Any]:
        """Runs single-pass humanization for the input text, then destroys browser data and restarts browser."""
        async with self._lock:
            start_time = time.time()
            self._request_count += 1
            req_idx = self._request_count

            word_count = len(text.split())
            preview = text[:80].replace("\n", " ") + ("..." if len(text) > 80 else "")
            logger.info(f"{BOLD}{MAGENTA}[Req #{req_idx}] Processing text ({len(text)} chars, {word_count} words) -> '{preview}'{RESET}")

            if self._page is None or self._page.is_closed():
                await self._open_fresh_session()

            # 1. Generate Task (with retry on fresh browser with new deviceId if any limit/glitch occurs)
            try:
                task_id = await self.generate_task(self._page, text, model_type=model_type)
            except Exception as e:
                logger.warning(f"Generate encountered error ({e}). Wiping data and retrying with fresh device session...")
                await self._open_fresh_session()
                task_id = await self.generate_task(self._page, text, model_type=model_type)

            logger.info(f"{GREEN}[Req #{req_idx}] Received Task ID: {task_id}{RESET}")

            # 2. Poll Retrieval until finished
            retrieval_data = await self.poll_retrieval(self._page, task_id, poll_interval=4.0, max_attempts=25)
            output_text = retrieval_data.get("output", "")

            elapsed = time.time() - start_time
            preview_out = (output_text[:80].replace("\n", " ") + "...") if output_text else "None"
            logger.info(f"{BOLD}{GREEN}✓ [Req #{req_idx}] Successfully humanized in {elapsed:.2f}s -> '{preview_out}'{RESET}")

            # 3. Destroy browser, completely wipe all profile data, and pre-warm next browser with fresh deviceId
            logger.info(f"{BOLD}{YELLOW}[RESTART PER REQUEST] Closing browser, completely destroying data, and opening fresh instance with new deviceId...{RESET}")
            await self._open_fresh_session()

            return {
                "task_id": task_id,
                "output": output_text,
                "raw": retrieval_data,
                "elapsed": elapsed
            }

    async def close(self):
        logger.info("Shutting down BrowserManager...")
        await self._destroy_current_session()
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass


browser_manager = BrowserManager(headless=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await browser_manager.initialize()
    yield
    await browser_manager.close()


app = FastAPI(
    title="Humanize.io Service",
    description="Local service for humanizing text via humanize.io session (Head Mode / Per-Request Dynamic Device Session)",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/humanize", response_model=HumanizeResponse)
@app.post("/process", response_model=HumanizeResponse)
@app.post("/", response_model=HumanizeResponse)
async def humanize_endpoint(req: HumanizeRequest):
    input_text = req.text or req.message
    if not input_text or not input_text.strip():
        logger.warning(f"{RED}Rejected request with empty text payload.{RESET}")
        raise HTTPException(
            status_code=400,
            detail="Missing text content. Provide 'text' or 'message' field in JSON payload."
        )

    words = input_text.strip().split()
    word_count = len(words)
    if word_count > MAX_WORDS:
        logger.warning(f"{RED}Rejected request: input has {word_count} words (max allowed: {MAX_WORDS}).{RESET}")
        raise HTTPException(
            status_code=400,
            detail=f"Input exceeds maximum limit of {MAX_WORDS} words (received {word_count} words). Please reduce text."
        )

    try:
        result = await browser_manager.humanize(
            text=input_text.strip(),
            model_type=req.modelType or "latest"
        )

        return HumanizeResponse(
            success=True,
            text=result["output"],
            humanized_text=result["output"],
            task_id=result.get("task_id"),
            raw_response=result.get("raw")
        )

    except Exception as e:
        logger.exception(f"Error processing humanization request: {e}")
        return HumanizeResponse(
            success=False,
            error=str(e)
        )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "humanizer2-humanize-io",
        "browser_mode": "head (not headless)" if not browser_manager.headless else "headless",
        "tab_status": "open" if browser_manager._page and not browser_manager._page.is_closed() else "closed",
        "device_id": browser_manager._current_browser_id,
        "policy": "restart browser, inject fresh deviceId, and completely destroy data after each request",
        "max_words": MAX_WORDS,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Start Humanize.io Service (Restarts browser & destroys all data per request)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to listen on (default: 8001)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run browser in headless mode (default: False, runs in visible head mode)"
    )

    args = parser.parse_args()

    browser_manager.headless = args.headless

    logger.info(f"{BOLD}{GREEN}Starting humanizer2 server for humanize.io:{RESET}")
    logger.info(f"  • Endpoint: http://{args.host}:{args.port}/humanize")
    logger.info(f"  • Mode: {'Headless Mode' if args.headless else 'Head Mode (Visible Browser)'}")
    logger.info(f"  • Word Limit: Max {MAX_WORDS} words")
    logger.info("  • Policy: Restart browser and completely destroy all profile data after each request with dynamic deviceId")

    uvicorn.run(app, host=args.host, port=args.port, reload=False)
