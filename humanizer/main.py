import asyncio
import json
import logging
import random
import string
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from playwright_stealth import Stealth
from pydantic import BaseModel, Field
import uvicorn

# Terminal ANSI Color Codes for high visibility logging
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BLUE = "\033[94m"
RED = "\033[91m"
MAGENTA = "\033[95m"

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

logger = logging.getLogger("humanizer")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter(datefmt="%H:%M:%S"))
logger.handlers = [console_handler]


# User's legit stealth credentials & cookies
INITIAL_COOKIES: List[Dict[str, Any]] = [
    {
        "name": "consent_jurisdiction",
        "value": "opt-out",
        "domain": ".humanizeai.pro",
        "path": "/",
    },
    {
        "name": "cf_clearance",
        "value": "WZmdMZKXnuRBUTzzDnjjDUPe6NACx8YzwCqdm1tLZbg-1788503880-1.2.1.1-BZys16fgNG4iGM99RuqpVSsynmp.WQFMyPGMO.HU5xDoW7VC1OoD6zDWbTNyGCXUyZfUMHMRz.Yog9bFZusBw2lI5K62rAEH.PT3Z4hV8fxUfpPcIR9v2RNK_3xT9eGR_0B_0J_8.ead9Zlhds6PvU0WObv_oXwQCF3xDvrfTOpfk1L_jr._jfkAkSl7hBYK.YywqWDlGc_S4Kw.I0m6zbr2LCiYLoC7hB_i97dGTzOR9dMyQ6v_eNoX0Aj0ujOJkcSZSg2INkBkWvEA9KfJm25x4mcUeiwcoe.ACwtrAdjGEt.0marGFMN_xlRl8t04qDo0F5IGE5.KQyvlXgxoOLX3nFnoPelhFMy65DlPazo",
        "domain": ".humanizeai.pro",
        "path": "/",
    },
    {
        "name": "__stripe_mid",
        "value": "eab13403-2bb6-4ff7-b4c4-5663975f1be648a7d8",
        "domain": ".humanizeai.pro",
        "path": "/",
    },
    {
        "name": "__stripe_sid",
        "value": "2a1e46a8-668f-4369-9497-2a19365c340517fa74",
        "domain": ".humanizeai.pro",
        "path": "/",
    },
    {
        "name": "g_state",
        "value": '{"i_l":1,"i_ll":1788503886642,"i_b":"sx0uk6/PkGqmEaQPae33sKMSs18I6ycdMZtt1FblMKM","i_e":{"enable_itp_optimization":24},"i_et":1788503886642}',
        "domain": ".humanizeai.pro",
        "path": "/",
    },
]

EXTRA_HEADERS = {
    "Accept-Language": "en-US,en;q=0.5",
    "DNT": "1",
    "Sec-GPC": "1",
}


def generate_session_id(length: int = 19) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


class HumanizeRequest(BaseModel):
    text: Optional[str] = Field(None, description="Text to humanize")
    message: Optional[str] = Field(None, description="Alias for text")
    style: Optional[str] = Field("standard", description="Rewriting style")
    alg: Optional[int] = Field(0, description="Algorithm mode")
    sessionId: Optional[str] = Field(None, description="Custom session ID")
    repeat: Optional[int] = Field(None, description="Number of times to sequentially repeat humanization (overrides server default)")
    repeats: Optional[int] = Field(None, description="Alias for repeat")


class HumanizeResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    humanized_text: Optional[str] = None
    passes_completed: Optional[int] = None
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BrowserManager:
    def __init__(self, page_limit: int = 20, default_repeat: int = 1):
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._stealth = Stealth()
        self._lock = asyncio.Lock()
        self._request_count = 0
        self._page_usage_count = 0
        self.page_limit = page_limit
        self.default_repeat = default_repeat

    async def _create_fresh_context_and_page(self):
        """Creates a fresh isolated browser context with stealth and initial cookies."""
        logger.info(f"{CYAN}Creating fresh browser context & page...{RESET}")
        self._context = await self._browser.new_context(
            viewport={"width": 1366, "height": 768},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            extra_http_headers=EXTRA_HEADERS,
        )

        try:
            await self._context.add_cookies(INITIAL_COOKIES)
            logger.info(f"{GREEN}Injected {len(INITIAL_COOKIES)} clean stealth session cookies.{RESET}")
        except Exception as e:
            logger.warning(f"Failed to inject cookies: {e}")

        self._page = await self._context.new_page()
        self._page_usage_count = 0
        await self._stealth.apply_stealth_async(self._page)

        logger.info("Navigating fresh tab to https://www.humanizeai.pro/ ...")
        await self._page.goto("https://www.humanizeai.pro/", wait_until="load", timeout=60000)
        await self._page.wait_for_timeout(2000)
        
        title = await self._page.title()
        logger.info(f"{BOLD}{GREEN}Fresh session ready: '{title}'{RESET}")

    async def initialize(self):
        logger.info(f"{BOLD}{GREEN}Launching Chromium in Head Mode (headless=False) with Stealth & Injected Cookies...{RESET}")
        self._playwright = await async_playwright().start()
        
        self._browser = await self._playwright.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-infobars",
                "--start-maximized",
            ]
        )
        
        await self._create_fresh_context_and_page()

    async def reopen_page(self):
        """Closes current page and reopens a fresh page after reaching page limit uses."""
        logger.info(f"{BOLD}{YELLOW}[PAGE REOPEN] Page reached {self.page_limit} uses limit. Closing page and reopening fresh tab...{RESET}")
        try:
            if self._page and not self._page.is_closed():
                await self._page.close()
        except Exception as e:
            logger.warning(f"Error closing page: {e}")

        if self._context is None:
            await self._create_fresh_context_and_page()
            return

        self._page = await self._context.new_page()
        self._page_usage_count = 0
        await self._stealth.apply_stealth_async(self._page)

        logger.info("Navigating reopened tab to https://www.humanizeai.pro/ ...")
        await self._page.goto("https://www.humanizeai.pro/", wait_until="load", timeout=60000)
        await self._page.wait_for_timeout(2000)
        
        title = await self._page.title()
        logger.info(f"{BOLD}{GREEN}Fresh reopened page ready: '{title}'{RESET}")

    async def ensure_page(self) -> Page:
        if self._page is None or self._page.is_closed():
            logger.warning("Tab was closed. Re-opening persistent tab...")
            if self._context is not None:
                await self.reopen_page()
            else:
                await self._create_fresh_context_and_page()
        return self._page

    async def is_captcha_active(self, page: Page) -> bool:
        try:
            return await page.evaluate(
                """() => {
                    const title = (document.title || "").toLowerCase();
                    if (title.includes("just a moment") || 
                        title.includes("cloudflare") || 
                        title.includes("attention required") || 
                        title.includes("verify you are human") ||
                        title.includes("security check")) {
                        return true;
                    }
                    const cf = document.getElementById("challenge-stage") || 
                               document.getElementById("cf-turnstile") ||
                               document.querySelector("iframe[src*='challenges.cloudflare.com']");
                    return cf !== null;
                }"""
            )
        except Exception:
            return False

    async def execute_in_page_fetch(self, page: Page, payload: Dict[str, Any]) -> Dict[str, Any]:
        js_script = """
        async (payload) => {
            try {
                const response = await fetch('https://www.humanizeai.pro/api/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });
                
                const status = response.status;
                const responseText = await response.text();
                let data = null;
                try {
                    data = JSON.parse(responseText);
                } catch (e) {
                    data = { raw: responseText };
                }
                
                return {
                    status: status,
                    ok: response.ok,
                    data: data
                };
            } catch (err) {
                return {
                    status: 500,
                    ok: false,
                    error: err.toString()
                };
            }
        }
        """
        return await page.evaluate(js_script, payload)

    async def send_single_pass(
        self,
        text: str,
        style: str = "standard",
        alg: int = 0,
        session_id: Optional[str] = None,
        pass_idx: int = 1,
        total_passes: int = 1
    ) -> Dict[str, Any]:
        start_time = time.time()
        page = await self.ensure_page()
        effective_session_id = session_id or generate_session_id()

        payload = {
            "test_allultra": None,
            "text": text,
            "trialNumber": 0,
            "alg": alg,
            "sessionId": effective_session_id,
            "keywords": [],
            "style": style,
            "isLogged": False,
            "ultra": False,
            "arm": "a",
            "isSample": False,
            "multilang": True
        }

        self._request_count += 1
        self._page_usage_count += 1
        current_count = self._request_count
        current_page_uses = self._page_usage_count
        
        pass_info = f" [Pass {pass_idx}/{total_passes}]" if total_passes > 1 else ""
        preview = text[:80].replace("\n", " ") + ("..." if len(text) > 80 else "")
        logger.info(f"{CYAN}[Req #{current_count}]{pass_info} Sending in-page request (Page use {current_page_uses}/{self.page_limit}, {len(text)} chars) -> '{preview}'{RESET}")
        
        res = await self.execute_in_page_fetch(page, payload)
        status_code = res.get("status", 500)

        # Check if captcha or Cloudflare challenge triggered
        if status_code in (403, 429) or await self.is_captcha_active(page):
            logger.warning(f"{BOLD}{YELLOW}" + "=" * 75 + f"{RESET}")
            logger.warning(f"{BOLD}{YELLOW}[ACTION REQUIRED] Captcha / Cloudflare challenge triggered!{RESET}")
            logger.warning(f"{BOLD}{YELLOW}Please look at the open Chromium window and solve the captcha.{RESET}")
            logger.warning(f"{BOLD}{YELLOW}The service is waiting and will resume automatically once solved...{RESET}")
            logger.warning(f"{BOLD}{YELLOW}" + "=" * 75 + f"{RESET}")

            try:
                await page.bring_to_front()
            except Exception:
                pass

            # Wait loop for user to solve
            retry_count = 0
            while retry_count < 120:
                await asyncio.sleep(2.5)
                retry_count += 1
                
                if not await self.is_captcha_active(page):
                    logger.info(f"{GREEN}Challenge solved! Retrying API fetch (attempt {retry_count})...{RESET}")
                    res = await self.execute_in_page_fetch(page, payload)
                    if res.get("ok") and res.get("status") == 200:
                        break
                else:
                    if retry_count % 5 == 0:
                        logger.warning(f"{YELLOW}Waiting for captcha solution in browser (waiting {retry_count * 2.5:.0f}s)...{RESET}")

        elapsed = time.time() - start_time
        if res.get("ok") and res.get("status") == 200:
            logger.info(f"{BOLD}{GREEN}✓ [Req #{current_count}]{pass_info} Received 200 OK from humanizeai.pro in {elapsed:.2f}s{RESET}")
        else:
            logger.error(f"{BOLD}{RED}✗ [Req #{current_count}]{pass_info} Failed with status {res.get('status')} in {elapsed:.2f}s: {res.get('error') or 'Blocked/Error'}{RESET}")

        # When same page has reached page_limit uses, close page and reopen
        if self._page_usage_count >= self.page_limit:
            await self.reopen_page()

        return res

    async def process_message(
        self,
        text: str,
        style: str = "standard",
        alg: int = 0,
        session_id: Optional[str] = None,
        repeats: int = 1
    ) -> Dict[str, Any]:
        async with self._lock:
            total_passes = max(1, repeats)
            current_text = text
            last_res: Dict[str, Any] = {}
            pass_history = []

            if total_passes > 1:
                logger.info(f"{BOLD}{MAGENTA}" + "=" * 60 + f"{RESET}")
                logger.info(f"{BOLD}{MAGENTA}[MULTI-PASS START] Beginning {total_passes} sequential iteration(s){RESET}")
                logger.info(f"{BOLD}{MAGENTA}" + "=" * 60 + f"{RESET}")

            for p in range(1, total_passes + 1):
                res = await self.send_single_pass(
                    text=current_text,
                    style=style,
                    alg=alg,
                    session_id=session_id if p == 1 else None,
                    pass_idx=p,
                    total_passes=total_passes
                )
                last_res = res
                status_code = res.get("status", 500)
                if status_code != 200 or not res.get("ok"):
                    logger.warning(f"{RED}Pass {p}/{total_passes} failed. Aborting further repeat passes.{RESET}")
                    break

                # Extract humanized text for next pass
                data = res.get("data")
                extracted = None
                if isinstance(data, dict):
                    results = data.get("result")
                    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
                        extracted = results[0].get("text")
                    elif "text" in data:
                        extracted = data.get("text")

                if extracted and extracted.strip():
                    pass_history.append(extracted)
                    current_text = extracted.strip()
                    if p < total_passes:
                        next_preview = (current_text[:60].replace("\n", " ") + "...") if len(current_text) > 60 else current_text
                        logger.info(f"{BOLD}{CYAN}--> Output from Pass {p} will be the input for Pass {p+1}: '{next_preview}'{RESET}")
                        await asyncio.sleep(0.5)
                else:
                    logger.warning(f"{YELLOW}Pass {p}/{total_passes} returned empty text. Keeping previous text.{RESET}")

            if total_passes > 1:
                logger.info(f"{BOLD}{MAGENTA}" + "=" * 60 + f"{RESET}")
                logger.info(f"{BOLD}{GREEN}[MULTI-PASS COMPLETE] Finished {len(pass_history)}/{total_passes} iteration(s){RESET}")
                logger.info(f"{BOLD}{MAGENTA}" + "=" * 60 + f"{RESET}")

            return {
                "last_res": last_res,
                "final_text": current_text,
                "passes_completed": len(pass_history) if pass_history else (1 if last_res.get("ok") else 0),
                "history": pass_history
            }

    async def close(self):
        logger.info("Shutting down Chromium browser...")
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            logger.warning(f"Error during browser close: {e}")


browser_manager = BrowserManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await browser_manager.initialize()
    yield
    await browser_manager.close()


app = FastAPI(
    title="HumanizeAI Playwright Service",
    description="Continuously listening Head-Mode Playwright API scraper with stealth & captcha handling",
    version="1.2.0",
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

    logger.info(f"{BOLD}{MAGENTA}--> Incoming POST request on route{RESET}")

    try:
        repeat_count = (
            req.repeats
            if req.repeats is not None
            else (req.repeat if req.repeat is not None else browser_manager.default_repeat)
        )
        if repeat_count < 1:
            repeat_count = 1

        result_dict = await browser_manager.process_message(
            text=input_text,
            style=req.style or "standard",
            alg=req.alg if req.alg is not None else 0,
            session_id=req.sessionId,
            repeats=repeat_count
        )

        last_res = result_dict.get("last_res", {})
        status_code = last_res.get("status", 500)
        data = last_res.get("data")
        final_text = result_dict.get("final_text")
        passes_completed = result_dict.get("passes_completed", 0)

        if passes_completed == 0 or (status_code != 200 and not last_res.get("ok")):
            error_msg = last_res.get("error") or f"Upstream returned HTTP status {status_code}"
            return HumanizeResponse(
                success=False,
                error=error_msg,
                raw_response=data,
                passes_completed=passes_completed
            )

        preview_out = (final_text[:80].replace("\n", " ") + "...") if final_text else "None"
        logger.info(f"{BOLD}{GREEN}<-- Returning response after {passes_completed} pass(es): '{preview_out}'{RESET}")

        return HumanizeResponse(
            success=True,
            text=final_text,
            humanized_text=final_text,
            passes_completed=passes_completed,
            raw_response=data
        )

    except Exception as e:
        logger.exception(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "humanizer-playwright-head",
        "tab_status": "open" if browser_manager._page and not browser_manager._page.is_closed() else "closed",
        "page_usage_count": browser_manager._page_usage_count,
        "page_limit": browser_manager.page_limit,
        "default_repeat": browser_manager.default_repeat,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Start HumanizeAI Playwright Service in Head Mode"
    )
    parser.add_argument(
        "-r", "--repeat", "--repeats",
        type=int,
        default=1,
        help="Number of times to sequentially repeat humanization for each incoming request (default: 1)"
    )
    parser.add_argument(
        "--page-limit", "--page-iterations", "--page-max-uses",
        type=int,
        default=20,
        dest="page_limit",
        help="Number of iterations/uses before closing and reopening the page (default: 20)"
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
        default=8000,
        help="Port to listen on (default: 8000)"
    )

    args = parser.parse_args()

    browser_manager.page_limit = max(1, args.page_limit)
    browser_manager.default_repeat = max(1, args.repeat)

    logger.info(f"{BOLD}{GREEN}Starting server with backend configuration:{RESET}")
    logger.info(f"  • Sequential repeat count per request: {browser_manager.default_repeat}")
    logger.info(f"  • Page reopen limit: {browser_manager.page_limit} iterations")
    logger.info(f"  • Endpoint: http://{args.host}:{args.port}/humanize")

    uvicorn.run(app, host=args.host, port=args.port, reload=False)



