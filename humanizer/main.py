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


class HumanizeResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    humanized_text: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BrowserManager:
    def __init__(self):
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._stealth = Stealth()
        self._lock = asyncio.Lock()
        self._request_count = 0

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

    async def reset_session(self):
        """Completely destroys current context/cookies/storage and initializes a fresh session."""
        logger.info(f"{BOLD}{YELLOW}[SESSION RESET] 2 requests processed. Wiping cookies, storage & creating brand new session...{RESET}")
        try:
            if self._page and not self._page.is_closed():
                await self._page.close()
            if self._context:
                await self._context.close()
        except Exception as e:
            logger.warning(f"Error during context cleanup: {e}")

        await self._create_fresh_context_and_page()
        logger.info(f"{BOLD}{GREEN}[SESSION RESET COMPLETE] Brand new clean browser session initialized.{RESET}")

    async def ensure_page(self) -> Page:
        if self._page is None or self._page.is_closed():
            logger.warning("Tab was closed. Re-opening persistent tab...")
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

    async def process_message(
        self,
        text: str,
        style: str = "standard",
        alg: int = 0,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        async with self._lock:
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
            current_count = self._request_count
            preview = text[:80].replace("\n", " ") + ("..." if len(text) > 80 else "")
            logger.info(f"{CYAN}[Req #{current_count}] Sending in-page request ({len(text)} chars) -> '{preview}'{RESET}")
            
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
                logger.info(f"{BOLD}{GREEN}✓ [Req #{current_count}] Received 200 OK from humanizeai.pro in {elapsed:.2f}s{RESET}")
            else:
                logger.error(f"{BOLD}{RED}✗ [Req #{current_count}] Failed with status {res.get('status')} in {elapsed:.2f}s: {res.get('error') or 'Blocked/Error'}{RESET}")

            # Every 2 requests, reset context, clear cookies and initialize a fully fresh session
            if current_count % 2 == 0:
                await self.reset_session()

            return res

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
        res = await browser_manager.process_message(
            text=input_text,
            style=req.style or "standard",
            alg=req.alg if req.alg is not None else 0,
            session_id=req.sessionId
        )

        status_code = res.get("status", 500)
        data = res.get("data")

        if status_code != 200 or not res.get("ok"):
            error_msg = res.get("error") or f"Upstream returned HTTP status {status_code}"
            return HumanizeResponse(
                success=False,
                error=error_msg,
                raw_response=data
            )

        # Extract humanized text
        extracted_text = None
        if isinstance(data, dict):
            results = data.get("result")
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
                extracted_text = results[0].get("text")
            elif "text" in data:
                extracted_text = data.get("text")

        preview_out = (extracted_text[:80].replace("\n", " ") + "...") if extracted_text else "None"
        logger.info(f"{BOLD}{GREEN}<-- Returning response: '{preview_out}'{RESET}")

        return HumanizeResponse(
            success=True,
            text=extracted_text,
            humanized_text=extracted_text,
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
        "tab_status": "open" if browser_manager._page and not browser_manager._page.is_closed() else "closed"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)



