# api.py (v1.0) - Patient-Flow Scanner w/ AI scoring (OpenAI Responses API)
# Run:
#   python -m pip install -U fastapi uvicorn beautifulsoup4 playwright openai
#   python -m playwright install chromium
#
# PowerShell (IMPORTANT: run each line ONCE, do NOT paste your key by itself on the next line):
#   $env:OPENAI_API_KEY="sk-...your key..."
#   $env:OPENAI_MODEL="gpt-5"     # or "gpt-5.2" if your account has it
#   $env:SCORING_MODE="ai"        # or "rules"
#
# Optional branding:
#   $env:APP_NAME="Keyturn Studio"
#   $env:APP_PRODUCT="Patient-Flow Quick Scan"
#   $env:PRIMARY_CTA_TEXT="Get the Blueprint"
#   $env:PRIMARY_CTA_URL="https://www.keyturn.studio/quote.html"
#
# Start server:
#   python -m uvicorn api:app --reload

import base64
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl, conint
from playwright.sync_api import sync_playwright

# Import OpenAI safely so missing module doesn't crash the whole server at import-time.
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parent

DB_PATH = "patientflow.sqlite"
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

APP_NAME = os.getenv("APP_NAME", "Keyturn Studio").strip() or "Keyturn Studio"
APP_PRODUCT = os.getenv("APP_PRODUCT", "Patient-Flow Quick Scan").strip() or "Patient-Flow Quick Scan"
PRIMARY_CTA_TEXT = os.getenv("PRIMARY_CTA_TEXT", "Get the Blueprint").strip() or "Get the Blueprint"
PRIMARY_CTA_URL = os.getenv("PRIMARY_CTA_URL", "https://www.keyturn.studio/quote.html").strip() or "https://www.keyturn.studio/quote.html"

# Defaults
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-5")
OPENAI_MODEL_FALLBACKS = [
    os.getenv("OPENAI_MODEL", OPENAI_MODEL_DEFAULT),
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
]

# If you want to disable AI quickly:
#   $env:SCORING_MODE="rules"
SCORING_MODE = os.getenv("SCORING_MODE", "ai").lower().strip()  # ai | rules

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

# Constants
MAX_PUBLIC_ID_ATTEMPTS = 10

RUBRIC_TEXT = """Clinic Patient-Flow Score – Rubric v0.2
Categories (0–10 each, total 60)
Final Patient-Flow Score = Total / 6 (0–10 scale)

1. Clarity & first impression (0–10)
0–3: Confusing, vague headline. I can’t tell what you do or for whom.
4–7: I can figure it out, but it’s wordy or not patient-friendly.
8–10: Instantly clear: what you do, for whom, and what to do next.

2. Booking path (0–10)
0–3: No obvious “Book / Request consult”. Hidden in nav or buried.
4–7: Booking exists but takes multiple clicks, or is a bit clunky.
8–10: Single, obvious path to book. Few clicks. Very simple.

3. Mobile experience (0–10)
0–3: Feels broken: tiny text, overlapping bits, need to zoom.
4–7: Usable but cramped, some awkward sections.
8–10: Clean, easy to scroll and tap, feels designed for phone.

4. Trust & proof (0–10)
0–3: Almost no social proof; feels sketchy or anonymous.
4–7: Some testimonials / credentials, but scattered or weak.
8–10: Strong doctor presence, credentials, reviews, and before/afters.

5. Treatments & offer (0–10)
0–3: Can’t really tell what they specialize in or what’s important.
4–7: List exists but is bland / technical; not framed as outcomes.
8–10: Clear high-value treatments, in simple language, with outcomes & maybe price hints.

6. Tech basics (0–10)
0–3: Very slow, broken layout, “coming soon” pages.
4–7: Works fine but looks dated / slightly janky.
8–10: Loads fast, feels modern, no obvious glitches.
"""

WORKED_EXAMPLES = """
Calibration examples (use these to avoid score inflation):

GOLD example
- URL: https://www.cpsdocs.com/
- Human score: total 54/60 => 9.0/10
- Notes: fast, clean, modern; strong proof (doctors + reviews + video testimonials + before/after); great mobile;
         booking path clear, but booking form has too many fields.

WEAK example
- URL: http://www.vitasurgical.com/
- Human score: total 15/60 => 2.5/10
- Notes: very ugly/outdated; weak design; thin homepage; poor digital storefront vibe.
"""

app = FastAPI()

# Serve screenshots + artifacts at /artifacts/<scan_id>/<file>
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")


# Favicon routes
@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico():
    return FileResponse(BASE_DIR / "favicon.ico", media_type="image/x-icon")


@app.get("/favicon.svg", include_in_schema=False)
def favicon_svg():
    return FileResponse(BASE_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/favicon-16x16.png", include_in_schema=False)
def favicon_16():
    return FileResponse(BASE_DIR / "favicon-16x16.png", media_type="image/png")


@app.get("/favicon-32x32.png", include_in_schema=False)
def favicon_32():
    return FileResponse(BASE_DIR / "favicon-32x32.png", media_type="image/png")


@app.get("/android-chrome-192x192.png", include_in_schema=False)
def android_chrome_192():
    return FileResponse(BASE_DIR / "android-chrome-192x192.png", media_type="image/png")


@app.get("/android-chrome-512x512.png", include_in_schema=False)
def android_chrome_512():
    return FileResponse(BASE_DIR / "android-chrome-512x512.png", media_type="image/png")


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_columns() -> None:
    conn = db()
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(scans)").fetchall()}
        if "email" not in cols:
            conn.execute("ALTER TABLE scans ADD COLUMN email TEXT")
        if "public_id" not in cols:
            conn.execute("ALTER TABLE scans ADD COLUMN public_id TEXT")
        if "slug" not in cols:
            conn.execute("ALTER TABLE scans ADD COLUMN slug TEXT")
        conn.commit()
        
        # Create UNIQUE index on public_id if it doesn't exist
        try:
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_public_id ON scans(public_id)")
            conn.commit()
        except (sqlite3.IntegrityError, sqlite3.OperationalError):
            # Index might already exist or fail for expected reasons
            pass
    finally:
        conn.close()


def init_db() -> None:
    conn = db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
          id TEXT PRIMARY KEY,
          url TEXT NOT NULL,
          status TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          email TEXT,
          public_id TEXT,
          slug TEXT,
          evidence_json TEXT,
          score_json TEXT,
          error TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          event_type TEXT NOT NULL,
          scan_id TEXT,
          public_id TEXT,
          metadata TEXT,
          created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()
    _ensure_columns()


init_db()


class ScanRequest(BaseModel):
    url: HttpUrl
    email: Optional[str] = None


class EventRequest(BaseModel):
    event_type: str
    scan_id: Optional[str] = None
    public_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EmailReportRequest(BaseModel):
    email: str
    report_id: str


# ---- Structured output model (Option A) ----
class PatientFlowScores(BaseModel):
    clarity_first_impression: conint(ge=0, le=10)
    booking_path: conint(ge=0, le=10)
    mobile_experience: conint(ge=0, le=10)
    trust_and_proof: conint(ge=0, le=10)
    treatments_and_offer: conint(ge=0, le=10)
    tech_basics: conint(ge=0, le=10)


class QuickWinItem(BaseModel):
    action: str = Field(..., description="The action text describing what to do")
    impact: Literal["High", "Med", "Low"] = Field(..., description="Impact level: High, Med, or Low")
    effort: Literal["Low", "Med", "High"] = Field(..., description="Effort level: Low, Med, or High")


class PatientFlowAIOutput(BaseModel):
    clinic_name: str = Field(..., description="Clinic name inferred from the site")
    scores: PatientFlowScores
    strengths: List[str] = Field(default_factory=list)
    leaks: List[str] = Field(default_factory=list)
    quick_wins: List[QuickWinItem] = Field(default_factory=list)


def _model_to_dict(m: Any) -> Dict[str, Any]:
    if hasattr(m, "model_dump"):
        return m.model_dump()  # pydantic v2
    if hasattr(m, "dict"):
        return m.dict()  # pydantic v1
    return dict(m)


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def slugify(name: str) -> str:
    """Convert a name into a URL-friendly slug.
    
    Lowercases, removes non-alphanumeric characters, collapses whitespace/dashes, and trims.
    Example: "Aurora Aesthetic Clinic" -> "aurora-aesthetic-clinic"
    """
    if not name:
        return ""
    
    # Lowercase
    s = name.lower()
    
    # Replace spaces and underscores with dashes
    s = re.sub(r'[\s_]+', '-', s)
    
    # Remove non-alphanumeric characters except dashes
    s = re.sub(r'[^a-z0-9-]', '', s)
    
    # Collapse multiple dashes into one
    s = re.sub(r'-+', '-', s)
    
    # Trim dashes from start and end
    s = s.strip('-')
    
    return s


def _ai_ready() -> (bool, str):
    if SCORING_MODE != "ai":
        return True, ""
    if OpenAI is None:
        return False, "AI mode needs the 'openai' package. Run: python -m pip install -U openai"
    if not os.getenv("OPENAI_API_KEY"):
        return False, "AI mode needs OPENAI_API_KEY. In PowerShell: $env:OPENAI_API_KEY=\"sk-...\""
    return True, ""


# Simple signals
CTA_RE = re.compile(r"\b(book|booking|schedule|consult|appointment|request|call)\b", re.I)
MD_RE = re.compile(r"\b(m\.?d\.?|d\.?o\.?)\b", re.I)
DR_RE = re.compile(r"\bdr\.?\b", re.I)
BEFORE_AFTER_RE = re.compile(r"\bbefore\s*(?:and|&|/)\s*after\b", re.I)
GOOGLE_REVIEWS_RE = re.compile(r"\bgoogle\b.*\breview", re.I)

OUTCOME_WORDS = [
    "results",
    "before",
    "after",
    "younger",
    "tighten",
    "lift",
    "reduce",
    "clear",
    "improve",
    "refresh",
    "natural",
    "confidence",
    "glow",
    "brighter",
]

MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _img_to_data_url(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None
    if path_str.startswith("/artifacts/"):
        rel = path_str.replace("/artifacts/", "", 1)
        p = ARTIFACTS_DIR / rel
    else:
        p = Path(path_str)

    if not p.exists():
        return None

    mime = MIME_BY_EXT.get(p.suffix.lower(), "image/jpeg")
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def band_from_score(score10: float) -> str:
    if score10 >= 8.0:
        return "Strong, but still leaking bookings"
    if score10 >= 6.0:
        return "Decent, but not patient-flow optimized"
    if score10 >= 4.0:
        return "Weak digital storefront"
    return "Risky / probably losing bookings"


def infer_clinic_name(home_title: str, home_h1: str, url: str) -> str:
    def first_part(s: str) -> str:
        for sep in ["|", " - ", " — ", " – "]:
            if sep in s:
                return s.split(sep)[0].strip()
        return s.strip()

    t = first_part(home_title or "")
    if 3 <= len(t) <= 80:
        return t
    h = (home_h1 or "").strip()
    if 3 <= len(h) <= 80:
        return h
    return urlparse(url).netloc.replace("www.", "")


def extract_evidence_from_html(url: str, html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    h1 = (soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else "")[:200]
    title = (soup.title.get_text(" ", strip=True) if soup.title else "")[:200]

    meta_viewport = bool(soup.find("meta", attrs={"name": "viewport"}))

    forms = soup.find_all("form")
    form_count = len(forms)
    input_count = len(soup.find_all("input")) + len(soup.find_all("select")) + len(soup.find_all("textarea"))

    script_srcs: List[str] = []
    for s in soup.find_all("script"):
        src = (s.get("src") or "").strip()
        if src:
            script_srcs.append(src[:260])
    script_srcs = script_srcs[:80]

    links: List[Dict[str, Any]] = []
    for a in soup.find_all("a"):
        txt = a.get_text(" ", strip=True)[:120]
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        abs_url = urljoin(url, href)
        try:
            is_internal = urlparse(abs_url).netloc == urlparse(url).netloc
        except Exception:
            is_internal = False
        links.append({"text": txt, "href": href[:240], "abs_url": abs_url[:360], "is_internal": is_internal})
    links = links[:350]

    ctas = [l for l in links if l["text"] and CTA_RE.search(l["text"])]
    ctas = ctas[:40]

    tel_links = [l["href"] for l in links if l["href"].lower().startswith("tel:")][:10]
    mail_links = [l["href"] for l in links if l["href"].lower().startswith("mailto:")][:10]

    raw_text = soup.get_text(" ", strip=True)
    page_text = raw_text.lower()

    words = re.findall(r"[A-Za-z0-9]+", raw_text)
    word_count = len(words)
    img_count = len(soup.find_all("img"))
    internal_link_count = sum(1 for l in links if l.get("is_internal"))

    proof_hits = {
        "before_after": bool(BEFORE_AFTER_RE.search(page_text)),
        "testimonials": ("testimonial" in page_text) or ("patient review" in page_text) or ("what patients say" in page_text),
        "reviews": bool(GOOGLE_REVIEWS_RE.search(page_text)) or ("google reviews" in page_text),
        "credentials": ("board certified" in page_text) or bool(MD_RE.search(page_text)) or bool(DR_RE.search(page_text)),
    }

    outcome_hits = sum(1 for w in OUTCOME_WORDS if w in page_text)
    coming_soon = ("coming soon" in page_text) or ("under construction" in page_text)

    font_count = len(soup.find_all("font"))
    center_count = len(soup.find_all("center"))
    marquee_count = len(soup.find_all("marquee"))
    table_count = len(soup.find_all("table"))
    has_frames = bool(soup.find("frameset") or soup.find("frame"))

    style_text = " ".join((s.get_text(" ", strip=True) for s in soup.find_all("style")))
    has_media_queries = "@media" in style_text.lower()

    fixed_nums: List[int] = []
    for n in re.findall(r"width\s*:\s*(\d{3,4})px", html, re.I):
        fixed_nums.append(int(n))
    for n in re.findall(r'width\s*=\s*["\']?(\d{3,4})["\']?', html, re.I):
        fixed_nums.append(int(n))
    fixed_width_hits = sum(1 for n in fixed_nums if n >= 900)

    text_sample = _clean_text(raw_text)[:2200]
    nav_link_texts = sorted({(l.get("text") or "").strip() for l in links if l.get("is_internal") and l.get("text")})[:160]

    return {
        "page_url": url,
        "title": title,
        "h1": h1,
        "meta_viewport": meta_viewport,
        "form_count": form_count,
        "input_count": input_count,
        "cta_links": [{"text": c["text"], "abs_url": c["abs_url"]} for c in ctas],
        "tel_links": tel_links,
        "mailto_links": mail_links,
        "proof_hits": proof_hits,
        "word_count": word_count,
        "img_count": img_count,
        "internal_link_count": internal_link_count,
        "script_srcs": script_srcs,
        "outcome_hits": outcome_hits,
        "coming_soon": coming_soon,
        "font_count": font_count,
        "center_count": center_count,
        "marquee_count": marquee_count,
        "table_count": table_count,
        "has_frames": has_frames,
        "has_media_queries": has_media_queries,
        "fixed_width_hits": fixed_width_hits,
        "text_sample": text_sample,
        "nav_link_texts": nav_link_texts,
    }


def _detect_mobile_overflow(page) -> Dict[str, Any]:
    try:
        overflow = page.evaluate(
            """
            () => {
              const de = document.documentElement;
              const b = document.body;
              const sw = Math.max(de ? de.scrollWidth : 0, b ? b.scrollWidth : 0);
              const cw = de ? de.clientWidth : 0;
              return { scrollWidth: sw, clientWidth: cw, overflow: sw > cw + 2 };
            }
            """
        )
        return overflow if isinstance(overflow, dict) else {}
    except Exception:
        return {}


def _to_artifact_url(path: Path) -> str:
    try:
        rel = path.relative_to(ARTIFACTS_DIR).as_posix()
        return f"/artifacts/{rel}"
    except Exception:
        return "/artifacts/" + path.as_posix().replace("\\", "/").split("artifacts/", 1)[-1]


def _wait_network_idle(page, timeout_ms: int = 20000) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except Exception:
        pass


def _eagerize_images(page) -> None:
    try:
        page.evaluate(
            """
            () => {
              document.querySelectorAll('img[loading="lazy"]').forEach(img => {
                img.setAttribute('loading', 'eager');
              });
            }
            """
        )
    except Exception:
        pass


def _get_scroll_positions(page) -> Dict[str, int]:
    try:
        meta = page.evaluate(
            """
            () => {
              const de = document.documentElement;
              const b = document.body;
              const sh = Math.max(de ? de.scrollHeight : 0, b ? b.scrollHeight : 0);
              const ih = window.innerHeight || 0;
              const maxY = Math.max(0, sh - ih);
              return { sh, ih, maxY };
            }
            """
        )
        max_y = int((meta or {}).get("maxY") or 0)
    except Exception:
        max_y = 0

    mid = int(max_y * 0.45)
    bot = int(max_y * 0.90)
    return {"top": 0, "mid": mid, "bottom": bot}


def fetch_page_evidence(
    url: str,
    mobile: bool,
    screenshot_paths: List[Path],
) -> Dict[str, Any]:
    t0 = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-dev-shm-usage", "--no-sandbox"],
        )
        viewport = {"width": 390, "height": 844} if mobile else {"width": 1365, "height": 768}

        context = browser.new_context(
            viewport=viewport,
            user_agent=(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
                if mobile
                else "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
            ),
            is_mobile=mobile,
            has_touch=mobile,
            device_scale_factor=2 if mobile else 1,
        )
        page = context.new_page()
        page.set_default_timeout(25000)

        saved_urls: List[str] = []
        screenshot_manifest: List[Dict[str, str]] = []
        screenshot_errors: List[Dict[str, str]] = []

        def try_shot(key: str, sp: Path, scroll_y: Optional[int] = None) -> None:
            try:
                sp.parent.mkdir(parents=True, exist_ok=True)

                if scroll_y is not None:
                    page.evaluate(f"window.scrollTo(0, {int(scroll_y)})")
                    page.wait_for_timeout(450)
                    _wait_network_idle(page, 9000)

                page.screenshot(path=str(sp), full_page=False, type="jpeg", quality=72)

                # Retry if capture looks suspiciously empty (often white)
                if sp.exists() and sp.stat().st_size < 35_000:
                    page.wait_for_timeout(1200)
                    _wait_network_idle(page, 9000)
                    page.screenshot(path=str(sp), full_page=False, type="jpeg", quality=72)

                if sp.exists() and sp.stat().st_size > 12_000:
                    u = _to_artifact_url(sp)
                    saved_urls.append(u)
                    screenshot_manifest.append({"key": key, "url": u})
                else:
                    screenshot_errors.append({"key": key, "reason": "screenshot file missing/too small"})
            except Exception as e:
                screenshot_errors.append({"key": key, "reason": str(e)})

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=90000)
            _wait_network_idle(page, 20000)
            page.wait_for_timeout(700)
            _eagerize_images(page)

            if screenshot_paths:
                for sp in screenshot_paths:
                    sp.parent.mkdir(parents=True, exist_ok=True)

                if not mobile:
                    try_shot("home_desktop_top", screenshot_paths[0], scroll_y=0)
                else:
                    pos = _get_scroll_positions(page)
                    if len(screenshot_paths) > 0:
                        try_shot("home_mobile_top", screenshot_paths[0], scroll_y=pos["top"])
                    if len(screenshot_paths) > 1:
                        try_shot("home_mobile_mid", screenshot_paths[1], scroll_y=pos["mid"])
                    if len(screenshot_paths) > 2:
                        try_shot("home_mobile_bottom", screenshot_paths[2], scroll_y=pos["bottom"])

            html = page.content()
            evidence = extract_evidence_from_html(page.url, html)
            evidence["load_seconds"] = round(time.time() - t0, 2)
            evidence["mobile_viewport"] = mobile
            evidence["final_url"] = page.url
            evidence["screenshot_urls"] = saved_urls
            evidence["screenshot_manifest"] = screenshot_manifest
            evidence["screenshot_errors"] = screenshot_errors
            if mobile:
                evidence["mobile_overflow"] = _detect_mobile_overflow(page)

            return evidence
        finally:
            context.close()
            browser.close()


def score_rules_only(evidence: Dict[str, Any]) -> Dict[str, Any]:
    home_d = evidence.get("home_desktop", {})
    home_m = evidence.get("home_mobile", {})

    h1 = (home_d.get("h1") or "").strip()
    ctas = home_d.get("cta_links") or []
    has_booking_cta = any(re.search(r"\b(book|schedule|appointment|consult)\b", (c.get("text") or ""), re.I) for c in ctas)

    if not h1:
        clarity = 3
    elif len(h1) <= 90:
        clarity = 8
    else:
        clarity = 6

    booking = 8 if has_booking_cta else 4

    mv = bool(home_m.get("meta_viewport"))
    overflow = bool((home_m.get("mobile_overflow") or {}).get("overflow"))
    if (not mv) or overflow:
        mobile_score = 4
    else:
        mobile_score = 7

    proof_hits = (home_d.get("proof_hits") or {})
    proof_signal = sum(1 for v in proof_hits.values() if v)
    img_count = int(home_d.get("img_count") or 0)
    if proof_signal >= 3 and img_count >= 8:
        trust = 9
    elif proof_signal >= 2:
        trust = 7
    else:
        trust = 5

    nav = " ".join(home_d.get("nav_link_texts") or []).lower()
    txt = (home_d.get("text_sample") or "").lower()
    treatment_signals = 0
    if any(k in nav for k in ["treat", "service", "procedure", "laser", "inject", "surgery", "derm", "skin"]):
        treatment_signals += 1
    if any(k in txt for k in ["treatments", "services", "procedures"]):
        treatment_signals += 1
    treatments = 8 if treatment_signals >= 2 else 6 if treatment_signals == 1 else 5

    load_sec = float(home_d.get("load_seconds") or 0)
    legacy = int(home_d.get("font_count") or 0) + int(home_d.get("center_count") or 0) + int(home_d.get("marquee_count") or 0)
    if legacy >= 5 or bool(home_d.get("has_frames")):
        tech = 2
    elif load_sec > 15:
        tech = 3
    elif load_sec > 8:
        tech = 5
    else:
        tech = 8

    scores = {
        "clarity_first_impression": int(clarity),
        "booking_path": int(booking),
        "mobile_experience": int(mobile_score),
        "trust_and_proof": int(trust),
        "treatments_and_offer": int(treatments),
        "tech_basics": int(tech),
    }
    total = sum(scores.values())
    score10 = round(total / 6.0, 1)

    return {
        "clinic_name": infer_clinic_name(home_d.get("title", ""), home_d.get("h1", ""), evidence.get("target_url", "")),
        "url": evidence.get("target_url", ""),
        "scores": scores,
        "total_score_60": total,
        "patient_flow_score_10": score10,
        "band": band_from_score(score10),
        "strengths": [],
        "leaks": [],
        "quick_wins": [],
        "debug": {"mode": "rules"},
    }


def _clamp_int(n: Any, lo: int = 0, hi: int = 10) -> int:
    try:
        v = int(n)
    except Exception:
        v = 0
    return max(lo, min(hi, v))


def _post_guardrails(output: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    home_d = evidence.get("home_desktop", {})
    home_m = evidence.get("home_mobile", {})
    scores = output.get("scores", {}) or {}

    legacy = (
        int(home_d.get("font_count") or 0)
        + int(home_d.get("center_count") or 0)
        + int(home_d.get("marquee_count") or 0)
        + int(home_d.get("table_count") or 0)
    )
    has_frames = bool(home_d.get("has_frames"))
    fixed_width_hits = int(home_d.get("fixed_width_hits") or 0)

    mv = bool(home_m.get("meta_viewport"))
    overflow = bool((home_m.get("mobile_overflow") or {}).get("overflow"))
    if (not mv) or overflow or fixed_width_hits >= 3:
        scores["mobile_experience"] = min(_clamp_int(scores.get("mobile_experience")), 4)

    if has_frames or legacy >= 10:
        scores["tech_basics"] = min(_clamp_int(scores.get("tech_basics")), 2)
    elif legacy >= 5:
        scores["tech_basics"] = min(_clamp_int(scores.get("tech_basics")), 4)

    fixed_scores = {
        "clarity_first_impression": _clamp_int(scores.get("clarity_first_impression")),
        "booking_path": _clamp_int(scores.get("booking_path")),
        "mobile_experience": _clamp_int(scores.get("mobile_experience")),
        "trust_and_proof": _clamp_int(scores.get("trust_and_proof")),
        "treatments_and_offer": _clamp_int(scores.get("treatments_and_offer")),
        "tech_basics": _clamp_int(scores.get("tech_basics")),
    }
    total = sum(fixed_scores.values())
    score10 = round(total / 6.0, 1)

    output["scores"] = fixed_scores
    output["total_score_60"] = total
    output["patient_flow_score_10"] = score10
    output["band"] = band_from_score(score10)
    return output


def ai_score_patient_flow(target_url: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    ok, msg = _ai_ready()
    if not ok:
        raise RuntimeError(msg)

    client = OpenAI()  # type: ignore

    home_d = evidence.get("home_desktop", {})
    home_m = evidence.get("home_mobile", {})

    evidence_payload = {
        "target_url": target_url,
        "home_desktop": {
            k: home_d.get(k)
            for k in [
                "final_url",
                "title",
                "h1",
                "cta_links",
                "tel_links",
                "mailto_links",
                "word_count",
                "img_count",
                "internal_link_count",
                "proof_hits",
                "outcome_hits",
                "coming_soon",
                "form_count",
                "input_count",
                "font_count",
                "center_count",
                "marquee_count",
                "table_count",
                "has_frames",
                "has_media_queries",
                "fixed_width_hits",
                "load_seconds",
                "text_sample",
                "nav_link_texts",
                "screenshot_urls",
            ]
        },
        "home_mobile": {
            k: home_m.get(k)
            for k in [
                "final_url",
                "title",
                "h1",
                "meta_viewport",
                "mobile_overflow",
                "word_count",
                "img_count",
                "proof_hits",
                "form_count",
                "input_count",
                "has_media_queries",
                "fixed_width_hits",
                "load_seconds",
                "text_sample",
                "screenshot_urls",
            ]
        },
        "notes": {
            "screenshots": "Use screenshots to judge design quality + mobile usability (do not over-index on raw HTML text).",
            "booking_scoring_hint": "Booking path is about discoverability + clicks. A long booking form can be noted as a leak without tanking booking_path.",
        },
    }

    sys_prompt = f"""
You are scoring a clinic website using this rubric ONLY.

{RUBRIC_TEXT}

{WORKED_EXAMPLES}

Scoring rules:
- Each category score must be an integer 0–10.
- Use calibration examples to avoid score inflation: 8+ should be rare unless it truly matches the GOLD example vibe.
- Booking path is about how obvious + low-click the "Book/Consult" path is. If the form itself is long, keep booking_path high if the path is still obvious; list form friction under leaks/quick_wins.
- Write strengths/leaks as short, plain bullets (no essays). Be specific.

Quick Wins rules:
- Each quick win must be a structured object with:
  * action: Clear, actionable text (e.g., "Add a prominent 'Book Now' button above the fold")
  * impact: High, Med, or Low (how much it will improve patient flow)
  * effort: Low, Med, or High (how much work is required)
- Prioritize quick wins by high impact and low effort
- Be specific and actionable

Return output that fits the required JSON schema.
""".strip()

    # Multi-part user content: evidence + screenshots (as data URLs)
    content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": "EVIDENCE_JSON:\n" + json.dumps(evidence_payload, indent=2),
        }
    ]

    # Desktop: send 1 screenshot if available
    for u in (home_d.get("screenshot_urls") or [])[:1]:
        img = _img_to_data_url(u)
        if img:
            content.append({"type": "input_image", "image_url": img})

    # Mobile: send up to 3 screenshots if available
    for u in (home_m.get("screenshot_urls") or [])[:3]:
        img = _img_to_data_url(u)
        if img:
            content.append({"type": "input_image", "image_url": img})

    last_err: Optional[Exception] = None

    # Dedup + keep order
    seen_models: set = set()
    models_to_try: List[str] = []
    for m in [x for x in OPENAI_MODEL_FALLBACKS if x]:
        if m not in seen_models:
            seen_models.add(m)
            models_to_try.append(m)

    for model in models_to_try:
        try:
            # Option A: Structured outputs (preferred)
            if hasattr(client.responses, "parse"):
                try:
                    resp = client.responses.parse(
                        model=model,
                        input=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": content},
                        ],
                        text_format=PatientFlowAIOutput,
                        temperature=0.1,
                    )
                except TypeError:
                    # Some SDK versions don't accept temperature on parse()
                    resp = client.responses.parse(
                        model=model,
                        input=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": content},
                        ],
                        text_format=PatientFlowAIOutput,
                    )

                parsed = getattr(resp, "output_parsed", None)
                if parsed is None:
                    raise RuntimeError("AI returned no parsed output (output_parsed is None).")

                data = _model_to_dict(parsed)
            else:
                # Fallback (older SDK): plain create + manual JSON parse
                resp = client.responses.create(
                    model=model,
                    input=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": content}],
                    temperature=0.1,
                )
                raw = (getattr(resp, "output_text", "") or "").strip()
                try:
                    data = json.loads(raw)
                except Exception:
                    m = re.search(r"\{.*\}", raw, flags=re.S)
                    if not m:
                        raise RuntimeError(f"AI returned non-JSON. First 400 chars: {raw[:400]}")
                    data = json.loads(m.group(0))

            scores = (data.get("scores") or {}) if isinstance(data, dict) else {}
            fixed_scores = {
                "clarity_first_impression": _clamp_int(scores.get("clarity_first_impression")),
                "booking_path": _clamp_int(scores.get("booking_path")),
                "mobile_experience": _clamp_int(scores.get("mobile_experience")),
                "trust_and_proof": _clamp_int(scores.get("trust_and_proof")),
                "treatments_and_offer": _clamp_int(scores.get("treatments_and_offer")),
                "tech_basics": _clamp_int(scores.get("tech_basics")),
            }
            total = sum(fixed_scores.values())
            score10 = round(total / 6.0, 1)

            out: Dict[str, Any] = {
                "clinic_name": (data.get("clinic_name") or "").strip()
                or infer_clinic_name(home_d.get("title", ""), home_d.get("h1", ""), target_url),
                "url": str(target_url),
                "scores": fixed_scores,
                "total_score_60": total,
                "patient_flow_score_10": score10,
                "band": band_from_score(score10),
                "strengths": (data.get("strengths") or [])[:20],
                "leaks": (data.get("leaks") or [])[:20],
                "quick_wins": (data.get("quick_wins") or [])[:20],
                "debug": {
                    "mode": "ai",
                    "ai_model_used": model,
                    "home_desktop_final_url": home_d.get("final_url"),
                    "home_mobile_final_url": home_m.get("final_url"),
                    "structured_output": bool(hasattr(client.responses, "parse")),
                },
            }

            out = _post_guardrails(out, evidence)
            return out

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"AI scoring failed. Last error: {last_err}")


def run_scan(scan_id: str, url: str) -> None:
    conn = db()
    try:
        conn.execute("UPDATE scans SET status=?, updated_at=? WHERE id=?", ("running", now_iso(), scan_id))
        conn.commit()

        scan_dir = ARTIFACTS_DIR / scan_id
        scan_dir.mkdir(parents=True, exist_ok=True)

        home_desktop = fetch_page_evidence(
            url=str(url),
            mobile=False,
            screenshot_paths=[scan_dir / "home_desktop_top.jpg"],
        )

        home_mobile = fetch_page_evidence(
            url=str(url),
            mobile=True,
            screenshot_paths=[
                scan_dir / "home_mobile_top.jpg",
                scan_dir / "home_mobile_mid.jpg",
                scan_dir / "home_mobile_bottom.jpg",
            ],
        )

        evidence: Dict[str, Any] = {
            "target_url": str(url),
            "home_desktop": home_desktop,
            "home_mobile": home_mobile,
        }

        # Save evidence early so the report page can show screenshots while AI is scoring.
        conn.execute(
            "UPDATE scans SET status=?, updated_at=?, evidence_json=? WHERE id=?",
            ("scoring", now_iso(), json.dumps(evidence), scan_id),
        )
        conn.commit()

        if SCORING_MODE == "rules":
            output = score_rules_only(evidence)
        else:
            output = ai_score_patient_flow(str(url), evidence)

        # Generate slug from clinic_name or fallback to domain
        clinic_name = output.get("clinic_name", "")
        if clinic_name:
            slug = slugify(clinic_name)
        else:
            # Fallback to domain (minus www)
            parsed = urlparse(str(url))
            domain = parsed.netloc.replace("www.", "")
            slug = slugify(domain)
        
        # Update scans with slug before marking as done
        conn.execute(
            "UPDATE scans SET status=?, updated_at=?, evidence_json=?, score_json=?, error=?, slug=? WHERE id=?",
            ("done", now_iso(), json.dumps(evidence), json.dumps(output), None, slug, scan_id),
        )
        conn.commit()
        
        # Get public_id for event logging
        row = conn.execute("SELECT public_id FROM scans WHERE id=?", (scan_id,)).fetchone()
        public_id = row["public_id"] if row else None
        
        # Log scan_completed event
        conn.execute(
            "INSERT INTO events (event_type, scan_id, public_id, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            ("scan_completed", scan_id, public_id, json.dumps({"score": output.get("patient_flow_score_10")}), now_iso()),
        )
        conn.commit()

    except Exception as e:
        conn.execute(
            "UPDATE scans SET status=?, updated_at=?, error=? WHERE id=?",
            ("error", now_iso(), str(e), scan_id),
        )
        conn.commit()
    finally:
        conn.close()


HOME_HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>__PRODUCT__</title>
  <link rel="icon" href="/favicon.ico" sizes="any">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
  <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">
  <style>
    :root{
      --bg1:#0b1220; --bg2:#070a12;
      --card: rgba(255,255,255,.06);
      --card2: rgba(255,255,255,.09);
      --text:#e8eefc; --muted:rgba(232,238,252,.72);
      --line: rgba(255,255,255,.10);
      --accent:#7aa2ff;
      --accent2:#7cf7c3;
      --shadow: 0 20px 60px rgba(0,0,0,.55);
      --radius: 18px;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      color:var(--text);
      background:
        radial-gradient(1100px 700px at 20% -10%, rgba(122,162,255,.25), transparent 60%),
        radial-gradient(900px 600px at 90% 10%, rgba(124,247,195,.18), transparent 60%),
        linear-gradient(180deg, var(--bg1), var(--bg2));
      min-height:100vh;
    }
    a{color:inherit; text-decoration:none}
    .wrap{max-width:1050px; margin:0 auto; padding:26px 18px 60px;}
    .topbar{
      display:flex; align-items:center; justify-content:space-between;
      padding:10px 0 24px;
    }
    .brand{display:flex; flex-direction:column; gap:2px}
    .brand .name{font-weight:700; letter-spacing:.2px}
    .brand .sub{font-size:13px; color:var(--muted)}
    .cta{
      display:inline-flex; align-items:center; gap:10px;
      padding:10px 14px; border-radius:999px;
      background:linear-gradient(135deg, rgba(122,162,255,.22), rgba(124,247,195,.14));
      border:1px solid rgba(255,255,255,.14);
      box-shadow: 0 10px 30px rgba(0,0,0,.35);
      font-weight:600;
    }
    .hero{
      display:grid;
      grid-template-columns: 1.2fr .8fr;
      gap:18px;
      align-items:stretch;
    }
    @media (max-width: 900px){ .hero{grid-template-columns:1fr;}}
    .panel{
      background:linear-gradient(180deg, var(--card), rgba(255,255,255,.04));
      border:1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding:18px;
    }
    .h1{
      font-size:34px; line-height:1.1;
      margin:0 0 10px;
      letter-spacing:-.4px;
    }
    .lead{margin:0; color:var(--muted); font-size:15px; line-height:1.5}
    .form{
      display:flex; flex-direction:column; gap:10px; margin-top:14px;
    }
    label{font-size:12px; color:var(--muted); margin:8px 0 2px}
    input{
      width:100%;
      padding:12px 12px;
      border-radius:14px;
      border:1px solid rgba(255,255,255,.14);
      background: rgba(0,0,0,.22);
      color:var(--text);
      outline:none;
    }
    input:focus{border-color: rgba(122,162,255,.55); box-shadow: 0 0 0 4px rgba(122,162,255,.12);}
    button{
      padding:12px 14px;
      border-radius:14px;
      border:1px solid rgba(255,255,255,.14);
      background: linear-gradient(135deg, rgba(122,162,255,.30), rgba(124,247,195,.16));
      color:var(--text);
      font-weight:700;
      cursor:pointer;
    }
    button:hover{filter:brightness(1.06)}
    .hint{color:var(--muted); font-size:13px; margin-top:10px}
    .list{display:grid; gap:10px; margin-top:6px;}
    .item{
      display:flex; gap:10px; align-items:flex-start;
      padding:12px; border-radius:16px;
      border:1px solid rgba(255,255,255,.10);
      background: rgba(255,255,255,.04);
    }
    .dot{
      width:10px; height:10px; border-radius:999px;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      margin-top:5px; flex:0 0 auto;
    }
    .item b{display:block; margin-bottom:2px}
    .fine{margin-top:12px; font-size:12px; color:rgba(232,238,252,.58)}
    .foot{margin-top:18px; font-size:12px; color:rgba(232,238,252,.55)}
    code{background:rgba(255,255,255,.08); padding:2px 6px; border-radius:8px}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div class="brand">
        <div class="name">__APP_NAME__</div>
        <div class="sub">__PRODUCT__</div>
      </div>
      <a class="cta" href="__CTA_URL__" target="_blank" rel="noopener">__CTA_TEXT__</a>
    </div>

    <div class="hero">
      <div class="panel">
        <h1 class="h1">Instant patient-flow score for any clinic website</h1>
        <p class="lead">
          Paste a URL and get a clean report: screenshots, scores (0–60), strengths, leaks, and quick wins.
          Built for consult-driven clinics.
        </p>

        <div class="form">
          <div>
            <label>Clinic website URL</label>
            <input id="url" placeholder="https://example.com" />
          </div>

          <div>
            <label>Email (optional, saved for follow-up)</label>
            <input id="email" placeholder="name@clinic.com" />
          </div>

          <button onclick="runScan()">Run free scan</button>
          <div class="hint" id="hint"></div>
          <div class="fine">Tip: use the homepage URL. The scan is best effort and may miss screenshots on heavily protected sites.</div>
        </div>
      </div>

      <div class="panel">
        <div class="list">
          <div class="item">
            <div class="dot"></div>
            <div>
              <b>Clear scorecard</b>
              6 categories, 0–10 each, total 60. Easy to understand.
            </div>
          </div>
          <div class="item">
            <div class="dot"></div>
            <div>
              <b>Proof with screenshots</b>
              Desktop + mobile captures so the report matches reality.
            </div>
          </div>
          <div class="item">
            <div class="dot"></div>
            <div>
              <b>Actionable next steps</b>
              Quick wins and the biggest leaks you can fix first.
            </div>
          </div>
        </div>

        <div class="foot">
          Mode: <code>__MODE__</code>
        </div>
      </div>
    </div>
  </div>

<script>
async function runScan() {
  const url = document.getElementById('url').value.trim();
  const email = document.getElementById('email').value.trim();
  const hint = document.getElementById('hint');

  if (!url) {
    hint.textContent = "Paste a URL first.";
    return;
  }

  hint.textContent = "Starting scan...";
  const res = await fetch('/api/scan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, email: email || null })
  });

  if (!res.ok) {
    hint.textContent = "Error: " + (await res.text());
    return;
  }

  const data = await res.json();
  window.location.href = data.report_path;
}
</script>
</body>
</html>
""".strip()


@app.get("/", response_class=HTMLResponse)
def home():
    html = HOME_HTML_TEMPLATE
    html = html.replace("__APP_NAME__", APP_NAME)
    html = html.replace("__PRODUCT__", APP_PRODUCT)
    html = html.replace("__MODE__", SCORING_MODE)
    html = html.replace("__CTA_TEXT__", PRIMARY_CTA_TEXT)
    html = html.replace("__CTA_URL__", PRIMARY_CTA_URL)
    return html


@app.get("/health")
def health():
    ok, msg = _ai_ready()
    return {
        "ok": True,
        "scoring_mode": SCORING_MODE,
        "openai_installed": OpenAI is not None,
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "openai_model_env": os.getenv("OPENAI_MODEL", OPENAI_MODEL_DEFAULT),
        "ai_ready": ok,
        "ai_message": msg or None,
    }


@app.post("/api/scan")
def create_scan(req: ScanRequest):
    email = (req.email or "").strip() or None
    if email and not EMAIL_RE.match(email):
        raise HTTPException(status_code=422, detail="Email looks invalid. Leave it blank or use a real address.")

    # Fail fast (better than creating a scan that will just error later)
    if SCORING_MODE == "ai":
        ok, msg = _ai_ready()
        if not ok:
            raise HTTPException(status_code=500, detail=msg)

    scan_id = uuid.uuid4().hex
    
    # Generate unique 8-char public_id
    conn = db()
    public_id = None
    try:
        for _ in range(MAX_PUBLIC_ID_ATTEMPTS):
            candidate = uuid.uuid4().hex[:8]
            # Check if this public_id already exists
            existing = conn.execute("SELECT id FROM scans WHERE public_id=?", (candidate,)).fetchone()
            if not existing:
                public_id = candidate
                break
        
        if not public_id:
            raise HTTPException(status_code=500, detail="Failed to generate unique public_id")
        
        conn.execute(
            "INSERT INTO scans (id, url, status, created_at, updated_at, email, public_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (scan_id, str(req.url), "queued", now_iso(), now_iso(), email, public_id),
        )
        conn.commit()
        
        # Log scan_started event
        conn.execute(
            "INSERT INTO events (event_type, scan_id, public_id, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            ("scan_started", scan_id, public_id, json.dumps({"url": str(req.url), "email": email}), now_iso()),
        )
        conn.commit()
    finally:
        conn.close()

    t = threading.Thread(target=run_scan, args=(scan_id, str(req.url)), daemon=True)
    t.start()

    return {
        "id": scan_id,
        "public_id": public_id,
        "report_path": f"/report/scanning-{public_id}"
    }


@app.get("/api/scan/{scan_id}")
def get_scan(scan_id: str):
    conn = db()
    row = conn.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Scan not found")

    resp = {
        "id": row["id"],
        "url": row["url"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "email": row["email"],
        "error": row["error"],
        "score": json.loads(row["score_json"]) if row["score_json"] else None,
        "evidence": json.loads(row["evidence_json"]) if row["evidence_json"] else None,
    }
    return JSONResponse(resp)


@app.post("/api/events")
def log_event(req: EventRequest):
    """Log a frontend event to the events table."""
    conn = db()
    try:
        conn.execute(
            "INSERT INTO events (event_type, scan_id, public_id, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (req.event_type, req.scan_id, req.public_id, json.dumps(req.metadata) if req.metadata else None, now_iso()),
        )
        conn.commit()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log event: {str(e)}")
    finally:
        conn.close()


@app.post("/api/email_report")
def email_report(req: EmailReportRequest):
    """Send a plain-text email with the report link."""
    # Validate email
    if not EMAIL_RE.match(req.email):
        raise HTTPException(status_code=422, detail="Invalid email address")
    
    # Validate report exists
    conn = db()
    row = conn.execute("SELECT * FROM scans WHERE public_id=?", (req.report_id,)).fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Build report URL
    # Use the slug if available, otherwise use "scanning"
    db_slug = row["slug"] if row["slug"] else "scanning"
    report_url = f"/report/{db_slug}-{req.report_id}"
    
    # For now, we'll just log this action since we don't have an email service configured
    # In a production environment, you would integrate with an email service like SendGrid, AWS SES, etc.
    
    # Log the email_report event
    conn = db()
    try:
        conn.execute(
            "INSERT INTO events (event_type, scan_id, public_id, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            ("email_report_requested", row["id"], req.report_id, json.dumps({"email": req.email, "report_url": report_url}), now_iso()),
        )
        conn.commit()
    finally:
        conn.close()
    
    # In production, send email here
    # For now, return success with a message
    return {
        "ok": True,
        "message": f"Email would be sent to {req.email} with link: {report_url}",
        "report_url": report_url
    }


REPORT_HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>__PAGE_TITLE__</title>
  <link rel="canonical" href="__CANONICAL_URL__">
  
  <!-- OpenGraph tags -->
  <meta property="og:title" content="__OG_TITLE__">
  <meta property="og:description" content="__OG_DESC__">
  <meta property="og:url" content="__OG_URL__">
  <meta property="og:image" content="__OG_IMAGE__">
  <meta property="og:type" content="website">
  
  <!-- Twitter Card tags -->
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="__OG_TITLE__">
  <meta name="twitter:description" content="__OG_DESC__">
  <meta name="twitter:image" content="__OG_IMAGE__">
  
  <link rel="icon" href="/favicon.ico" sizes="any">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
  <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">
  <style>
    :root{
      --bg1:#0b1220; --bg2:#070a12;
      --card: rgba(255,255,255,.06);
      --text:#e8eefc; --muted:rgba(232,238,252,.72);
      --line: rgba(255,255,255,.10);
      --accent:#7aa2ff;
      --accent2:#7cf7c3;
      --shadow: 0 20px 60px rgba(0,0,0,.55);
      --radius: 18px;
      --ring-color: rgba(122,162,255,.75);
      --ring-bg: rgba(255,255,255,.08);
      --ring-thickness: 62%;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      color:var(--text);
      background:
        radial-gradient(1100px 700px at 20% -10%, rgba(122,162,255,.25), transparent 60%),
        radial-gradient(900px 600px at 90% 10%, rgba(124,247,195,.18), transparent 60%),
        linear-gradient(180deg, var(--bg1), var(--bg2));
      min-height:100vh;
    }
    a{color:inherit; text-decoration:none}
    .wrap{max-width:1050px; margin:0 auto; padding:22px 18px 64px;}
    .topbar{display:flex; align-items:center; justify-content:space-between; gap:12px; padding:8px 0 16px;}
    .brand{display:flex; flex-direction:column; gap:2px}
    .brand .name{font-weight:700; letter-spacing:.2px}
    .brand .sub{font-size:13px; color:var(--muted)}
    .actions{display:flex; align-items:center; gap:10px; flex-wrap:wrap;}
    .btn, .btn2{
      display:inline-flex; align-items:center; justify-content:center;
      padding:10px 14px; border-radius:999px;
      border:1px solid rgba(255,255,255,.14);
      font-weight:700;
      cursor:pointer;
      transition:filter 0.2s;
    }
    .btn{
      background:linear-gradient(135deg, rgba(122,162,255,.30), rgba(124,247,195,.16));
      box-shadow: 0 10px 30px rgba(0,0,0,.35);
    }
    .btn:hover{filter:brightness(1.06);}
    .btn2{background: rgba(255,255,255,.06);}
    .btn2:hover{filter:brightness(1.1);}
    
    /* Shared k-btn styles with higher contrast */
    .k-btn{
      display:inline-flex; align-items:center; justify-content:center;
      padding:10px 14px; border-radius:999px;
      border:2px solid rgba(255,255,255,.30);
      font-weight:700;
      cursor:pointer;
      background: rgba(255,255,255,.08);
      color: rgba(232,238,252,.95);
      transition: all 0.2s ease;
      outline:none;
    }
    .k-btn:hover{
      background: rgba(255,255,255,.14);
      border-color: rgba(255,255,255,.40);
      filter:brightness(1.08);
    }
    .k-btn:focus-visible{
      border-color: rgba(122,162,255,.75);
      box-shadow: 0 0 0 4px rgba(122,162,255,.20);
    }
    
    /* Primary variant with higher prominence */
    .k-btn--primary{
      background:linear-gradient(135deg, rgba(122,162,255,.35), rgba(124,247,195,.20));
      border-color: rgba(122,162,255,.45);
      box-shadow: 0 10px 30px rgba(0,0,0,.35);
    }
    .k-btn--primary:hover{
      background:linear-gradient(135deg, rgba(122,162,255,.45), rgba(124,247,195,.28));
      border-color: rgba(122,162,255,.60);
      filter:brightness(1.10);
    }
    .k-btn--primary:focus-visible{
      border-color: rgba(122,162,255,.85);
      box-shadow: 0 0 0 4px rgba(122,162,255,.25), 0 10px 30px rgba(0,0,0,.35);
    }
    .panel{
      background:linear-gradient(180deg, var(--card), rgba(255,255,255,.04));
      border:1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding:18px;
    }
    .headerRow{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:14px;
      margin-bottom:14px;
      flex-wrap:wrap;
    }
    .headerLeft{display:flex; flex-direction:column; gap:8px; flex:1; min-width:0;}
    .breadcrumb{
      font-size:13px;
      color:var(--muted);
      margin-bottom:4px;
    }
    .breadcrumb a{
      color:var(--muted);
      text-decoration:none;
      transition:color 0.2s;
    }
    .breadcrumb a:hover{
      color:var(--text);
    }
    .breadcrumb-sep{
      margin:0 6px;
      color:rgba(232,238,252,.4);
    }
    .h1{margin:0; font-size:28px; letter-spacing:-.35px}
    .meta{display:flex; gap:10px; flex-wrap:wrap; align-items:center; color:var(--muted); font-size:13px}
    .pill{
      display:inline-flex; align-items:center; gap:8px;
      padding:6px 10px; border-radius:999px;
      border:1px solid rgba(255,255,255,.14);
      background: rgba(0,0,0,.18);
      color: rgba(232,238,252,.88);
    }
    .status{margin-top:8px; color:var(--muted); font-size:13px}
    .summaryRow{
      display:grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap:14px;
      margin-bottom:14px;
    }
    @media (max-width: 1200px){ .summaryRow{grid-template-columns:1fr 1fr;}}
    @media (max-width: 980px){ .summaryRow{grid-template-columns:1fr;}}
    .screenshotsCard{
      display:flex; flex-direction:column;
    }
    .tabs{
      display:flex; gap:8px; margin-bottom:14px; border-bottom:1px solid var(--line);
    }
    .tab{
      padding:10px 16px; cursor:pointer; position:relative;
      color:var(--muted); font-weight:600; font-size:13px;
      border:none; background:none; transition:color 0.2s;
    }
    .tab:hover{color:var(--text);}
    .tab.active{color:var(--text);}
    .tab.active::after{
      content:''; position:absolute; bottom:-1px; left:0; right:0;
      height:2px; background:linear-gradient(90deg, var(--accent), var(--accent2));
    }
    .tabContent{display:none;}
    .tabContent.active{display:block;}
    .thumbGrid{
      display:grid; grid-template-columns:repeat(auto-fill, minmax(160px, 1fr));
      gap:12px; margin-top:12px;
    }
    .thumb{
      position:relative; border-radius:12px; overflow:hidden; cursor:pointer;
      border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.04);
      transition:transform 0.2s, box-shadow 0.2s;
    }
    .thumb:hover{
      transform:translateY(-2px);
      box-shadow:0 8px 20px rgba(0,0,0,.4);
    }
    .thumb img{display:block; width:100%; height:auto; background:#fff;}
    .thumbLabel{
      padding:8px 10px; font-size:11px; color:rgba(232,238,252,.72);
      background:rgba(0,0,0,.3);
    }
    .modal{
      display:none; position:fixed; inset:0; z-index:1000;
      background:rgba(0,0,0,.92); align-items:center; justify-content:center;
    }
    .modal.active{display:flex;}
    .modalContent{
      position:relative; max-width:90vw; max-height:90vh;
      display:flex; flex-direction:column; align-items:center;
    }
    .modalImg{
      max-width:100%; max-height:80vh; border-radius:12px;
      box-shadow:0 20px 60px rgba(0,0,0,.8);
    }
    .modalLabel{
      margin-top:12px; font-size:14px; color:var(--text);
      padding:8px 16px; border-radius:999px;
      background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14);
    }
    .modalClose, .modalNav{
      position:absolute; width:48px; height:48px; border-radius:50%;
      background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.20);
      display:flex; align-items:center; justify-content:center;
      cursor:pointer; color:var(--text); font-size:24px; font-weight:700;
      transition:background 0.2s;
    }
    .modalClose:hover, .modalNav:hover{background:rgba(255,255,255,.16);}
    .modalClose{top:20px; right:20px;}
    .modalNav{top:50%; transform:translateY(-50%);}
    .modalPrev{left:20px;}
    .modalNext{right:20px;}
    .scoreboardCard{
      display:flex; flex-direction:column;
      padding:18px; border-radius: var(--radius);
      border:1px solid rgba(255,255,255,.12);
      background: linear-gradient(135deg, rgba(122,162,255,.16), rgba(124,247,195,.10));
    }
    .scoreboardTop{display:flex; justify-content:space-between; align-items:flex-start; gap:16px; margin-bottom:14px;}
    @media (max-width: 480px){ .scoreboardTop{flex-direction:column; gap:12px;}}
    .scoreboardPrimary{display:flex; flex-direction:column; gap:4px; position:relative; align-items:center;}
    .scoreRingWrap{position:relative; width:180px; height:180px; display:flex; align-items:center; justify-content:center;}
    .scoreRing{
      position:absolute; inset:0; border-radius:50%;
      background:conic-gradient(
        from -90deg,
        var(--ring-color) 0%,
        var(--ring-color) var(--progress, 0%),
        var(--ring-bg) var(--progress, 0%),
        var(--ring-bg) 100%
      );
      mask:radial-gradient(circle, transparent 0%, transparent var(--ring-thickness), black var(--ring-thickness), black 100%);
      -webkit-mask:radial-gradient(circle, transparent 0%, transparent var(--ring-thickness), black var(--ring-thickness), black 100%);
    }
    .scoreRingInner{position:relative; z-index:1; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center;}
    .scoreLarge{font-size:52px; font-weight:900; letter-spacing:-.9px; line-height:1.2;}
    .scoreLabel{font-size:13px; color:var(--muted); margin-top:4px;}
    .scoreSecondary{display:flex; flex-direction:column; align-items:flex-end; gap:4px;}
    .scoreMedium{font-size:32px; font-weight:900; letter-spacing:-.6px; line-height:1.2;}
    .verdictChip{
      padding:6px 12px; border-radius:999px;
      border:1px solid rgba(255,255,255,.18);
      background: rgba(0,0,0,.22);
      font-size:12px; font-weight:600;
      color:rgba(232,238,252,.92);
      margin-top:4px;
      white-space:nowrap;
    }
    .miniBars{display:grid; gap:6px; margin-top:8px;}
    .miniBarRow{display:grid; gap:4px;}
    .miniBarLabel{font-size:11px; color:rgba(232,238,252,.78); text-transform:uppercase; letter-spacing:.3px;}
    .miniTrack{height:6px; border-radius:999px; background:rgba(255,255,255,.10); overflow:hidden; border:1px solid rgba(255,255,255,.10); position:relative;}
    .miniFill{position:absolute; left:0; top:0; height:100%; border-radius:999px; background:linear-gradient(90deg, rgba(122,162,255,.85), rgba(124,247,195,.75));}
    .miniScore{font-size:11px; color:rgba(232,238,252,.88); font-weight:700;}
    .grid{
      display:grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap:14px;
      margin-top:14px;
    }
    @media (max-width: 980px){ .grid{grid-template-columns:1fr;}}
    h2{margin:0 0 10px; font-size:14px; letter-spacing:.2px; color:rgba(232,238,252,.92); text-transform:uppercase}
    .bars{display:grid; gap:10px;}
    .barRow{display:grid; gap:6px;}
    .barTop{display:flex; justify-content:space-between; gap:10px; font-size:13px; color:rgba(232,238,252,.88)}
    .track{height:10px; border-radius:999px; background:rgba(255,255,255,.10); overflow:hidden; border:1px solid rgba(255,255,255,.10)}
    .fill{height:100%; width:0%; border-radius:999px; background:linear-gradient(90deg, rgba(122,162,255,.85), rgba(124,247,195,.75));}
    
    /* Score cards grid */
    .scoreCards{
      display:grid;
      grid-template-columns: repeat(3, 1fr);
      gap:12px;
    }
    @media (max-width: 980px){ .scoreCards{grid-template-columns:1fr;}}
    .scoreCard{
      display:flex;
      flex-direction:column;
      gap:8px;
      padding:14px;
      border-radius:16px;
      border:1px solid rgba(255,255,255,.10);
      background: rgba(255,255,255,.04);
    }
    .scoreCardTop{
      display:flex;
      justify-content:space-between;
      align-items:flex-start;
      gap:8px;
    }
    .scoreCardLabel{
      font-size:12px;
      font-weight:600;
      letter-spacing:.2px;
      color:rgba(232,238,252,.92);
      line-height:1.3;
    }
    .scoreCardValue{
      font-size:18px;
      font-weight:900;
      letter-spacing:-.3px;
      color:var(--text);
      flex-shrink:0;
    }
    .scoreCardHelper{
      font-size:11px;
      color:var(--muted);
      line-height:1.3;
      margin-top:-2px;
    }
    .scoreCardTrack{
      height:6px;
      border-radius:999px;
      background:rgba(255,255,255,.10);
      overflow:hidden;
      border:1px solid rgba(255,255,255,.10);
      position:relative;
    }
    .scoreCardFill{
      position:absolute;
      left:0;
      top:0;
      height:100%;
      border-radius:999px;
      background:linear-gradient(90deg, rgba(122,162,255,.85), rgba(124,247,195,.75));
    }
    ul{margin:0; padding:0; list-style:none; color:rgba(232,238,252,.84)}
    li{
      margin:10px 0;
      padding:12px;
      border-radius:12px;
      border:1px solid rgba(255,255,255,.08);
      background: rgba(255,255,255,.03);
      line-height:1.4;
      display:flex;
      gap:12px;
      align-items:flex-start;
    }
    .itemIcon{
      flex-shrink:0;
      width:20px;
      height:20px;
      display:flex;
      align-items:center;
      justify-content:center;
      font-size:14px;
      margin-top:2px;
    }
    .itemContent{
      flex:1;
      min-width:0;
    }
    .itemTitle{
      font-weight:700;
      color:rgba(232,238,252,.95);
      margin-bottom:4px;
      font-size:14px;
    }
    .itemWhy{
      color:rgba(232,238,252,.75);
      font-size:13px;
      line-height:1.4;
      margin-bottom:6px;
    }
    .itemEvidence{
      display:inline-flex;
      align-items:center;
      gap:4px;
      font-size:12px;
      color:rgba(122,162,255,.92);
      text-decoration:none;
      padding:4px 8px;
      border-radius:6px;
      background:rgba(122,162,255,.10);
      border:1px solid rgba(122,162,255,.20);
      transition:background 0.2s;
      margin-top:4px;
    }
    .itemEvidence:hover{
      background:rgba(122,162,255,.18);
    }
    
    /* Quick Win Checklist Styles */
    .quickWinCheckbox{
      flex-shrink:0;
      width:18px;
      height:18px;
      border:2px solid rgba(255,255,255,.30);
      border-radius:4px;
      background:rgba(0,0,0,.2);
      cursor:pointer;
      display:flex;
      align-items:center;
      justify-content:center;
      transition:all 0.2s;
      margin-top:2px;
    }
    .quickWinCheckbox:hover{
      border-color:rgba(122,162,255,.60);
      background:rgba(122,162,255,.10);
    }
    .quickWinCheckbox.checked{
      background:linear-gradient(135deg, rgba(122,162,255,.45), rgba(124,247,195,.30));
      border-color:rgba(122,162,255,.85);
      border-width:2.5px;
    }
    .quickWinCheckbox.checked::after{
      content:'✓';
      color:var(--text);
      font-size:13px;
      font-weight:900;
    }
    .quickWinAction{
      flex:1;
      color:rgba(232,238,252,.88);
      font-size:14px;
      line-height:1.4;
    }
    .quickWinChips{
      display:flex;
      gap:6px;
      margin-top:6px;
      flex-wrap:wrap;
    }
    .chip{
      display:inline-flex;
      align-items:center;
      padding:4px 10px;
      border-radius:999px;
      font-size:11px;
      font-weight:700;
      letter-spacing:.3px;
      text-transform:uppercase;
      border:1px solid;
    }
    .chipImpactHigh{
      background:rgba(255,100,100,.18);
      color:rgba(255,180,180,.95);
      border-color:rgba(255,100,100,.35);
    }
    .chipImpactMed{
      background:rgba(255,200,100,.15);
      color:rgba(255,220,140,.95);
      border-color:rgba(255,200,100,.30);
    }
    .chipImpactLow{
      background:rgba(200,200,200,.12);
      color:rgba(220,220,220,.85);
      border-color:rgba(200,200,200,.25);
    }
    .chipEffortLow{
      background:rgba(124,247,195,.15);
      color:rgba(150,255,210,.95);
      border-color:rgba(124,247,195,.30);
    }
    .chipEffortMed{
      background:rgba(122,162,255,.15);
      color:rgba(160,190,255,.95);
      border-color:rgba(122,162,255,.30);
    }
    .chipEffortHigh{
      background:rgba(255,140,100,.15);
      color:rgba(255,180,150,.95);
      border-color:rgba(255,140,100,.30);
    }
    .err{
      background: rgba(255, 70, 70, .10);
      border: 1px solid rgba(255, 70, 70, .20);
      padding:10px 12px;
      border-radius:14px;
      color: rgba(255, 200, 200, .95);
      margin-top:10px;
      font-size:13px;
    }

    /* Technical details (debug-only) */
    #rawWrap{display:none; margin-top:14px;}
    details{
      border:1px solid rgba(255,255,255,.10);
      border-radius:16px;
      background: rgba(0,0,0,.18);
      padding:12px 14px;
    }
    summary{
      cursor:pointer;
      color:rgba(232,238,252,.90);
      font-weight:800;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:12px;
      list-style:none;
    }
    summary::-webkit-details-marker{display:none;}
    .copyBtn{
      display:inline-flex; align-items:center; justify-content:center;
      padding:8px 10px;
      border-radius:999px;
      border:1px solid rgba(255,255,255,.14);
      background: rgba(255,255,255,.06);
      color:rgba(232,238,252,.92);
      font-weight:800;
      font-size:12px;
      user-select:none;
    }
    .copyBtn:hover{filter:brightness(1.07)}
    pre{margin:12px 0 0; background:#0b0f19; color:#cfe3ff; padding:14px; border-radius:14px; overflow:auto; border:1px solid rgba(255,255,255,.08)}
    
    /* Blueprint Card Styles */
    .blueprintCard{
      display:flex;
      flex-direction:column;
      gap:16px;
    }
    .blueprintCard h2{
      margin:0 0 12px;
      font-size:18px;
      letter-spacing:-.2px;
      color:var(--text);
    }
    .blueprintBullets{
      display:flex;
      flex-direction:column;
      gap:10px;
      margin-bottom:8px;
    }
    .blueprintBullet{
      display:flex;
      gap:12px;
      align-items:flex-start;
      font-size:14px;
      line-height:1.5;
      color:rgba(232,238,252,.88);
    }
    .blueprintBullet::before{
      content:'✓';
      color:rgba(124,247,195,.92);
      font-weight:900;
      font-size:16px;
      flex-shrink:0;
      margin-top:2px;
    }
    .blueprintPrice{
      display:flex;
      align-items:baseline;
      gap:8px;
      margin:8px 0;
      padding:12px 0;
      border-top:1px solid rgba(255,255,255,.10);
      border-bottom:1px solid rgba(255,255,255,.10);
    }
    .blueprintPriceLabel{
      font-size:13px;
      color:var(--muted);
      font-weight:600;
    }
    .blueprintPriceValue{
      font-size:28px;
      font-weight:900;
      letter-spacing:-.5px;
      background:linear-gradient(135deg, var(--accent), var(--accent2));
      -webkit-background-clip:text;
      -webkit-text-fill-color:transparent;
      background-clip:text;
    }
    .blueprintCredit{
      font-size:12px;
      color:rgba(124,247,195,.85);
      font-style:italic;
      margin:4px 0 12px;
    }
    .blueprintCTA{
      display:flex;
      align-items:center;
      justify-content:center;
      padding:12px 18px;
      border-radius:14px;
      border:2px solid rgba(122,162,255,.45);
      background:linear-gradient(135deg, rgba(122,162,255,.35), rgba(124,247,195,.20));
      color:var(--text);
      font-weight:800;
      font-size:15px;
      cursor:pointer;
      text-decoration:none;
      transition:all 0.2s;
      box-shadow: 0 10px 30px rgba(0,0,0,.35);
    }
    .blueprintCTA:hover{
      background:linear-gradient(135deg, rgba(122,162,255,.45), rgba(124,247,195,.28));
      border-color:rgba(122,162,255,.60);
      filter:brightness(1.10);
      transform:translateY(-1px);
    }
    
    /* Email Report Section */
    .emailReportSection{
      margin-top:18px;
      padding-top:18px;
      border-top:1px solid rgba(255,255,255,.10);
    }
    .emailReportLabel{
      font-size:13px;
      color:var(--muted);
      font-weight:600;
      margin-bottom:8px;
    }
    .emailReportForm{
      display:flex;
      flex-direction:column;
      gap:10px;
    }
    .emailReportInput{
      width:100%;
      padding:12px;
      border-radius:12px;
      border:1px solid rgba(255,255,255,.14);
      background:rgba(0,0,0,.22);
      color:var(--text);
      font-size:14px;
      outline:none;
      transition:all 0.2s;
    }
    .emailReportInput:focus{
      border-color:rgba(122,162,255,.55);
      box-shadow:0 0 0 4px rgba(122,162,255,.12);
    }
    .emailReportInput::placeholder{
      color:rgba(232,238,252,.45);
    }
    .emailReportBtn{
      display:flex;
      align-items:center;
      justify-content:center;
      padding:12px 16px;
      border-radius:12px;
      border:2px solid rgba(255,255,255,.30);
      background:rgba(255,255,255,.08);
      color:rgba(232,238,252,.95);
      font-weight:700;
      font-size:14px;
      cursor:pointer;
      transition:all 0.2s;
      outline:none;
    }
    .emailReportBtn:hover{
      background:rgba(255,255,255,.14);
      border-color:rgba(255,255,255,.40);
    }
    .emailReportBtn:disabled{
      opacity:0.5;
      cursor:not-allowed;
    }
    .emailReportHint{
      font-size:12px;
      color:rgba(124,247,195,.85);
      margin-top:4px;
      display:none;
    }
    .emailReportHint.visible{
      display:block;
    }
    
    /* Print styles for PDF */
    @media print {
      /* Force white background and black text */
      body{
        background:white !important;
        color:black !important;
      }
      
      /* Hide all interactive UI elements */
      .topbar, .modal, #rawWrap, .actions,
      .btn, .btn2, .k-btn, .k-btn--primary,
      button, .copyBtn, .modalClose, .modalNav,
      .quickWinCheckbox, .blueprintCard, .emailReportSection{
        display:none !important;
      }
      
      /* Remove shadows and glass effects */
      .panel, .scoreboardCard, .thumb, .scoreCard, li,
      .pill, .verdictChip, .chip, .item, .err{
        box-shadow:none !important;
        background:white !important;
        border:1px solid #ddd !important;
      }
      
      /* Prevent page breaks inside cards and important sections */
      .panel, .scoreboardCard, .scoreCard, .summaryRow,
      .grid > div, li, .thumb, .barRow, .miniBarRow{
        page-break-inside:avoid !important;
        break-inside:avoid !important;
      }
      
      /* Make all text fully readable - high contrast */
      .wrap, .h1, .meta, .pill, .status,
      .scoreLabel, .verdictChip, .miniBarLabel, .miniScore,
      .barTop, .scoreCardLabel, .scoreCardValue, .scoreCardHelper,
      .itemIcon, .itemContent, .itemTitle, .itemWhy, .itemEvidence,
      .quickWinAction, .chip, .thumbLabel, .modalLabel,
      .brand .name, .brand .sub, h2, ul, li,
      .lead, .hint, .fine, .foot, label, .err{
        color:black !important;
        opacity:1 !important;
      }
      
      /* Ensure color elements print correctly */
      .scoreRing, .fill, .miniFill, .scoreCardFill, .dot{
        print-color-adjust:exact !important;
        -webkit-print-color-adjust:exact !important;
        color-adjust:exact !important;
      }
      
      /* Optimize layout for print */
      .wrap{
        max-width:100% !important;
        padding:10px !important;
      }
      
      /* Ensure images are visible */
      .thumb img, .modalImg{
        display:block !important;
        max-width:100% !important;
      }
      
      /* Make track backgrounds visible in print */
      .track, .miniTrack, .scoreCardTrack{
        background:#eee !important;
        border:1px solid #ccc !important;
      }
      
      /* Keep gradient fills visible */
      .fill, .miniFill, .scoreCardFill{
        background:#4a7bc8 !important;
      }
      
      /* Simplify grid layouts for better printing */
      .grid, .scoreCards{
        display:block !important;
      }
      
      .grid > div, .scoreCard{
        margin-bottom:15px !important;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div class="brand">
        <div class="name">__APP_NAME__</div>
        <div class="sub">__PRODUCT__</div>
      </div>
      <div class="actions">
        <a class="btn2" href="/">New scan</a>
        <a class="btn" href="__CTA_URL__" target="_blank" rel="noopener">__CTA_TEXT__</a>
      </div>
    </div>

    <div class="panel headerRow">
      <div class="headerLeft">
        <div class="breadcrumb">
          <a href="/">Reports</a>
          <span class="breadcrumb-sep">/</span>
          <span id="clinicNameCrumb">Patient-Flow Report</span>
        </div>
        <div class="h1" id="clinicName">Patient-Flow Report</div>
        <div class="meta">
          <a id="siteUrl" href="#" target="_blank" rel="noopener" class="pill">Website</a>
          <span class="pill" id="band">Band</span>
          <span class="pill">Mode: __MODE__</span>
        </div>
        <div class="status" id="status">Loading...</div>
      </div>
      <div class="actions">
        <button class="k-btn" id="copyLinkBtn" onclick="copyReportLink()">Copy report link</button>
        <button class="k-btn k-btn--primary" id="downloadPdfBtn" onclick="downloadPDF()">Download PDF</button>
      </div>
    </div>

    <div class="summaryRow">
      <div class="panel scoreboardCard">
        <div class="scoreboardTop">
          <div class="scoreboardPrimary">
            <div class="scoreRingWrap">
              <div class="scoreRing" id="scoreRing"></div>
              <div class="scoreRingInner">
                <div class="scoreLarge" id="score60Main">--</div>
                <div class="scoreLabel">out of 60</div>
              </div>
            </div>
          </div>
          <div class="scoreSecondary">
            <div class="scoreMedium" id="score10Main">--</div>
            <div class="scoreLabel">out of 10</div>
            <div class="verdictChip" id="verdictChip">—</div>
          </div>
        </div>
        <div class="miniBars" id="miniScoreBars"></div>
      </div>
      <div class="panel screenshotsCard">
        <h2>Screenshots</h2>
        <div class="tabs">
          <button class="tab active" data-tab="desktop">Desktop</button>
          <button class="tab" data-tab="mobile">Mobile</button>
        </div>
        <div class="tabContent active" id="desktop-tab">
          <div class="thumbGrid" id="desktopGrid"></div>
        </div>
        <div class="tabContent" id="mobile-tab">
          <div class="thumbGrid" id="mobileGrid"></div>
        </div>
        <div id="errs"></div>
      </div>
      <div class="panel blueprintCard">
        <h2>Get the Blueprint</h2>
        <div class="blueprintBullets">
          <div class="blueprintBullet">Complete patient-flow audit & roadmap</div>
          <div class="blueprintBullet">Prioritized fixes with timeline</div>
          <div class="blueprintBullet">Custom design & copy recommendations</div>
          <div class="blueprintBullet">Mobile optimization strategy</div>
        </div>
        <div class="blueprintPrice">
          <span class="blueprintPriceLabel">Investment:</span>
          <span class="blueprintPriceValue">$1,000</span>
        </div>
        <div class="blueprintCredit">Fully credited toward rebuild if you proceed</div>
        <a class="blueprintCTA" href="__CTA_URL__" target="_blank" rel="noopener" id="blueprintCTA">__CTA_TEXT__</a>
        
        <div class="emailReportSection">
          <div class="emailReportLabel">Send full report to my inbox</div>
          <div class="emailReportForm">
            <input type="email" class="emailReportInput" id="emailInput" placeholder="your.email@clinic.com" />
            <button class="emailReportBtn" id="sendEmailBtn" onclick="sendReportEmail()">Send Report</button>
            <div class="emailReportHint" id="emailHint"></div>
          </div>
        </div>
      </div>
    </div>

    <div class="panel">
      <h2>Scores</h2>
      <div class="scoreCards" id="scoreBars"></div>
    </div>

    <div class="grid">
      <div class="panel">
        <h2>Strengths</h2>
        <ul id="strengths"><li>Waiting for score...</li></ul>
      </div>
      <div class="panel">
        <h2>Leaks</h2>
        <ul id="leaks"><li>Waiting for score...</li></ul>
      </div>
      <div class="panel">
        <h2>Quick wins</h2>
        <ul id="quickWins"><li>Waiting for score...</li></ul>
      </div>
    </div>


    <div id="rawWrap">
      <details>
        <summary>
          <span>Technical details</span>
          <span class="copyBtn" id="copyJson" role="button" tabindex="0">Copy JSON</span>
        </summary>
        <pre id="out">...</pre>
      </details>
    </div>
  </div>

  <div class="modal" id="modal">
    <div class="modalContent">
      <div class="modalClose" id="modalClose">&times;</div>
      <div class="modalNav modalPrev" id="modalPrev">&lsaquo;</div>
      <div class="modalNav modalNext" id="modalNext">&rsaquo;</div>
      <img class="modalImg" id="modalImg" src="" alt="Screenshot">
      <div class="modalLabel" id="modalLabel"></div>
    </div>
  </div>

<script>
const scanId = "__SCAN_ID__";
const params = new URLSearchParams(window.location.search);
const debug = params.get('debug') === '1';
const printMode = params.get('print') === '1';

const rawWrap = document.getElementById('rawWrap');
if (rawWrap) rawWrap.style.display = debug ? 'block' : 'none';

// Hide interactive elements in print mode
if (printMode) {
  document.querySelectorAll('.actions, .btn, .btn2, .k-btn, button').forEach(el => {
    el.style.display = 'none';
  });
}

// Extract public_id from URL path
function getPublicId() {
  const path = window.location.pathname;
  const match = path.match(/\/report\/.*-([a-f0-9]{8})$/);
  return match ? match[1] : null;
}

// Helper function to log events
async function logEvent(eventType, metadata = {}) {
  const publicId = getPublicId();
  if (!scanId || scanId === "__SCAN_ID__") {
    console.warn('scanId not initialized, skipping event logging');
    return;
  }
  try {
    await fetch('/api/events', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        event_type: eventType,
        scan_id: scanId,
        public_id: publicId,
        metadata: metadata
      })
    });
  } catch (e) {
    console.error('Failed to log event:', e);
  }
}

// Log report_viewed event on page load
logEvent('report_viewed');

// Send report email function
async function sendReportEmail() {
  const emailInput = document.getElementById('emailInput');
  const btn = document.getElementById('sendEmailBtn');
  const hint = document.getElementById('emailHint');
  const email = emailInput.value.trim();
  
  if (!email) {
    hint.textContent = "Please enter your email address";
    hint.classList.add('visible');
    hint.style.color = 'rgba(255, 200, 200, .95)';
    setTimeout(() => { hint.classList.remove('visible'); }, 3000);
    return;
  }
  
  const originalText = btn.textContent;
  btn.textContent = "Sending...";
  btn.disabled = true;
  
  const publicId = getPublicId();
  
  try {
    const res = await fetch('/api/email_report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: email,
        report_id: publicId
      })
    });
    
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText);
    }
    
    const data = await res.json();
    hint.textContent = "✓ Report link sent to your inbox!";
    hint.style.color = 'rgba(124,247,195,.85)';
    hint.classList.add('visible');
    emailInput.value = '';
    
    setTimeout(() => {
      btn.textContent = originalText;
      btn.disabled = false;
      hint.classList.remove('visible');
    }, 3000);
  } catch (error) {
    console.error('Failed to send email:', error);
    hint.textContent = "Failed to send email. Please try again.";
    hint.style.color = 'rgba(255, 200, 200, .95)';
    hint.classList.add('visible');
    
    setTimeout(() => {
      btn.textContent = originalText;
      btn.disabled = false;
      hint.classList.remove('visible');
    }, 3000);
  }
}

// Copy report link function
async function copyReportLink() {
  const btn = document.getElementById('copyLinkBtn');
  const originalText = btn.textContent;
  const linkToCopy = window.location.origin + window.location.pathname;
  
  // Log event
  logEvent('copy_report_link_clicked');
  
  try {
    await navigator.clipboard.writeText(linkToCopy);
    btn.textContent = "Link copied!";
    setTimeout(() => { btn.textContent = originalText; }, 2000);
  } catch (e) {
    // Fallback for browsers that don't support clipboard API
    const ta = document.createElement('textarea');
    ta.value = linkToCopy;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
    btn.textContent = "Link copied!";
    setTimeout(() => { btn.textContent = originalText; }, 2000);
  }
}

// Download PDF function (uses server-side PDF generation)
async function downloadPDF() {
  const btn = document.getElementById('downloadPdfBtn');
  const originalText = btn.textContent;
  
  try {
    btn.textContent = "Generating PDF...";
    btn.disabled = true;
    
    // Trigger server-side PDF generation
    const pdfUrl = `/api/scan/${scanId}/pdf`;
    const response = await fetch(pdfUrl);
    
    if (!response.ok) {
      throw new Error(`Failed to generate PDF: ${response.statusText}`);
    }
    
    // Download the PDF
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `patient-flow-report-${scanId}.pdf`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    a.remove();
    
    btn.textContent = "PDF Downloaded!";
    setTimeout(() => {
      btn.textContent = originalText;
      btn.disabled = false;
    }, 2000);
  } catch (error) {
    console.error('PDF generation failed:', error);
    btn.textContent = "PDF failed - try browser print";
    setTimeout(() => {
      btn.textContent = originalText;
      btn.disabled = false;
    }, 3000);
    // Fallback to browser print dialog
    window.print();
  }
}

const KEY_LABELS = {
  "home_desktop_top": "Desktop (top)",
  "home_mobile_top": "Mobile (top)",
  "home_mobile_mid": "Mobile (mid)",
  "home_mobile_bottom": "Mobile (bottom)"
};

let currentImages = [];
let currentIndex = 0;

function esc(s){ return (""+s).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }

function isValidUrl(u) {
  if (!u) return false;
  const s = ("" + u).trim();
  if (!s) return false;
  return s.startsWith("/artifacts/") || s.startsWith("http://") || s.startsWith("https://") || s.startsWith("data:");
}

// Quick wins sorting constants
const IMPACT_ORDER = { 'High': 3, 'Med': 2, 'Low': 1 };
const EFFORT_ORDER = { 'Low': 3, 'Med': 2, 'High': 1 };

// Normalize impact/effort values to handle case variations
function normalizeValue(value, defaultValue, validValues) {
  if (!value) return defaultValue;
  const normalized = value.charAt(0).toUpperCase() + value.slice(1).toLowerCase();
  return validValues.includes(normalized) ? normalized : defaultValue;
}

function setList(id, items) {
  const el = document.getElementById(id);
  const arr = Array.isArray(items) ? items : [];
  if (!arr.length) {
    el.innerHTML = `<li>
      <div class="itemIcon">—</div>
      <div class="itemContent">
        <div class="itemWhy">None listed.</div>
      </div>
    </li>`;
    return;
  }
  
  // Special handling for quick wins with checkboxes and chips
  if (id === 'quickWins') {
    const sortedItems = [...arr].sort((a, b) => {
      // Handle both old string format and new object format
      if (typeof a === 'string' || typeof b === 'string') return 0;
      
      // Use 1 (lowest priority) as fallback for invalid values
      const impactDiff = (IMPACT_ORDER[a.impact] || 1) - (IMPACT_ORDER[b.impact] || 1);
      if (impactDiff !== 0) return -impactDiff; // High impact first
      
      const effortDiff = (EFFORT_ORDER[a.effort] || 1) - (EFFORT_ORDER[b.effort] || 1);
      return -effortDiff; // Low effort first
    });
    
    el.innerHTML = sortedItems.slice(0, 12).map((item, idx) => {
      if (typeof item === 'string') {
        // Legacy format: plain string (fallback)
        return `<li>
          <div class="itemIcon">⚡</div>
          <div class="itemContent">
            <div class="itemWhy">${esc(item)}</div>
          </div>
        </li>`;
      } else if (item && typeof item === 'object' && item.action) {
        // New structured format with checkbox and chips
        // Normalize values to handle case variations and ensure valid CSS classes
        const impact = normalizeValue(item.impact, 'Med', ['High', 'Med', 'Low']);
        const effort = normalizeValue(item.effort, 'Med', ['Low', 'Med', 'High']);
        
        const impactClass = `chipImpact${impact}`;
        const effortClass = `chipEffort${effort}`;
        
        return `<li>
          <div class="quickWinCheckbox" data-index="${idx}" role="checkbox" aria-checked="false" aria-label="Mark ${esc(item.action)} as complete" tabindex="0"></div>
          <div class="itemContent">
            <div class="quickWinAction">${esc(item.action)}</div>
            <div class="quickWinChips">
              <span class="chip ${impactClass}">Impact: ${esc(impact)}</span>
              <span class="chip ${effortClass}">Effort: ${esc(effort)}</span>
            </div>
          </div>
        </li>`;
      }
      return '';
    }).join("");
    
    // Add click event listeners to checkboxes
    el.querySelectorAll('.quickWinCheckbox').forEach(checkbox => {
      const toggleCheck = () => {
        const isChecked = checkbox.classList.toggle('checked');
        checkbox.setAttribute('aria-checked', isChecked.toString());
      };
      
      checkbox.addEventListener('click', toggleCheck);
      checkbox.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          toggleCheck();
        }
      });
    });
    
    return;
  }
  
  // Icon mappings for other sections (strengths, leaks)
  const icons = {
    'strengths': '✓',
    'leaks': '⚠',
    'quickWins': '⚡'
  };
  const icon = icons[id] || '•';
  
  el.innerHTML = arr.slice(0, 12).map(item => {
    // Handle both string items (old format) and structured items (new format)
    if (typeof item === 'string') {
      // Legacy format: plain string
      return `<li>
        <div class="itemIcon">${icon}</div>
        <div class="itemContent">
          <div class="itemWhy">${esc(item)}</div>
        </div>
      </li>`;
    } else if (item && typeof item === 'object') {
      // New structured format
      const title = item.title ? `<div class="itemTitle">${esc(item.title)}</div>` : '';
      const why = item.why ? `<div class="itemWhy">${esc(item.why)}</div>` : '';
      const evidence = item.evidence ? 
        `<a href="#${esc(item.evidence)}" class="itemEvidence" data-anchor="${esc(item.evidence)}">
          <span>📸</span>
          <span>View evidence</span>
        </a>` : '';
      
      return `<li>
        <div class="itemIcon">${icon}</div>
        <div class="itemContent">
          ${title}
          ${why}
          ${evidence}
        </div>
      </li>`;
    }
    return '';
  }).join("");
  
  // Add event listeners for evidence links (using event delegation after DOM update)
  el.querySelectorAll('.itemEvidence').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const anchor = link.getAttribute('data-anchor');
      if (anchor) {
        scrollToScreenshot(anchor);
      }
    });
  });
}

function renderBars(scores) {
  const wrap = document.getElementById("scoreBars");
  wrap.innerHTML = "";
  const order = [
    ["clarity_first_impression","Clarity & first impression", "Is it obvious what you do?"],
    ["booking_path","Booking path", "How easy to book/consult?"],
    ["mobile_experience","Mobile experience", "Usable on phone?"],
    ["trust_and_proof","Trust & proof", "Credentials & reviews visible?"],
    ["treatments_and_offer","Treatments & offer", "Clear services & outcomes?"],
    ["tech_basics","Tech basics", "Fast, modern, functional?"],
  ];
  for (const [k, label, helper] of order) {
    const v = Number((scores||{})[k] ?? 0);
    const pct = Math.max(0, Math.min(100, (v/10)*100));
    const card = document.createElement("div");
    card.className = "scoreCard";
    card.innerHTML = `
      <div class="scoreCardTop">
        <div class="scoreCardLabel">${esc(label)}</div>
        <div class="scoreCardValue">${v}/10</div>
      </div>
      <div class="scoreCardHelper">${esc(helper)}</div>
      <div class="scoreCardTrack"><div class="scoreCardFill" style="width:${pct}%"></div></div>
    `;
    wrap.appendChild(card);
  }
}

function renderMiniBars(scores) {
  const wrap = document.getElementById("miniScoreBars");
  wrap.innerHTML = "";
  const order = [
    ["clarity_first_impression","Clarity"],
    ["booking_path","Booking"],
    ["mobile_experience","Mobile"],
    ["trust_and_proof","Trust"],
    ["treatments_and_offer","Treatments"],
    ["tech_basics","Tech"],
  ];
  for (const [k, label] of order) {
    const v = Number((scores||{})[k] ?? 0);
    const pct = Math.max(0, Math.min(100, (v/10)*100));
    const row = document.createElement("div");
    row.className = "miniBarRow";
    row.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span class="miniBarLabel">${esc(label)}</span>
        <span class="miniScore">${v}</span>
      </div>
      <div class="miniTrack"><div class="miniFill" style="width:${pct}%"></div></div>
    `;
    wrap.appendChild(row);
  }
}

function openModal(images, index) {
  currentImages = images;
  currentIndex = index;
  const modal = document.getElementById('modal');
  const img = document.getElementById('modalImg');
  const label = document.getElementById('modalLabel');
  
  img.src = currentImages[currentIndex].url;
  label.textContent = currentImages[currentIndex].label;
  modal.classList.add('active');
  document.body.style.overflow = 'hidden';
}

function closeModal() {
  const modal = document.getElementById('modal');
  modal.classList.remove('active');
  document.body.style.overflow = '';
}

function showPrev() {
  if (currentImages.length === 0) return;
  currentIndex = (currentIndex - 1 + currentImages.length) % currentImages.length;
  const img = document.getElementById('modalImg');
  const label = document.getElementById('modalLabel');
  img.src = currentImages[currentIndex].url;
  label.textContent = currentImages[currentIndex].label;
}

function showNext() {
  if (currentImages.length === 0) return;
  currentIndex = (currentIndex + 1) % currentImages.length;
  const img = document.getElementById('modalImg');
  const label = document.getElementById('modalLabel');
  img.src = currentImages[currentIndex].url;
  label.textContent = currentImages[currentIndex].label;
}

function renderImages(manifest, fallbackUrls) {
  const desktopGrid = document.getElementById('desktopGrid');
  const mobileGrid = document.getElementById('mobileGrid');
  desktopGrid.innerHTML = "";
  mobileGrid.innerHTML = "";
  
  const items = (manifest && manifest.length) ? manifest : (fallbackUrls||[]).map(u => ({url:u, key:""}));
  
  const desktopImages = [];
  const mobileImages = [];
  const seen = new Set();
  
  for (const it of items) {
    const u = (it && it.url) ? ("" + it.url).trim() : "";
    if (!isValidUrl(u)) continue;
    if (seen.has(u)) continue;
    seen.add(u);
    
    const label = KEY_LABELS[it.key] || (it.key ? it.key : "Screenshot");
    const imgData = { url: u, key: it.key, label: label };
    
    if (it.key && it.key.startsWith('home_desktop')) {
      desktopImages.push(imgData);
    } else if (it.key && it.key.startsWith('home_mobile')) {
      mobileImages.push(imgData);
    }
  }
  
  // Render desktop thumbnails
  desktopImages.forEach((imgData, idx) => {
    const thumb = document.createElement('div');
    thumb.className = 'thumb';
    thumb.onclick = () => openModal(desktopImages, idx);
    
    const img = document.createElement('img');
    img.src = imgData.url;
    img.loading = 'lazy';
    img.alt = imgData.label;
    img.onerror = () => { try { thumb.remove(); } catch(e) {} };
    
    const thumbLabel = document.createElement('div');
    thumbLabel.className = 'thumbLabel';
    thumbLabel.textContent = imgData.label;
    
    thumb.appendChild(img);
    thumb.appendChild(thumbLabel);
    desktopGrid.appendChild(thumb);
  });
  
  // Render mobile thumbnails
  mobileImages.forEach((imgData, idx) => {
    const thumb = document.createElement('div');
    thumb.className = 'thumb';
    thumb.onclick = () => openModal(mobileImages, idx);
    
    const img = document.createElement('img');
    img.src = imgData.url;
    img.loading = 'lazy';
    img.alt = imgData.label;
    img.onerror = () => { try { thumb.remove(); } catch(e) {} };
    
    const thumbLabel = document.createElement('div');
    thumbLabel.className = 'thumbLabel';
    thumbLabel.textContent = imgData.label;
    
    thumb.appendChild(img);
    thumb.appendChild(thumbLabel);
    mobileGrid.appendChild(thumb);
  });
}

function renderErrors(errs) {
  const wrap = document.getElementById('errs');
  wrap.innerHTML = "";
  const list = Array.isArray(errs) ? errs : [];
  for (const e of list) {
    const div = document.createElement('div');
    div.className = "err";
    div.textContent = "Screenshot failed: " + (e.key || "unknown") + " (" + (e.reason || "unknown") + ")";
    wrap.appendChild(div);
  }
}

function statusNice(s) {
  const m = (s || "").toLowerCase();
  if (m === "queued") return "Queued...";
  if (m === "running") return "Capturing site...";
  if (m === "scoring") return "Scoring...";
  if (m === "done") return "Done";
  if (m === "error") return "Error";
  return s || "Loading...";
}

async function copyJsonNow() {
  const text = document.getElementById('out')?.textContent || '';
  const btn = document.getElementById('copyJson');
  try {
    await navigator.clipboard.writeText(text);
    if (btn) btn.textContent = "Copied";
    setTimeout(() => { if (btn) btn.textContent = "Copy JSON"; }, 1200);
  } catch (e) {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
    if (btn) btn.textContent = "Copied";
    setTimeout(() => { if (btn) btn.textContent = "Copy JSON"; }, 1200);
  }
}

const copyBtn = document.getElementById('copyJson');
if (copyBtn) {
  copyBtn.addEventListener('click', (ev) => { ev.preventDefault(); ev.stopPropagation(); copyJsonNow(); });
  copyBtn.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); ev.stopPropagation(); copyJsonNow(); }
  });
}

function scrollToScreenshot(anchor) {
  // Map anchor names to tab and potentially scroll to specific section
  const tabMap = {
    'desktop': 'desktop',
    'mobile': 'mobile',
    'mobile-top': 'mobile',
    'mobile-mid': 'mobile',
    'mobile-bottom': 'mobile'
  };
  
  const tab = tabMap[anchor] || 'desktop';
  
  // Activate the correct tab
  const tabButton = document.querySelector(`.tab[data-tab="${tab}"]`);
  if (tabButton) {
    tabButton.click();
  }
  
  // Scroll to the screenshots section
  const screenshotsSection = document.querySelector('.screenshotsCard');
  if (screenshotsSection) {
    screenshotsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

if (copyBtn) {
  copyBtn.addEventListener('click', (ev) => { ev.preventDefault(); ev.stopPropagation(); copyJsonNow(); });
  copyBtn.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); ev.stopPropagation(); copyJsonNow(); }
  });
}

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    const targetTab = tab.dataset.tab;
    
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.tabContent').forEach(tc => tc.classList.remove('active'));
    document.getElementById(targetTab + '-tab').classList.add('active');
  });
});

// Modal controls
document.getElementById('modalClose').addEventListener('click', closeModal);
document.getElementById('modalPrev').addEventListener('click', showPrev);
document.getElementById('modalNext').addEventListener('click', showNext);

// Modal background click to close
document.getElementById('modal').addEventListener('click', (e) => {
  if (e.target.id === 'modal') closeModal();
});

// Keyboard navigation for modal
document.addEventListener('keydown', (e) => {
  const modal = document.getElementById('modal');
  if (!modal.classList.contains('active')) return;
  
  if (e.key === 'Escape') {
    closeModal();
  } else if (e.key === 'ArrowLeft') {
    showPrev();
  } else if (e.key === 'ArrowRight') {
    showNext();
  }
});

// Add event listener for blueprint CTA
const blueprintCTA = document.getElementById('blueprintCTA');
if (blueprintCTA) {
  blueprintCTA.addEventListener('click', () => {
    logEvent('get_blueprint_clicked');
  });
}

async function tick() {
  const res = await fetch('/api/scan/' + scanId);
  const data = await res.json();

  const st = data.status || "loading";
  const err = data.error ? (" | " + data.error) : "";
  document.getElementById('status').textContent = "Status: " + statusNice(st) + err;

  const ev = data.evidence || {};
  const hd = (ev.home_desktop || {});
  const hm = (ev.home_mobile || {});

  const manifest = []
    .concat(hd.screenshot_manifest || [])
    .concat(hm.screenshot_manifest || []);

  const fallbackUrls = []
    .concat(hd.screenshot_urls || [])
    .concat(hm.screenshot_urls || []);

  renderImages(manifest, fallbackUrls);

  const errs = []
    .concat(hd.screenshot_errors || [])
    .concat(hm.screenshot_errors || []);
  renderErrors(errs);

  const score = data.score || null;
  if (score) {
    const clinic = score.clinic_name || "Patient-Flow Report";
    document.getElementById("clinicName").textContent = clinic;
    document.getElementById("clinicNameCrumb").textContent = clinic;

    const url = score.url || data.url || "";
    const a = document.getElementById("siteUrl");
    if (url) { a.href = url; a.textContent = url.replace(/^https?:\/\//,""); }

    document.getElementById("band").textContent = score.band || "—";
    
    // Update ScoreboardCard
    const MAX_SCORE = 60;
    const total60 = score.total_score_60 ?? 0;
    const score10 = score.patient_flow_score_10 ?? 0;
    
    document.getElementById("score10Main").textContent = (score10 === 0 ? "--" : score10);
    document.getElementById("score60Main").textContent = (total60 === 0 ? "--" : total60);
    document.getElementById("verdictChip").textContent = score.band || "—";
    
    // Update the radial progress ring
    const ringEl = document.getElementById("scoreRing");
    if (ringEl) {
      const progressPct = (total60 / MAX_SCORE) * 100;
      ringEl.style.setProperty('--progress', progressPct + '%');
    }

    renderBars(score.scores || {});
    renderMiniBars(score.scores || {});
    setList("strengths", score.strengths || []);
    setList("leaks", score.leaks || []);
    setList("quickWins", score.quick_wins || []);
  }

  if (debug) {
    document.getElementById('out').textContent = JSON.stringify(score || data, null, 2);
  }

  if (st === 'queued' || st === 'running' || st === 'scoring') {
    setTimeout(tick, 1200);
  }
}
tick();
</script>
</body>
</html>
""".strip()


@app.get("/report/{slug_and_id}", response_class=HTMLResponse)
def report_page_public(slug_and_id: str, request: Request):
    """Public report route with pretty URLs.
    
    Accepts URLs like:
    - /report/aurora-aesthetic-clinic-a1b2c3d4
    - /report/scanning-a1b2c3d4
    - /report/a1b2c3d4 (just the public_id)
    """
    # Parse public_id from the slug_and_id
    # If there's a dash, public_id is everything after the last dash
    # If no dash, the whole thing is the public_id
    if '-' in slug_and_id:
        public_id = slug_and_id.split('-')[-1]
    else:
        public_id = slug_and_id
    
    # Query database by public_id
    conn = db()
    row = conn.execute("SELECT * FROM scans WHERE public_id=?", (public_id,)).fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Report not found")
    
    scan_id = row["id"]
    db_slug = row["slug"]
    
    # Build canonical path
    if db_slug:
        canonical_slug = db_slug
    else:
        canonical_slug = "scanning"
    
    canonical_path = f"/report/{canonical_slug}-{public_id}"
    
    # Check if the requested path matches canonical
    requested_path = f"/report/{slug_and_id}"
    
    if requested_path != canonical_path:
        # Build redirect URL with query string if present
        query_string = str(request.url.query)
        if query_string:
            redirect_url = f"{canonical_path}?{query_string}"
        else:
            redirect_url = canonical_path
        return RedirectResponse(url=redirect_url, status_code=301)
    
    # Build absolute URLs using request info
    # Respect X-Forwarded-Proto if present (for reverse proxies)
    forwarded_proto = request.headers.get("x-forwarded-proto")
    if forwarded_proto:
        scheme = forwarded_proto
    else:
        scheme = request.url.scheme
    
    host = request.headers.get("host") or str(request.url.netloc)
    base_url = f"{scheme}://{host}"
    canonical_url = f"{base_url}{canonical_path}"
    
    # Parse score_json and evidence_json to compute SEO values
    score_json = row["score_json"]
    evidence_json = row["evidence_json"]
    
    if score_json:
        score = json.loads(score_json)
        clinic_name = score.get("clinic_name", "").strip()
        patient_flow_score = score.get("patient_flow_score_10", 0)
        band = score.get("band", "").strip()
        
        if clinic_name:
            page_title = f"{clinic_name} - Patient-Flow Report"
            og_title = f"{clinic_name} - Patient-Flow Score: {patient_flow_score}/10"
            og_desc = f"{band}. Detailed patient-flow analysis for {clinic_name}."
        else:
            page_title = "Patient-Flow Report"
            og_title = "Patient-Flow Report"
            og_desc = f"Patient-Flow Score: {patient_flow_score}/10. {band}"
    else:
        # Scan in progress - use hostname
        url_str = row["url"]
        parsed_url = urlparse(url_str)
        hostname = parsed_url.netloc.replace("www.", "")
        
        page_title = f"{hostname} - Scan in progress"
        og_title = f"{hostname} - Patient-Flow Scan"
        og_desc = "Scan in progress. Check back soon for detailed patient-flow analysis."
    
    # Determine og:image
    og_image = f"{base_url}/android-chrome-512x512.png"  # Default fallback
    if evidence_json:
        evidence = json.loads(evidence_json)
        home_desktop = evidence.get("home_desktop", {})
        screenshot_urls = home_desktop.get("screenshot_urls", [])
        if screenshot_urls and len(screenshot_urls) > 0:
            # Use first desktop screenshot
            screenshot_url = screenshot_urls[0]
            if screenshot_url.startswith("/"):
                og_image = f"{base_url}{screenshot_url}"
            else:
                og_image = screenshot_url
    
    # Render the report HTML using the internal scan_id
    html = REPORT_HTML_TEMPLATE
    html = html.replace("__SCAN_ID__", scan_id)
    html = html.replace("__APP_NAME__", APP_NAME)
    html = html.replace("__PRODUCT__", APP_PRODUCT)
    html = html.replace("__MODE__", SCORING_MODE)
    html = html.replace("__CTA_TEXT__", PRIMARY_CTA_TEXT)
    html = html.replace("__CTA_URL__", PRIMARY_CTA_URL)
    
    # Replace SEO placeholders
    html = html.replace("__PAGE_TITLE__", page_title)
    html = html.replace("__CANONICAL_URL__", canonical_url)
    html = html.replace("__OG_TITLE__", og_title)
    html = html.replace("__OG_DESC__", og_desc)
    html = html.replace("__OG_URL__", canonical_url)
    html = html.replace("__OG_IMAGE__", og_image)
    
    return html


@app.get("/r/{scan_id}", response_class=HTMLResponse)
def report_page(scan_id: str, request: Request):
    """Legacy report route - redirects to new pretty URL format."""
    # Look up scan by internal id
    conn = db()
    row = conn.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    public_id = row["public_id"]
    
    # If scan has public_id, redirect to new format
    if public_id:
        db_slug = row["slug"]
        if db_slug:
            canonical_slug = db_slug
        else:
            canonical_slug = "scanning"  # Use "scanning" for consistency
        
        redirect_path = f"/report/{canonical_slug}-{public_id}"
        
        # Preserve query params
        query_string = str(request.url.query)
        if query_string:
            redirect_url = f"{redirect_path}?{query_string}"
        else:
            redirect_url = redirect_path
        
        return RedirectResponse(url=redirect_url, status_code=301)
    
    # Legacy scan without public_id - generate one and save it
    # This ensures all scans eventually have public_ids
    public_id_new = uuid.uuid4().hex[:8]
    conn = db()
    try:
        # Check uniqueness and retry if needed
        for attempt in range(MAX_PUBLIC_ID_ATTEMPTS):
            existing = conn.execute("SELECT id FROM scans WHERE public_id=?", (public_id_new,)).fetchone()
            if not existing:
                break
            if attempt < MAX_PUBLIC_ID_ATTEMPTS - 1:
                public_id_new = uuid.uuid4().hex[:8]
            else:
                # All attempts failed
                raise HTTPException(status_code=500, detail="Failed to generate unique public_id for legacy scan")
        
        # Update the scan with the new public_id
        conn.execute("UPDATE scans SET public_id=? WHERE id=?", (public_id_new, scan_id))
        conn.commit()
    finally:
        conn.close()
    
    # Redirect to new format
    redirect_path = f"/report/scanning-{public_id_new}"
    query_string = str(request.url.query)
    if query_string:
        redirect_url = f"{redirect_path}?{query_string}"
    else:
        redirect_url = redirect_path
    
    return RedirectResponse(url=redirect_url, status_code=301)



@app.get("/api/scan/{scan_id}/pdf")
def generate_pdf(scan_id: str):
    """Generate a PDF of the scan report using Playwright."""
    conn = db()
    row = conn.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    if row["status"] != "done":
        raise HTTPException(status_code=400, detail="Scan not complete yet")
    
    # Generate PDF using Playwright
    pdf_path = ARTIFACTS_DIR / scan_id / f"report_{scan_id}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use localhost for PDF generation (server is always local to itself)
    # Note: This assumes the server is running on port 8000. For production,
    # consider using an environment variable for the base URL.
    report_url = f"http://127.0.0.1:8000/r/{scan_id}?print=1"
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Navigate to the report page with a reasonable timeout
            page.goto(report_url, wait_until="networkidle", timeout=20000)
            
            # Wait for the main score element to ensure content is loaded
            page.wait_for_selector("#score60Main", timeout=10000)
            
            # Wait for any remaining dynamic content
            page.wait_for_load_state("networkidle", timeout=5000)
            
            # Generate PDF with specific options
            page.pdf(
                path=str(pdf_path),
                format="A4",
                print_background=True,
                display_header_footer=False,
                margin={
                    "top": "12mm",
                    "right": "12mm",
                    "bottom": "12mm",
                    "left": "12mm"
                }
            )
            
            browser.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")
    
    # Return the PDF file
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"patient-flow-report-{scan_id}.pdf"
    )


if __name__ == "__main__":
    print("Run with: python -m uvicorn api:app --reload")
