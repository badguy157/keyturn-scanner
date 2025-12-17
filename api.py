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
#   $env:PRIMARY_CTA_TEXT="Get the Blueprint ($1,000)"
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
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl, conint
from playwright.sync_api import sync_playwright

# Import OpenAI safely so missing module doesn't crash the whole server at import-time.
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


DB_PATH = "patientflow.sqlite"
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

APP_NAME = os.getenv("APP_NAME", "Keyturn Studio").strip() or "Keyturn Studio"
APP_PRODUCT = os.getenv("APP_PRODUCT", "Patient-Flow Quick Scan").strip() or "Patient-Flow Quick Scan"
PRIMARY_CTA_TEXT = os.getenv("PRIMARY_CTA_TEXT", "Get the Blueprint ($1,000)").strip() or "Get the Blueprint ($1,000)"
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
        conn.commit()
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
          evidence_json TEXT,
          score_json TEXT,
          error TEXT
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


# ---- Structured output model (Option A) ----
class PatientFlowScores(BaseModel):
    clarity_first_impression: conint(ge=0, le=10)
    booking_path: conint(ge=0, le=10)
    mobile_experience: conint(ge=0, le=10)
    trust_and_proof: conint(ge=0, le=10)
    treatments_and_offer: conint(ge=0, le=10)
    tech_basics: conint(ge=0, le=10)


class PatientFlowAIOutput(BaseModel):
    clinic_name: str = Field(..., description="Clinic name inferred from the site")
    scores: PatientFlowScores
    strengths: List[str] = Field(default_factory=list)
    leaks: List[str] = Field(default_factory=list)
    quick_wins: List[str] = Field(default_factory=list)


def _model_to_dict(m: Any) -> Dict[str, Any]:
    if hasattr(m, "model_dump"):
        return m.model_dump()  # pydantic v2
    if hasattr(m, "dict"):
        return m.dict()  # pydantic v1
    return dict(m)


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


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
- Write strengths/leaks/quick_wins as short, plain bullets (no essays). Be specific.

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

        conn.execute(
            "UPDATE scans SET status=?, updated_at=?, evidence_json=?, score_json=?, error=? WHERE id=?",
            ("done", now_iso(), json.dumps(evidence), json.dumps(output), None, scan_id),
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
  window.location.href = '/r/' + data.id;
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
    conn = db()
    conn.execute(
        "INSERT INTO scans (id, url, status, created_at, updated_at, email) VALUES (?, ?, ?, ?, ?, ?)",
        (scan_id, str(req.url), "queued", now_iso(), now_iso(), email),
    )
    conn.commit()
    conn.close()

    t = threading.Thread(target=run_scan, args=(scan_id, str(req.url)), daemon=True)
    t.start()

    return {"id": scan_id}


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


REPORT_HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Report __SCAN_ID__</title>
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
    .wrap{max-width:1120px; margin:0 auto; padding:22px 18px 64px;}
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
    }
    .btn{
      background:linear-gradient(135deg, rgba(122,162,255,.30), rgba(124,247,195,.16));
      box-shadow: 0 10px 30px rgba(0,0,0,.35);
    }
    .btn2{background: rgba(255,255,255,.06);}
    .panel{
      background:linear-gradient(180deg, var(--card), rgba(255,255,255,.04));
      border:1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding:18px;
    }
    .hero{
      display:grid;
      grid-template-columns: 1fr;
      gap:14px;
      align-items:stretch;
    }
    .h1{margin:0; font-size:28px; letter-spacing:-.35px}
    .meta{margin-top:8px; display:flex; gap:10px; flex-wrap:wrap; align-items:center; color:var(--muted); font-size:13px}
    .pill{
      display:inline-flex; align-items:center; gap:8px;
      padding:6px 10px; border-radius:999px;
      border:1px solid rgba(255,255,255,.14);
      background: rgba(0,0,0,.18);
      color: rgba(232,238,252,.88);
    }
    .status{margin-top:10px; color:var(--muted); font-size:13px}
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
        rgba(122,162,255,.75) 0%,
        rgba(122,162,255,.75) var(--progress, 0%),
        rgba(255,255,255,.08) var(--progress, 0%),
        rgba(255,255,255,.08) 100%
      );
      mask:radial-gradient(circle, transparent 0%, transparent 62%, black 62%, black 100%);
      -webkit-mask:radial-gradient(circle, transparent 0%, transparent 62%, black 62%, black 100%);
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
      grid-template-columns: 1fr 1fr 1fr 1fr;
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
    ul{margin:0; padding-left:18px; color:rgba(232,238,252,.84)}
    li{margin:8px 0; line-height:1.35}
    .shots{display:flex; gap:12px; flex-wrap:wrap; margin-top:12px; align-items:flex-start;}
    figure{
      margin:0;
      border-radius:16px;
      border:1px solid rgba(255,255,255,.12);
      background: rgba(255,255,255,.04);
      overflow:hidden;
      max-width:340px;
    }
    img{display:block; width:100%; height:auto; background:#fff;}
    figcaption{padding:10px 10px; font-size:12px; color:rgba(232,238,252,.72)}
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

    <div class="hero">
      <div class="panel">
        <div class="h1" id="clinicName">Patient-Flow Report</div>
        <div class="meta">
          <a id="siteUrl" href="#" target="_blank" rel="noopener" class="pill">Website</a>
          <span class="pill" id="band">Band</span>
          <span class="pill">Mode: __MODE__</span>
        </div>
        <div class="status" id="status">Loading...</div>

        <div class="shots" id="imgs"></div>
        <div id="errs"></div>
      </div>
    </div>

    <div class="grid">
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
      <div class="panel">
        <h2>Scores</h2>
        <div class="bars" id="scoreBars"></div>
      </div>
      <div class="panel">
        <h2>Strengths</h2>
        <ul id="strengths"><li>Waiting for score...</li></ul>
      </div>
      <div class="panel">
        <h2>Leaks</h2>
        <ul id="leaks"><li>Waiting for score...</li></ul>
      </div>
    </div>

    <div class="grid">
      <div class="panel" style="grid-column: 1 / -1;">
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

<script>
const scanId = "__SCAN_ID__";
const params = new URLSearchParams(window.location.search);
const debug = params.get('debug') === '1';

const rawWrap = document.getElementById('rawWrap');
if (rawWrap) rawWrap.style.display = debug ? 'block' : 'none';

const KEY_LABELS = {
  "home_desktop_top": "Desktop (top)",
  "home_mobile_top": "Mobile (top)",
  "home_mobile_mid": "Mobile (mid)",
  "home_mobile_bottom": "Mobile (bottom)"
};

function esc(s){ return (""+s).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }

function isValidUrl(u) {
  if (!u) return false;
  const s = ("" + u).trim();
  if (!s) return false;
  return s.startsWith("/artifacts/") || s.startsWith("http://") || s.startsWith("https://");
}

function setList(id, items) {
  const el = document.getElementById(id);
  const arr = Array.isArray(items) ? items : [];
  if (!arr.length) {
    el.innerHTML = "<li>None listed.</li>";
    return;
  }
  el.innerHTML = arr.slice(0, 12).map(x => "<li>" + esc(x) + "</li>").join("");
}

function renderBars(scores) {
  const wrap = document.getElementById("scoreBars");
  wrap.innerHTML = "";
  const order = [
    ["clarity_first_impression","Clarity & first impression"],
    ["booking_path","Booking path"],
    ["mobile_experience","Mobile experience"],
    ["trust_and_proof","Trust & proof"],
    ["treatments_and_offer","Treatments & offer"],
    ["tech_basics","Tech basics"],
  ];
  for (const [k, label] of order) {
    const v = Number((scores||{})[k] ?? 0);
    const pct = Math.max(0, Math.min(100, (v/10)*100));
    const row = document.createElement("div");
    row.className = "barRow";
    row.innerHTML = `
      <div class="barTop"><span>${esc(label)}</span><span>${v}/10</span></div>
      <div class="track"><div class="fill" style="width:${pct}%"></div></div>
    `;
    wrap.appendChild(row);
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

function renderImages(manifest, fallbackUrls) {
  const wrap = document.getElementById('imgs');
  wrap.innerHTML = "";
  const items = (manifest && manifest.length) ? manifest : (fallbackUrls||[]).map(u => ({url:u, key:""}));

  const seen = new Set();
  for (const it of items) {
    const u = (it && it.url) ? ("" + it.url).trim() : "";
    if (!isValidUrl(u)) continue;
    if (seen.has(u)) continue;
    seen.add(u);

    const fig = document.createElement("figure");
    const img = document.createElement('img');
    img.src = u;
    img.loading = "lazy";
    img.alt = it.key || "screenshot";
    img.onerror = () => { try { fig.remove(); } catch(e) {} };

    const cap = document.createElement("figcaption");
    cap.textContent = KEY_LABELS[it.key] || (it.key ? it.key : "Screenshot");

    fig.appendChild(img);
    fig.appendChild(cap);
    wrap.appendChild(fig);
  }
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
    if (ringEl && total60 > 0) {
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


@app.get("/r/{scan_id}", response_class=HTMLResponse)
def report_page(scan_id: str):
    html = REPORT_HTML_TEMPLATE
    html = html.replace("__SCAN_ID__", scan_id)
    html = html.replace("__APP_NAME__", APP_NAME)
    html = html.replace("__PRODUCT__", APP_PRODUCT)
    html = html.replace("__MODE__", SCORING_MODE)
    html = html.replace("__CTA_TEXT__", PRIMARY_CTA_TEXT)
    html = html.replace("__CTA_URL__", PRIMARY_CTA_URL)
    return html


if __name__ == "__main__":
    print("Run with: python -m uvicorn api:app --reload")
