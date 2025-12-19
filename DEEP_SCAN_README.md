# Deep Scan Data Model

## Overview

The deep scan feature extends the quick scan by analyzing up to 10 pages per website, providing comprehensive per-page critiques, a patient journey map, prioritized action plan, and a 90-day roadmap.

## Database Schema

### `scan_pages` Table

Stores individual page analyses for deep scans.

**Columns:**
- `id` (INTEGER PRIMARY KEY): Auto-incrementing ID
- `scan_id` (TEXT): Foreign key to `scans.id`
- `url` (TEXT): Page URL
- `title` (TEXT): Page title
- `page_type` (TEXT): Classified page type (see Page Types below)
- `is_mobile` (INTEGER): 0 for desktop, 1 for mobile
- `screenshot_paths` (TEXT): JSON object with desktop and mobile screenshot paths
- `html_path` (TEXT): Optional path to saved HTML file
- `extracted_signals` (TEXT): JSON object with extracted page signals
- `analysis` (TEXT): JSON object with AI-generated page analysis
- `created_at` (TEXT): ISO timestamp

**Example `extracted_signals` JSON:**
```json
{
  "url": "https://example.com/services/botox",
  "page_type": "service_detail",
  "title": "Botox Treatments",
  "h1": "Expert Botox in Beverly Hills",
  "nav_labels": ["Home", "Services", "About", "Contact"],
  "cta_buttons": [
    {"text": "Book Consultation", "href": "/book"}
  ],
  "tel_links": ["tel:+13105551234"],
  "form_count": 1,
  "has_credentials": true
}
```

**Example `analysis` JSON:**
```json
{
  "page_type": "service_detail",
  "summary": "Service page with clear pricing and CTA but weak proof elements",
  "strengths": [
    "Clear service description with benefits",
    "Prominent booking CTA above fold"
  ],
  "leaks": [
    "No before/after photos",
    "Missing patient testimonials",
    "Price not mentioned anywhere"
  ],
  "quick_wins": [
    {
      "title": "Add before/after gallery",
      "why": "Builds trust and helps patients visualize results",
      "how": "Add 3-5 before/after photos with captions",
      "impact": "HIGH",
      "effort": "LOW"
    }
  ],
  "notes": {
    "cta": "Good: Clear 'Book Now' button in hero",
    "trust": "Weak: No proof elements visible",
    "mobile": "Good: Responsive design works well"
  }
}
```

### `scan_summaries` Table

Stores the synthesized deep scan report aggregating all page analyses.

**Columns:**
- `id` (INTEGER PRIMARY KEY): Auto-incrementing ID
- `scan_id` (TEXT): Foreign key to `scans.id` (UNIQUE)
- `executive_summary` (TEXT): JSON array of key findings
- `journey_map` (TEXT): JSON array of journey steps
- `action_plan` (TEXT): JSON array of prioritized action items
- `roadmap_90d` (TEXT): JSON array of roadmap phases
- `coverage` (TEXT): JSON object with page coverage report
- `created_at` (TEXT): ISO timestamp

**Example `executive_summary` JSON:**
```json
[
  "Strong homepage clarity but booking path requires 3+ clicks",
  "Excellent doctor credentials and proof throughout",
  "Mobile experience has overflow issues on service pages",
  "Missing pricing transparency - major conversion blocker",
  "Good content quality but inconsistent CTAs across pages"
]
```

**Example `journey_map` JSON:**
```json
[
  {
    "step": "Land",
    "pages": ["https://example.com/"],
    "friction": ["Slow page load (4.2s)", "Unclear value proposition"],
    "fixes": ["Optimize images", "Rewrite hero headline"],
    "risk": "MED"
  },
  {
    "step": "Trust",
    "pages": ["https://example.com/reviews", "https://example.com/about"],
    "friction": ["Reviews not prominently displayed", "No video testimonials"],
    "fixes": ["Move reviews to homepage", "Add doctor intro video"],
    "risk": "HIGH"
  }
]
```

**Example `action_plan` JSON:**
```json
[
  {
    "rank": 1,
    "title": "Add prominent booking CTA to all service pages",
    "description": "Currently 60% of service pages lack a clear booking path",
    "impact": "HIGH",
    "effort": "LOW",
    "page_references": [
      "https://example.com/services/botox",
      "https://example.com/services/fillers"
    ]
  }
]
```

**Example `roadmap_90d` JSON:**
```json
[
  {
    "phase": "Week 1-2",
    "focus": "Quick wins and critical fixes",
    "tasks": [
      "Add booking CTAs to service pages",
      "Fix mobile overflow issues",
      "Add phone number to header"
    ]
  },
  {
    "phase": "Week 3-6",
    "focus": "Trust and proof building",
    "tasks": [
      "Create before/after gallery",
      "Add video testimonials",
      "Improve doctor bio page"
    ]
  }
]
```

## Page Types

Deep scan classifies pages into funnel-critical types with the following priority order:

1. **home**: Homepage (always included)
2. **booking_consult**: Booking/consultation pages
3. **services_index**: Main services listing page
4. **service_detail**: Individual treatment/service pages (2-5 selected)
5. **results_gallery**: Before/after galleries
6. **reviews**: Reviews/testimonials pages
7. **pricing_financing**: Pricing and financing pages
8. **about_doctor**: About doctor/team pages
9. **contact_locations**: Contact and location pages
10. **form_step**: Multi-step form pages

Lower priority types (not targeted but may be included):
- **blog**: Blog/news pages
- **policy**: Privacy/terms pages (usually excluded)
- **other**: Uncategorized pages

## Page Selection Strategy

The deep scan discovers and selects up to 10 pages using this strategy:

1. **Always include home page** (seed URL)
2. **Classify all internal links** found in header, footer, and body
3. **Score pages** based on:
   - Page type priority
   - Location (header > footer > body)
   - CTA presence
   - URL keywords
4. **Select balanced mix**:
   - 1 booking page (if available)
   - 1 services index (if available)
   - 2-5 service detail pages (diversity preferred)
   - 1 results gallery (if available)
   - 1 reviews page (if available)
   - 1 pricing page (if available)
   - 1 about doctor page (if available)
   - 1 contact/locations page (if available)
5. **Fill remaining slots** with highest-scoring pages
6. **Exclude** blog, news, and policy pages unless no other candidates

## AI Analysis Flow

### Per-Page Analysis

For each selected page:

1. **Capture page** with desktop + mobile screenshots
2. **Extract signals**: CTAs, nav labels, phone/email, forms, proof elements
3. **Trim HTML**: Remove scripts/styles, keep headings/nav/buttons/forms/text (~6K chars)
4. **AI analysis**: Pass URL, page type, HTML, screenshots, signals to OpenAI
5. **Store results**: Save analysis to `scan_pages` table

### Synthesis

After all pages are analyzed:

1. **Aggregate findings** from all page analyses
2. **AI synthesis**: Generate executive summary, journey map, action plan, roadmap
3. **Store results**: Save synthesis to `scan_summaries` table

## API Response

The `/api/scan/{scan_id}` endpoint returns deep scan data in the response:

```json
{
  "id": "abc123",
  "url": "https://example.com",
  "status": "done",
  "score": { ... },
  "evidence": { ... },
  "entitlements": {
    "deep": true
  },
  "deep_scan": {
    "pages": [
      {
        "id": 1,
        "url": "https://example.com/",
        "title": "Homepage",
        "page_type": "home",
        "screenshot_paths": { ... },
        "extracted_signals": { ... },
        "analysis": { ... }
      }
    ],
    "synthesis": {
      "executive_summary": [ ... ],
      "journey_map": [ ... ],
      "action_plan": [ ... ],
      "roadmap_90d": [ ... ],
      "coverage": { ... }
    }
  }
}
```

## HTML Trimming

To avoid AI token blowups, HTML is trimmed to ~6K characters while preserving:
- Page title
- Headings (h1, h2, h3)
- Navigation links
- Buttons and CTAs
- Form elements
- Body text sample (~2K chars)

Scripts, styles, SVGs, iframes, and comments are removed.

## Error Handling

- If a page fails to capture, it's recorded with an error and skipped in analysis
- **Page Analysis Retry Logic**: Each page gets up to `MAX_RETRIES_PER_PAGE` (default: 2) attempts for AI analysis
  - If `analyze_single_page()` fails or returns an error, the page is retried
  - On success (no error field), the retry loop exits immediately
  - After all retries exhausted, an error row is stored for that page
  - The scan continues to the next page (no infinite loops on a single page)
- **Success Tracking**: Based on actual analysis results, not just "loop completed"
  - Success: `analysis` exists and has no `error` field
  - Failure: `analysis` has an `error` field or exception occurred on all attempts
- **Summary Logging**: After all pages processed:
  - "✓ All X pages succeeded" only printed if `successful_page_analyses == total_pages`
  - Otherwise shows detailed summary: "X/Y succeeded, Z/Y failed" with error messages
- If synthesis fails, partial results are stored with error flag
- Scan still completes and shows what data is available

## Future Enhancements

Potential improvements:
- Save full HTML to disk for later re-analysis
- Support re-running analysis without re-capturing pages
- Add page-to-page comparison analysis
- Track changes over time with historical scans
- Add custom page type definitions per vertical
