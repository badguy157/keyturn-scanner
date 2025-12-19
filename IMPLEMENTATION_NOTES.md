# Deep Scan Implementation Notes

## Overview

This implementation upgrades the "Deep Scan" feature from a basic multi-page screenshot capture into a comprehensive, AI-powered multi-page analysis system with synthesized insights and structured reporting.

## What Was Built

### 1. Database Schema Extensions

**New Tables:**
- `scan_pages`: Stores individual page analyses with extracted signals and AI-generated insights
- `scan_summaries`: Stores synthesized reports with executive summary, journey map, action plan, and roadmap

**Schema Features:**
- Foreign key relationships to `scans` table
- JSON columns for structured data storage
- Indexes for efficient querying
- Created via automatic migration in `init_db()`

### 2. Smart Page Selection

**Page Classification System:**
- 10+ page types: home, booking_consult, services_index, service_detail, results_gallery, reviews, pricing_financing, about_doctor, contact_locations, form_step, etc.
- Classification based on URL patterns, anchor text, and H1 content
- Priority scoring system (booking > services > reviews > pricing > about)

**Selection Algorithm:**
- Always includes homepage
- Targets 1 booking page if available
- Selects 2-5 service detail pages for diversity
- Fills remaining slots with highest-priority pages
- Excludes blog/news/policy pages by default

### 3. Per-Page AI Analysis

**HTML Trimming (`trim_html_for_ai`):**
- Removes scripts, styles, SVGs, iframes
- Removes HTML comments
- Preserves headings, nav, buttons, forms, CTAs
- Trims to ~6K characters to avoid AI token limits
- Fallback to text extraction if trimming fails

**Signal Extraction (`extract_page_signals`):**
- Title, H1, H2 headings
- Navigation labels
- CTA buttons with href
- Phone/email links
- Form fields and counts
- Proof elements (before/after, reviews, credentials)
- Image counts

**AI Analysis (`analyze_single_page`):**
- Uses OpenAI with structured output (Pydantic models)
- Focus areas customized per page type
- Outputs: summary, strengths, leaks, quick wins, notes
- Error handling with graceful degradation
- Fallback to JSON mode if structured output fails

### 4. Synthesis & Aggregation

**Deep Scan Synthesis (`synthesize_deep_scan`):**
- Aggregates findings from all page analyses
- Generates:
  - Executive Summary (5 key findings + "what to fix first")
  - Patient Journey Map (Land → Understand → Trust → Choose → Commit → Confirm)
    - Pages per step
    - Friction points
    - Recommended fixes
    - Risk levels
  - Action Plan (top 10 items ranked by impact/effort with page references)
  - 90-Day Roadmap (Week 1-2, Week 3-6, Week 7-12 phases)
  - Coverage Report (scanned pages + missing recommended types)

### 5. Report UI Updates

**New Report Sections:**
1. **Executive Summary** - Key findings and top priority fix
2. **Page-by-Page Analysis** - Collapsible cards per page with:
   - Page type tag
   - Summary
   - Strengths
   - Booking leaks
   - Quick wins (with impact/effort chips)
3. **Patient Journey Map** - Visual journey steps with friction and fixes
4. **Top Action Plan** - Ranked action items (1-10) with descriptions
5. **90-Day Roadmap** - Phased implementation timeline

**UI Features:**
- Collapsible page analysis cards (click to expand)
- Color-coded page type tags
- Impact/effort chips (High/Med/Low)
- Risk indicators (High/Med/Low)
- Responsive design
- Dark theme styling

### 6. API Extensions

**Extended `/api/scan/{scan_id}` Endpoint:**
```json
{
  "deep_scan": {
    "pages": [
      {
        "id": 1,
        "url": "...",
        "page_type": "home",
        "extracted_signals": {...},
        "analysis": {...}
      }
    ],
    "synthesis": {
      "executive_summary": [...],
      "journey_map": [...],
      "action_plan": [...],
      "roadmap_90d": [...],
      "coverage": {...}
    }
  }
}
```

## Integration with Existing Code

### Backward Compatibility
- Quick scan still works exactly as before
- Deep scan only activates when `mode="deep"` and `max_pages > 1`
- Existing `score_json` and `evidence_json` remain unchanged
- Report UI gracefully handles missing deep scan data

### Scan Flow Integration
The deep scan flow is integrated into `run_scan()`:

1. **Page Discovery** - `discover_site_pages()` finds and classifies pages
2. **Page Capture** - `capture_page()` screenshots and extracts HTML
3. **HTML Storage** - Raw HTML stored in evidence for analysis
4. **Per-Page Analysis** - Loop through captured pages:
   - Extract signals
   - Classify page type
   - Call AI analysis
   - Store in `scan_pages` table
5. **Synthesis** - Aggregate all page analyses:
   - Call synthesis AI
   - Store in `scan_summaries` table
6. **Backward Compatible Scoring** - Existing `analyze_pages()` still runs for compatibility

### Error Handling
- Page capture failures: recorded but don't stop scan
- Page analysis failures: error stored, synthesis continues
- Synthesis failures: partial results stored with error flag
- All errors logged for debugging

## Key Design Decisions

### Why Store in Database?
- Enables future querying and comparison
- Supports potential "re-analysis" without re-capture
- Allows historical tracking
- Structured data ready for BI/reporting

### Why Per-Page Analysis?
- More specific insights per page type
- Avoids one-size-fits-all analysis
- Better evidence tracking (which page has which issue)
- Enables page-specific recommendations

### Why Synthesis Step?
- Cross-page patterns invisible to per-page analysis
- Journey mapping requires understanding full flow
- Action plan needs prioritization across all findings
- Roadmap needs holistic view of all work

### Why HTML Trimming?
- OpenAI has token limits (~8K input tokens)
- Raw HTML often 50K+ characters
- Most HTML is noise (scripts, styles, analytics)
- Trimming preserves signal, removes noise

## Future Enhancements

### Potential Improvements
1. **HTML Persistence**: Save full HTML to disk for re-analysis
2. **Re-run Analysis**: Analyze existing captures without re-scanning
3. **Historical Comparison**: Track changes over time
4. **Custom Page Types**: Allow vertical-specific types (e.g., dentistry vs dermatology)
5. **Page Relationships**: Analyze click paths and navigation flows
6. **A/B Variant Analysis**: Compare different versions of pages
7. **Competitive Analysis**: Compare against competitor sites

### Known Limitations
1. **10-Page Cap**: Could be increased but impacts AI costs and scan time
2. **Single Analysis**: Each page analyzed once (could do multiple perspectives)
3. **No JavaScript Rendering Analysis**: Only HTML/CSS analyzed, not dynamic behavior
4. **English Only**: AI prompts and analysis assume English content
5. **Generic Journey Map**: Could be customized per medical specialty

## Testing Recommendations

### Manual Testing
1. Run deep scan on a real clinic website
2. Verify all database tables created
3. Check that page analyses are stored
4. Verify synthesis generates all sections
5. Test report UI renders all new sections
6. Try collapsing/expanding page cards
7. Verify page type tags display correctly
8. Check impact/effort chips render

### Automated Testing
1. Unit tests for page classification
2. Integration tests for scan flow
3. Mock AI responses for testing
4. Database schema validation
5. API endpoint tests

### Edge Cases to Test
1. Site with < 10 pages
2. Site with no booking page
3. Page capture failures
4. AI analysis failures
5. Missing page types
6. Very large HTML pages
7. Non-English content

## Performance Considerations

### AI Costs
- Per-page analysis: 1 AI call per page (max 10 calls)
- Synthesis: 1 AI call per scan
- Total: ~11 AI calls per deep scan
- Estimated cost: $0.10-0.50 per deep scan (depending on model)

### Scan Time
- Page capture: ~3-5 seconds per page
- AI analysis: ~5-10 seconds per page
- Synthesis: ~10-15 seconds
- Total: ~2-3 minutes for 10-page deep scan

### Optimization Opportunities
1. **Parallel AI Calls**: Analyze pages in parallel (careful with rate limits)
2. **Caching**: Cache page classifications and signals
3. **Incremental Analysis**: Only analyze changed pages on re-scan
4. **Batch Synthesis**: Generate multiple reports in batch

## Security Considerations

### CodeQL Results
- ✅ 0 security alerts detected
- All SQL uses parameterized queries
- HTML escaping in UI rendering
- Input validation on all endpoints

### Potential Risks
1. **AI Prompt Injection**: Malicious HTML could try to inject prompts (mitigated by HTML trimming)
2. **Storage Growth**: Storing HTML and analyses increases DB size (monitor and cleanup old scans)
3. **API Costs**: Deep scans cost more in AI calls (implement rate limiting)

## Maintenance

### Regular Tasks
1. Monitor database size growth
2. Review AI analysis quality
3. Update page type classifications as needed
4. Tune synthesis prompts based on feedback
5. Add new page types as patterns emerge

### Troubleshooting
- **Page classification wrong**: Update `_classify_page_type()` heuristics
- **Analysis quality poor**: Tune AI prompts in `analyze_single_page()`
- **Synthesis missing insights**: Update `synthesize_deep_scan()` prompt
- **UI not rendering**: Check browser console, verify data structure
- **Database errors**: Check schema migration in `init_db()`

## Conclusion

This implementation provides a robust foundation for comprehensive website analysis. The modular design allows for incremental improvements and customization while maintaining backward compatibility with the existing quick scan feature.

The system successfully transforms raw website data into actionable insights through a multi-stage pipeline: discovery → capture → analysis → synthesis → presentation.
