# Visual Guide: Email Delivery Features

## Flow A: Report Page - "Send full report to my inbox"

### Location
On any completed report page at `/report/{clinic-name}-{id}`, scroll down to the "Get the Blueprint" card.

### UI Elements

**Section Title:**
```
Send full report to my inbox
```

**Input Field:**
```
┌──────────────────────────────────────────────┐
│ your.email@clinic.com                        │
└──────────────────────────────────────────────┘
```

**Send Button (Normal State):**
```
┌──────────────────────────────────────────────┐
│               Send Report                     │
└──────────────────────────────────────────────┘
```

**Send Button (Loading State):**
```
┌──────────────────────────────────────────────┐
│               Sending...                      │
└──────────────────────────────────────────────┘
```

### Success Flow
1. User enters email address
2. Clicks "Send Report"
3. Button changes to "Sending..." and disables
4. Green success message appears: **"✓ Sent to user@example.com"**
5. Email input clears
6. After 3 seconds, button resets to "Send Report"
7. 15-second cooldown prevents immediate re-sending

### Error Scenarios

**Empty Email:**
```
⚠ Please enter your email address
```

**Invalid Email Format:**
```
⚠ Invalid email address
```

**Rate Limited:**
```
⚠ Too many email requests. Please try again in a few minutes.
```

**Report Not Ready:**
```
⚠ Report is not ready yet. Please wait for the scan to complete.
```

**Cooldown Active:**
```
⚠ Please wait 12 seconds before sending again
```

**Server Error:**
```
⚠ Failed to send email. Please try again.
```

### Email Received

**Subject:**
```
Your patient-flow report for Aurora Aesthetic Clinic
```

**Email Content (simplified):**
```
┌──────────────────────────────────────────────────────┐
│  Keyturn Studio                                      │
│  Patient-Flow Quick Scan                             │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Your patient-flow report for Aurora Aesthetic       │
│  Clinic                                              │
│                                                      │
│  Here's your complete patient-flow analysis for      │
│  Aurora Aesthetic Clinic.                            │
│                                                      │
│  View your complete analysis including screenshots,  │
│  scores, strengths, leaks, and actionable quick wins.│
│                                                      │
│  ┌──────────────────────────────────┐               │
│  │    View Your Report →            │               │
│  └──────────────────────────────────┘               │
│                                                      │
│  Or copy and paste this link:                        │
│  https://scan.keyturn.studio/report/aurora-abc123   │
│                                                      │
├──────────────────────────────────────────────────────┤
│  Questions or need help?                             │
│  Contact us at hello@keyturn.studio                  │
│                                                      │
│  © 2025 Keyturn Studio. All rights reserved.        │
└──────────────────────────────────────────────────────┘
```

---

## Flow B: Scan Page - Email field for automatic notification

### Location
On the main scan page at `/`, in the scan form.

### UI Elements

**Updated Label:**
```
Email (optional, we'll email you the report link)
```

**Input Field:**
```
┌──────────────────────────────────────────────┐
│ name@clinic.com                              │
└──────────────────────────────────────────────┘
```

### User Flow
1. User enters clinic URL
2. User enters email address (optional)
3. Clicks "Run free scan"
4. Scan runs for 1-2 minutes
5. **When scan completes:** Email automatically sent (if provided)
6. User redirected to report page

### Email Received After Scan

**Subject:**
```
Your report is ready: Aurora Aesthetic Clinic
```

**Email Content (simplified):**
```
┌──────────────────────────────────────────────────────┐
│  Keyturn Studio                                      │
│  Patient-Flow Quick Scan                             │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Your report is ready: Aurora Aesthetic Clinic       │
│                                                      │
│  Your patient-flow scan for Aurora Aesthetic Clinic  │
│  is complete.                                        │
│                                                      │
│  View your complete analysis including screenshots,  │
│  scores, strengths, leaks, and actionable quick wins.│
│                                                      │
│  ┌──────────────────────────────────────┐           │
│  │    View Your Report →                │           │
│  └──────────────────────────────────────┘           │
│                                                      │
│  Or copy and paste this link:                        │
│  https://scan.keyturn.studio/report/aurora-abc123   │
│                                                      │
├──────────────────────────────────────────────────────┤
│  Questions or need help?                             │
│  Contact us at hello@keyturn.studio                  │
│                                                      │
│  © 2025 Keyturn Studio. All rights reserved.        │
└──────────────────────────────────────────────────────┘
```

---

## Backend Logging

### Console Output (Success)
```
[EMAIL] Sent report to user@example.com | Message ID: msg_2b5f... | Report: abc12345
```

### Console Output (Failure)
```
[EMAIL] Failed to send scan_receipt to user@example.com | Error: Invalid API key
```

### Database Logging (email_logs table)
```sql
id | recipient              | report_id | email_type    | status | provider_message_id | error_message | created_at
---+------------------------+-----------+---------------+--------+--------------------+---------------+------------------
1  | user@example.com       | abc12345  | report        | sent   | msg_2b5f...        | NULL          | 2025-01-15T10:30:00Z
2  | test@clinic.com        | xyz789    | scan_receipt  | sent   | msg_8a3c...        | NULL          | 2025-01-15T10:35:00Z
3  | invalid@test.com       | def456    | report        | failed | NULL               | Invalid API   | 2025-01-15T10:40:00Z
```

---

## Developer Testing

### Test Endpoint (Development Only)

**Enable:**
```bash
export ENABLE_TEST_ENDPOINTS=true
```

**Test Command:**
```bash
curl "http://localhost:8000/api/email/test?to=your-email@example.com"
```

**Response:**
```json
{
  "ok": true,
  "message": "Test email sent to your-email@example.com",
  "message_id": "msg_abc123xyz"
}
```

**Test Email Received:**
```
Subject: Test Email from Keyturn Scanner

This is a test email from the Keyturn Scanner application.

If you received this, your email configuration is working correctly!

Sent at: 2025-01-15T10:30:00Z
```

---

## Rate Limiting Behavior

### Scenario: User sends 4 emails in quick succession

**Request 1:** ✅ Success - "✓ Sent to user@example.com"
**Request 2 (5 sec later):** ⚠ Cooldown - "Please wait 10 seconds before sending again"
**Request 3 (20 sec later):** ✅ Success - "✓ Sent to user@example.com"
**Request 4 (25 sec later):** ✅ Success - "✓ Sent to user@example.com"
**Request 5 (30 sec later):** ✅ Success - "✓ Sent to user@example.com"
**Request 6 (35 sec later):** ⚠ Rate Limited - "Too many email requests. Please try again in a few minutes."

The system allows up to 3 successful sends per 5-minute window per email address or IP.
