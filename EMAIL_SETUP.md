# Email Delivery Setup Guide

This guide explains how to configure and use the email delivery system for scan.keyturn.studio.

## Overview

The email system supports two flows:
1. **Report Email**: Users can send themselves a copy of the report from the report page
2. **Scan Completion Email**: Automatically sends "Your report is ready" email when scan completes (if email was provided)

## Configuration

### Required Environment Variables

```bash
# Resend API Key (required for email delivery)
RESEND_API_KEY=re_xxxxxxxxxxxxxxxxxxxxx

# Email "From" address (optional, defaults to Resend testing address)
EMAIL_FROM="Keyturn Studio <onboarding@resend.dev>"

# Public base URL for your deployment (required for correct report links)
PUBLIC_BASE_URL=https://scan.keyturn.studio

# Enable test endpoints (optional, defaults to false)
ENABLE_TEST_ENDPOINTS=false
```

### Setting Up Resend

1. Sign up for a free account at [resend.com](https://resend.com)
2. Verify your domain (or use the default `onboarding@resend.dev` for testing)
3. Create an API key from the dashboard
4. Set the `RESEND_API_KEY` environment variable

**For Production:**
```bash
export RESEND_API_KEY="re_your_production_key"
export EMAIL_FROM="Keyturn Studio <reports@keyturn.studio>"
export PUBLIC_BASE_URL="https://scan.keyturn.studio"
```

**For Development/Testing:**
```bash
export RESEND_API_KEY="re_your_test_key"
export EMAIL_FROM="Keyturn Studio <onboarding@resend.dev>"
export PUBLIC_BASE_URL="http://localhost:8000"
```

## API Endpoints

### POST /api/email_report
Send a full report email to a user.

**Request:**
```json
{
  "email": "user@example.com",
  "report_id": "abc12345"
}
```

**Response (Success):**
```json
{
  "ok": true,
  "message_id": "msg_xxxxx",
  "sent_to": "user@example.com"
}
```

**Response (Error):**
```json
{
  "detail": "Too many email requests. Please try again in a few minutes."
}
```

### POST /api/email/scan-receipt
Send a "Your report is ready" email.

Same request/response format as `/api/email_report`.

### GET /api/email/test
Development-only endpoint to test email configuration.

**Enabled by:** Set `ENABLE_TEST_ENDPOINTS=true` environment variable

**Query Parameters:**
- `to` (optional): Email address to send test email to (default: test@example.com)

**Example:**
```bash
curl "http://localhost:8000/api/email/test?to=myemail@example.com"
```

## Features

### Rate Limiting
- **Per Email**: 3 emails per 5 minutes per email address
- **Per IP**: 3 emails per 5 minutes per IP address
- Prevents abuse while allowing legitimate use

### Server-Side Logging
All email attempts are logged to the `email_logs` table with:
- Recipient email address
- Report ID
- Email type (`report` or `scan_receipt`)
- Status (`sent` or `failed`)
- Provider message ID (Resend)
- Error message (if failed)
- Timestamp

**Console logging format:**
```
[EMAIL] Sent report to user@example.com | Message ID: msg_xxxxx | Report: abc12345
[EMAIL] Failed to send scan_receipt to user@example.com | Error: Invalid API key
```

### UI Feedback
- **Loading state**: Button shows "Sending..." while email is being sent
- **Success toast**: Shows "✓ Sent to {email}" on success
- **Error toast**: Shows detailed error message from API (e.g., rate limit message)
- **Cooldown**: 15-second client-side cooldown prevents double-sends

### Email Templates
Professional HTML email templates include:
- Clinic name and report title
- Clean "View Your Report" button
- Plain URL fallback for email clients that don't render buttons
- Support contact: hello@keyturn.studio
- Responsive design for mobile and desktop

## Database Schema

### email_logs
```sql
CREATE TABLE email_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  recipient TEXT NOT NULL,
  report_id TEXT,
  email_type TEXT NOT NULL,
  status TEXT NOT NULL,
  provider_message_id TEXT,
  error_message TEXT,
  created_at TEXT NOT NULL
);
```

### email_rate_limits
```sql
CREATE TABLE email_rate_limits (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  identifier TEXT NOT NULL,
  email_type TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX idx_email_rate_limits_lookup 
ON email_rate_limits(identifier, email_type, created_at);
```

## Testing

### Unit Tests
The email service includes unit tests that verify:
- Email validation regex
- HTML email generation
- Email structure validation

To run tests, create a test script with the following content and execute it:

```python
#!/usr/bin/env python3
import sys
import os

# Set environment variables for testing
os.environ['RESEND_API_KEY'] = 'test-key-not-real'
os.environ['EMAIL_FROM'] = 'Keyturn Studio <onboarding@resend.dev>'
os.environ['PUBLIC_BASE_URL'] = 'https://scan.keyturn.studio'

# Import the functions
from api import build_email_html, EMAIL_RE

# Test email validation
valid_emails = ["test@example.com", "user.name@domain.co.uk"]
for email in valid_emails:
    assert EMAIL_RE.match(email), f"Valid email rejected: {email}"
    print(f"✓ {email}")

# Test HTML email generation
html = build_email_html("report", "Test Clinic", "https://example.com/report")
assert "Test Clinic" in html
assert "https://example.com/report" in html
print("✓ Email templates working correctly")
```

### Integration Testing
1. Set up Resend API key in environment:
   ```bash
   export RESEND_API_KEY="re_your_test_key"
   export ENABLE_TEST_ENDPOINTS=true
   ```
   
2. Start the server:
   ```bash
   python -m uvicorn api:app --reload
   ```
   
3. Test the dev endpoint:
   ```bash
   curl "http://localhost:8000/api/email/test?to=your-email@example.com"
   ```
   
4. Check your inbox for the test email

5. Run a scan with an email address and verify you receive the completion email

6. Visit a report page and test the "Send Report" button

### Monitoring Email Delivery

Query the email logs:
```sql
-- Recent emails
SELECT * FROM email_logs ORDER BY created_at DESC LIMIT 10;

-- Failed emails
SELECT * FROM email_logs WHERE status = 'failed' ORDER BY created_at DESC;

-- Email volume by type
SELECT email_type, status, COUNT(*) as count 
FROM email_logs 
GROUP BY email_type, status;
```

## Troubleshooting

### Emails not sending
1. Check that `RESEND_API_KEY` is set correctly
2. Verify the API key is valid in the Resend dashboard
3. Check server logs for error messages
4. Query `email_logs` table for error details

### Rate limiting issues
1. Check `email_rate_limits` table to see recent attempts
2. Wait 5 minutes and try again
3. Consider adjusting rate limits in `check_email_rate_limit()` function if needed

### Wrong report URLs
1. Verify `PUBLIC_BASE_URL` is set correctly
2. Check that it doesn't have a trailing slash
3. Test by sending an email and checking the link

### Email delivery delays
- Resend typically delivers emails within seconds
- Check Resend dashboard for delivery status using the `message_id`
- Verify recipient email isn't bouncing or blocking

## Security Considerations

1. **API Key Protection**: Never commit `RESEND_API_KEY` to version control
2. **Rate Limiting**: Server-side rate limiting prevents abuse
3. **Email Validation**: Both frontend and backend validate email addresses
4. **No PII Logging**: API keys are never logged to console or database
5. **Cooldown**: Client-side 15-second cooldown prevents accidental double-sends

## Production Checklist

- [ ] Set production `RESEND_API_KEY`
- [ ] Configure custom domain in Resend
- [ ] Update `EMAIL_FROM` to use your domain
- [ ] Set `PUBLIC_BASE_URL` to production URL
- [ ] Test email delivery end-to-end
- [ ] Monitor `email_logs` for delivery success
- [ ] Set up alerts for failed email attempts
- [ ] Document support process for email issues
