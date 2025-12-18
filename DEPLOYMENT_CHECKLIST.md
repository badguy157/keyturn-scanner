# Deployment Checklist for Email Delivery

## Pre-Deployment Setup

### 1. Resend Account Setup
- [ ] Create account at https://resend.com
- [ ] Verify your domain (or use onboarding@resend.dev for testing)
- [ ] Generate API key from dashboard
- [ ] Test API key with test endpoint

### 2. Environment Variables
Set these in your production environment (e.g., Render, Heroku, etc.):

```bash
# Required
RESEND_API_KEY="re_your_production_key"
EMAIL_FROM="Keyturn Studio <reports@keyturn.studio>"
PUBLIC_BASE_URL="https://scan.keyturn.studio"

# Optional
ENABLE_TEST_ENDPOINTS="false"  # Disable in production
```

### 3. Database Migration
The app will automatically create the required tables on startup:
- email_logs
- email_rate_limits

No manual migration needed!

### 4. Dependencies
The updated requirements.txt includes:
- resend (new)

Ensure your deployment pulls latest requirements:
```bash
pip install -r requirements.txt
```

## Testing Checklist

### Local Testing
- [ ] Set test API key: `export RESEND_API_KEY="re_test_key"`
- [ ] Enable test endpoint: `export ENABLE_TEST_ENDPOINTS=true`
- [ ] Start server: `uvicorn api:app --reload`
- [ ] Test endpoint: `curl "http://localhost:8000/api/email/test?to=your@email.com"`
- [ ] Verify email received

### Production Testing
- [ ] Deploy with production API key
- [ ] Run a test scan with email address
- [ ] Verify scan completion email received
- [ ] Visit report page
- [ ] Test "Send Report" button
- [ ] Verify report email received
- [ ] Check both emails have correct links
- [ ] Verify links work and point to production URL

### Rate Limiting Test
- [ ] Send 3 emails quickly (should succeed)
- [ ] Attempt 4th email (should fail with rate limit message)
- [ ] Wait 5 minutes
- [ ] Try again (should succeed)

### Error Handling Test
- [ ] Try sending to invalid email format
- [ ] Try sending before scan completes
- [ ] Verify error messages display correctly

## Monitoring Setup

### 1. Check Email Logs
Query database regularly:
```sql
-- Recent emails
SELECT * FROM email_logs ORDER BY created_at DESC LIMIT 20;

-- Failed emails in last 24 hours
SELECT * FROM email_logs 
WHERE status = 'failed' 
  AND created_at > datetime('now', '-1 day')
ORDER BY created_at DESC;

-- Email volume
SELECT 
  email_type, 
  status, 
  COUNT(*) as count,
  DATE(created_at) as date
FROM email_logs 
GROUP BY date, email_type, status
ORDER BY date DESC;
```

### 2. Server Logs
Watch for email-related log lines:
```bash
# Success
[EMAIL] Sent report to user@example.com | Message ID: msg_xxx | Report: abc123

# Failure
[EMAIL] Failed to send scan_receipt to user@example.com | Error: Invalid API key
```

### 3. Resend Dashboard
- Monitor delivery rates
- Check bounce/spam rates
- Review message IDs for tracking

## Troubleshooting

### Emails Not Sending
1. Verify RESEND_API_KEY is set correctly
2. Check Resend dashboard for API key validity
3. Query email_logs table for error details
4. Check server logs for [EMAIL] entries

### Wrong Report URLs
1. Verify PUBLIC_BASE_URL has no trailing slash
2. Correct format: `https://scan.keyturn.studio`
3. Test by sending email and clicking link

### Rate Limiting Issues
1. Check email_rate_limits table
2. Consider adjusting limits in code if needed
3. Clear old rate limit entries manually if needed

## Success Metrics

Track these in your monitoring:
- Email delivery success rate (should be >99%)
- Average time to delivery
- Rate limit hit rate (should be low)
- Report email vs scan receipt ratio
- Bounce/spam complaint rate in Resend

## Rollback Plan

If email delivery has critical issues:
1. Set RESEND_API_KEY to empty string
2. App will continue to work but emails won't send
3. Email logs will show failures
4. Fix issue and restore API key
5. No data loss - scans continue to work

## Support

For email-related user questions:
- Email support: hello@keyturn.studio
- Check email_logs table for user's email
- Verify message_id in Resend dashboard
- Resend manually if needed

## Post-Deployment Verification

Within first 24 hours:
- [ ] At least 1 successful scan completion email
- [ ] At least 1 successful report email
- [ ] No critical errors in logs
- [ ] Email delivery rate >95%
- [ ] All links in emails work correctly

## Done! 🎉

Your email delivery system is ready for production.
