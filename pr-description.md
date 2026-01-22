feat: add OAuth2 user authentication

## What
Added comprehensive OAuth2 authentication support for Google, GitHub, and Microsoft accounts

## Why
Users have been requesting social login options to avoid creating new passwords. Current email/password only authentication has:
- High abandonment rate during signup (65%)
- Password reset requests (40/month)
- Security concerns about password storage

## How
- Integrated Passport.js with OAuth2 strategies for Google, GitHub, and Microsoft
- Added new database tables for `user_identities` to link multiple auth providers
- Created secure token validation middleware
- Implemented account linking for existing users
- Added rate limiting to prevent abuse

## Testing
1. Start the application with `npm run dev`
2. Navigate to `/login` - you should see new "Sign in with..." buttons
3. Test each provider:
   - Google: Use personal account
   - GitHub: Use work account
   - Microsoft: Use company SSO
4. Verify account linking works for existing users
5. Check that sessions persist across browser restarts

## Security Considerations
- All OAuth tokens are encrypted at rest
- Implemented PKCE flow for additional security
- Added scope validation to prevent privilege escalation
- Rate limited to 5 attempts per IP per minute

## Configuration Required
Add these environment variables to your `.env`:
```
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
MICROSOFT_CLIENT_ID=your_microsoft_client_id
MICROSOFT_CLIENT_SECRET=your_microsoft_client_secret
```

## Database Migration
Run `npm run migrate` to create the `user_identities` table

## Related
Closes #234
Relates to #156, #189

## Screenshots
[Login page with new OAuth buttons]
[Account linking flow]
[Mobile responsive design]
