# Security & Privacy Policy

## Overview

Security and privacy are important parts of the ZarKos Mini v1 project.

ZarKos Mini v1 is an open-source language model developed by **TuZhi Codes** under **TuZhi Studio**.

This policy explains how to report security issues, which versions are currently supported, and what users should know about privacy when running or integrating the model.

---

## Supported Versions

Security support depends on the currently maintained project version.

| Version | Security Support |
| ------- | ---------------- |
| ZarKos Mini v1 | :white_check_mark: |

Older or modified copies of the project may not receive security fixes.

For the latest security-related changes, always use the latest version available in the official repository.

---

## Reporting a Security Vulnerability

If you discover a potential security vulnerability, please report it privately instead of publicly opening an issue.

When reporting a vulnerability, include as much useful information as possible:

- A clear description of the issue
- Affected version or commit
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Potential security impact
- Relevant logs or error messages
- A possible fix, if you have one

Please do not include passwords, API keys, tokens, personal information, or other sensitive data in the report.

### Responsible Disclosure

Please allow reasonable time for the issue to be reviewed and addressed before publicly disclosing technical details.

Security reports will be reviewed on a best-effort basis.

The project may:

1. Confirm the report.
2. Investigate and reproduce the issue.
3. Determine its security impact.
4. Develop or publish a fix when appropriate.
5. Update the documentation if the issue affects users.

Not every report will necessarily result in a security release. Reports may be declined when the issue is not considered a security vulnerability, cannot be reproduced, or is outside the scope of the project.

---

## Security Best Practices

When deploying ZarKos Mini v1, especially as a public API or web service:

- Never commit `.env` files containing secrets.
- Never publish API keys, access tokens, or private credentials.
- Use environment variables for secrets.
- Keep dependencies updated.
- Restrict administrative endpoints.
- Add authentication and rate limiting to public APIs.
- Validate user-provided input.
- Avoid exposing internal error details to public users.
- Keep model files and server components from untrusted modification.
- Use HTTPS when serving the model over a network.
- Monitor logs for unusual activity.
- Run the service with only the permissions it actually needs.

The model itself does not replace application-level security controls.

---

# Privacy

## Model Privacy

ZarKos Mini v1 is a model, not a standalone data collection service.

The model does not inherently provide a user account system, analytics platform, advertising system, or external data-storage service.

However, privacy depends on how the model is integrated and deployed.

For example, an application using ZarKos Mini v1 may store:

- User messages
- Generated responses
- IP addresses
- Logs
- Account information
- API usage information

Such data handling is controlled by the application, server, hosting provider, and other services used by the deployment.

---

## Local Usage

When the model is run locally, model inference can be performed within the local environment.

Whether user input is stored, logged, transmitted, or processed by another service depends on the surrounding application and configuration.

Running a model locally does not automatically guarantee privacy if the application sends data to external services.

---

## Public API Deployments

If you deploy ZarKos Mini v1 as a public API, you are responsible for defining and communicating your own privacy practices.

Before deploying publicly, review:

- Request logging
- Response logging
- IP address handling
- Authentication data
- API keys
- Database storage
- Third-party services
- Data retention
- Server access
- Backup systems

Do not collect or retain user data unless there is a legitimate reason to do so.

---

## Sensitive Information

Users should avoid sending highly sensitive information to an AI service unless the deployment is specifically designed and secured for that type of data.

This includes:

- Passwords
- API keys
- Authentication tokens
- Private keys
- Financial credentials
- Personal identification documents
- Confidential business information

AI-generated responses should also not be treated as a secure storage mechanism for sensitive information.

---

## Third-Party Services

ZarKos Mini v1 may be integrated into applications that use third-party services.

Those services may have their own security and privacy policies.

The privacy behavior of a specific deployment should therefore be evaluated based on the complete application stack, not only the model.

---

## Security Issues in Dependencies

Security problems may also originate from:

- Runtime environments
- Libraries
- Frameworks
- Operating systems
- Container images
- Hosting infrastructure
- Third-party APIs

Keeping the surrounding software stack updated is therefore part of maintaining a secure deployment.

---

## Scope

This policy covers the ZarKos Mini v1 project and its official source distribution.

It does not guarantee the security or privacy of:

- Unofficial forks
- Modified versions
- Third-party applications
- Third-party APIs
- User-managed servers
- Untrusted model files
- External hosting environments

Users and deployers are responsible for securing their own environments.

---

## Contact

For security reports, use the private security reporting method provided by the GitHub repository.

Please avoid publicly posting exploitable vulnerability details before the issue has been reviewed.

For general bugs and non-security issues, use the project's normal GitHub issue tracker.

---

**ZarKos Mini v1**  
Model ID: `tuzhi/zarkos-mini-v1`  
Developed by **TuZhi Codes**  
Part of **TuZhi Studio**
