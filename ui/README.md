# UI

This folder contains the web server and front-end for ZarKos Mini v1. It's plain Express + a single static page — no build step, no framework, no bundler.

## Layout

```
ui/
├── server.js              Express app: auth, training controls, and the inference API
└── public/
    ├── main.html           The dashboard page (login + training controls)
    ├── manifest.webmanifest PWA manifest (installable app metadata)
    ├── sw.js                Service worker, for offline caching / installability
    └── icon/
        └── app.png          App icon used by the page, manifest, and favicon
```

## What the dashboard does

The dashboard is a single page with two states:

1. **Login** — username/password form, checked against `ADMIN_USERNAME` / `ADMIN_PASSWORD` in `.env`.
2. **Training controls** — once logged in, start or stop training and watch live progress/logs.

That's it. There's no chat interface here — this page is only for operating training. Chat happens through the API (see the root [`README.md`](../README.md)) or via `npm run chat` in the terminal.

## Where the branding comes from

The page title, model name, and model ID shown in the dashboard aren't hardcoded — they're pulled from `.env` at runtime (`MODEL_NAME`, `MODEL_ID`, `MODEL_OWNER`, `MODEL_BRAND`). Change those values and the dashboard updates automatically, no HTML edits needed.

## Installing it as an app (PWA)

Because of the manifest and service worker, a supported browser can install the dashboard as a standalone app — handy for checking on training from a phone. Look for an **Install** option in the dashboard sidebar, or your browser's own **Add to Home screen** menu if it doesn't expose an in-page install prompt.

## Replacing the icon

Swap out `public/icon/app.png` with your own PNG. That single file is reused everywhere — the page, the favicon, and the PWA manifest all point to it, so there's nothing else to update.

## Security notes

- Training endpoints require a valid dashboard session (cookie set on login).
- The chat/completions API endpoints require `MODEL_API_KEY`, independent of the dashboard login.
- `/health` is intentionally left open for hosting providers' uptime checks.
- The page never echoes back the API key or admin password, even to a logged-in session.
