from __future__ import annotations

import argparse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from stock_knight.credentials import load_credentials, save_credentials


class _CallbackHandler(BaseHTTPRequestHandler):
    request_token: str | None = None

    def do_GET(self) -> None:  # noqa: N802
        qs = parse_qs(urlparse(self.path).query)
        token = (qs.get("request_token") or [None])[0]
        _CallbackHandler.request_token = token

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"<html><body><h3>stock-knight</h3><p>Login received. You can close this tab.</p></body></html>"
        )

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def _extract_request_token(text: str) -> str | None:
    """
    Accept either:
    - raw request_token
    - full redirected URL containing ?request_token=...
    """
    t = (text or "").strip()
    if not t:
        return None
    if "request_token=" in t:
        qs = parse_qs(urlparse(t).query)
        return (qs.get("request_token") or [None])[0]
    return t


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser("zerodha-login")
    p.add_argument("--creds-file", default="credentials.yaml")
    p.add_argument(
        "--mode",
        choices=["server", "manual"],
        default="manual",
        help=(
            "server: capture request_token on localhost callback (requires Zerodha app redirect URL "
            "set to http://127.0.0.1:<port>/). "
            "manual: open login URL and you paste request_token (or full redirected URL)."
        ),
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    args = p.parse_args(argv)

    creds = load_credentials(args.creds_file)
    z = creds.get("zerodha") or {}
    api_key = z.get("api_key")
    api_secret = z.get("api_secret")
    if not api_key or not api_secret:
        raise SystemExit(
            "Missing zerodha.api_key or zerodha.api_secret in creds file. "
            "Add them to credentials.yaml first."
        )

    try:
        from kiteconnect import KiteConnect
    except Exception as e:  # pragma: no cover
        raise SystemExit("Missing dependency: pip install -e '.[brokers]'") from e

    try:
        kite = KiteConnect(api_key=api_key)
        login_url = kite.login_url()
    except Exception as e:
        raise SystemExit(
            "Failed creating Kite client / login URL. "
            "Double-check zerodha.api_key in credentials.yaml (it must be the Kite Connect API key)."
        ) from e

    print("1) Opening login URL in browser:")
    print(login_url)
    webbrowser.open(login_url)
    print()
    print("Login note:")
    print("- Zerodha asks for Kite User ID or mobile number (not email).")
    print("- Complete 2FA; you will be redirected to your Redirect URL with request_token.")
    print()

    request_token: str | None
    if args.mode == "server":
        print(f"2) Waiting for callback on http://{args.host}:{args.port}/ ...")
        print(
            "   Make sure your Zerodha app redirect URL is exactly "
            f"http://{args.host}:{args.port}/"
        )
        httpd = HTTPServer((args.host, args.port), _CallbackHandler)

        # Wait until a request_token arrives
        while _CallbackHandler.request_token is None:
            httpd.handle_request()

        request_token = _CallbackHandler.request_token
        if not request_token:
            raise SystemExit("No request_token received. Check your app redirect URL and try again.")
    else:
        print("2) After login, Zerodha will redirect to your redirect URL like:")
        print("   https://your-redirect-url/?request_token=XXXX&status=success")
        print()
        try:
            pasted = input("   Paste request_token OR full redirected URL here: ")
        except KeyboardInterrupt:
            print()
            raise SystemExit("Cancelled. Paste the redirected URL (or request_token) when ready.") from None
        request_token = _extract_request_token(pasted)
        if not request_token:
            raise SystemExit("No request_token provided.")

    try:
        session = kite.generate_session(request_token, api_secret=api_secret)
    except Exception as e:
        raise SystemExit(
            "Failed exchanging request_token for access_token. "
            "Double-check zerodha.api_secret, and ensure the request_token is fresh (single-use)."
        ) from e
    access_token = session["access_token"]

    creds.setdefault("zerodha", {})
    creds["zerodha"]["access_token"] = access_token
    save_credentials(args.creds_file, creds)

    print("3) Saved fresh zerodha.access_token to creds file.")


if __name__ == "__main__":
    main()

