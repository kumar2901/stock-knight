"""
Angel One SmartAPI session helper.

Verifies credentials, generates a session, and saves the auth token
to credentials.yaml so it can be reused by the pipeline.

Usage:
    python -m stock_knight.auth.angelone_login [--creds-file credentials.yaml]

Required entries in credentials.yaml before running:
    angelone:
      api_key:      "YOUR_API_KEY"        # from SmartAPI console
      client_code:  "YOUR_ANGEL_CLIENT_ID" # e.g. A12345
      password:     "YOUR_TRADING_PIN"    # 4-digit Mpin / trading pin
      totp_secret:  "YOUR_TOTP_SECRET"    # Base-32 secret from authenticator app setup

Steps to get these credentials — see README or run with --help.
"""

from __future__ import annotations

import argparse

from stock_knight.credentials import load_credentials, save_credentials


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        "angelone-login",
        description=(
            "Verify Angel One SmartAPI credentials and save a fresh session token.\n\n"
            "Follow these one-time setup steps before running this script:\n"
            "  1. Open a FREE Angel One demat + trading account at https://www.angelone.in/\n"
            "  2. Go to https://smartapi.angelbroking.com/ → Login with your Angel One account\n"
            "  3. Click 'Create New App' → fill in App Name, redirect URL (use http://localhost)\n"
            "  4. Copy the API Key shown on the app page\n"
            "  5. Set up TOTP authenticator:\n"
            "       a) In the Angel One mobile app → Profile → SmartAPI → Enable TOTP\n"
            "       b) Scan the QR code shown with Google Authenticator or Authy\n"
            "       c) Important: also note/copy the Base-32 Secret key shown below the QR code\n"
            "  6. Fill credentials.yaml with the values from steps 4-5\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--creds-file", default="credentials.yaml")
    args = p.parse_args(argv)

    creds = load_credentials(args.creds_file)
    a = creds.get("angelone") or {}

    api_key = a.get("api_key", "")
    client_code = a.get("client_code", "")
    password = a.get("password", "")
    totp_secret = a.get("totp_secret", "")
    totp_value = a.get("totp_value", "")

    # Validate required fields
    missing = []
    if not api_key or api_key.startswith("YOUR_"):
        missing.append("angelone.api_key")
    if not client_code or client_code.startswith("YOUR_"):
        missing.append("angelone.client_code")
    if not password or password.startswith("YOUR_"):
        missing.append("angelone.password")
    if (not totp_secret or totp_secret.startswith("YOUR_")) and (not totp_value or totp_value.startswith("YOUR_")):
        missing.append("angelone.totp_secret  (or angelone.totp_value for a one-time code)")

    if missing:
        print("Missing credentials in credentials.yaml:")
        for m in missing:
            print(f"  {m}")
        print()
        print("Add them following the setup steps above, then re-run this script.")
        raise SystemExit(1)

    try:
        import pyotp
        has_pyotp = True
    except ImportError:
        has_pyotp = False

    # Resolve TOTP
    if totp_secret:
        if not has_pyotp:
            raise SystemExit(
                "pyotp is required to use totp_secret. Run:\n"
                "  pip install pyotp"
            )
        import pyotp
        totp_code = pyotp.TOTP(totp_secret).now()
        print("Generated TOTP code from secret.")  # Do not log the actual code
    elif totp_value:
        totp_code = str(totp_value).strip()
        print("Using provided TOTP value.")
    else:
        raise SystemExit("No TOTP source found.")

    try:
        from SmartApi import SmartConnect
    except Exception as e:
        raise SystemExit(
            "Missing dependency. Run:\n"
            "  pip install -e '.[brokers]'"
        ) from e

    print(f"\nConnecting to Angel One SmartAPI (client: {client_code})...")
    smart = SmartConnect(api_key=api_key)

    try:
        session = smart.generateSession(client_code, password, totp_code)
    except Exception as e:
        raise SystemExit(
            f"Session generation raised an exception: {e}\n"
            "Check your api_key, client_code, password, and TOTP code."
        ) from e

    if not isinstance(session, dict) or session.get("status") is False:
        msg = (session or {}).get("message", str(session))
        raise SystemExit(
            f"Session generation failed: {msg}\n\n"
            "Common causes:\n"
            "  - Wrong client_code (use your Angel One login ID, not email)\n"
            "  - Wrong password (it's the 4-digit MPIN, not the full password)\n"
            "  - TOTP code expired (TOTP codes are valid for 30 seconds only; retry)\n"
            "  - api_key copied incorrectly from the SmartAPI console"
        )

    data = session.get("data") or {}
    auth_token = data.get("jwtToken", "")
    refresh_token = data.get("refreshToken", "")
    feed_token = data.get("feedToken", "")

    print("\nSession generated successfully.")
    print(f"  jwtToken     : {'*' * 20}...  (saved to credentials file)")
    print(f"  feedToken    : {'*' * 10}...  (saved to credentials file)")

    # Save tokens back to credentials.yaml
    creds.setdefault("angelone", {})
    creds["angelone"]["auth_token"] = auth_token
    creds["angelone"]["refresh_token"] = refresh_token
    creds["angelone"]["feed_token"] = feed_token
    save_credentials(args.creds_file, creds)

    print(f"\nSaved tokens to {args.creds_file}")
    print("You can now use --broker angelone in the pipeline.")
    print("\nNote: Angel One tokens are valid for the trading session (typically until next login).")
    print("Re-run this script each morning before using live features.")


if __name__ == "__main__":
    main()
