#!/usr/bin/env python3
"""Send a Telegram message passed as command-line args. Reads TG_* from env."""
import os
import sys
import urllib.parse
import urllib.request

def main() -> None:
    token = os.environ.get("TG_BOT_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    text = " ".join(sys.argv[1:]) or "(no message)"
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        urllib.request.urlopen(url, data=data, timeout=10)
    except Exception:
        pass

if __name__ == "__main__":
    main()
