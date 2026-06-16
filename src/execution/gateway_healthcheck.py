#!/usr/bin/env python3
"""Healthcheck for the IB Gateway on the droplet.

Connects to the local API and confirms the session is logged in (a managed
account is returned). On failure it alerts via Telegram and restarts the
ibgateway service, then exits 0 once it has done its job (detect + remediate)
so systemd never leaves the unit 'failed' — the alert is the signal.
"""
import os
import sys
import subprocess
import urllib.parse
import urllib.request

HOST = "127.0.0.1"
PORT = 4002
CLIENT_ID = 99  # reserved for the healthcheck; never reuse for pipelines
FAIL_THRESHOLD = 3
STATE_FILE = "/home/hugo/ibc/healthcheck_fail.count"


def send_alert(text: str) -> None:
    token = os.environ.get("TG_BOT_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        urllib.request.urlopen(url, data=payload, timeout=10)
    except Exception:
        pass


def gateway_healthy() -> tuple[bool, str]:
    try:
        from ib_async import IB
        ib = IB()
        ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=15)
        ok = ib.isConnected() and bool(ib.managedAccounts())
        ib.disconnect()
        return ok, "" if ok else "connected but no managed account"
    except Exception as exc:
        return False, repr(exc)

def read_failures() -> int:
    try:
        with open(STATE_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return 0


def write_failures(n: int) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            f.write(str(n))
    except Exception as exc:
        print(f"[healthcheck] could not write state file: {exc!r}", file=sys.stderr)


def main() -> int:
    ok, detail = gateway_healthy()
    if ok:
        write_failures(0)
        return 0
    else:
        n = read_failures() + 1
        write_failures(n)

        if n >= FAIL_THRESHOLD:
            print(f"[healthcheck] gateway unhealthy ({detail}); alerting and restarting", file=sys.stderr)
            send_alert(f"[ALERT] IB Gateway healthcheck failed {n} consecutive times on the droplet: {detail}. Restarting ibgateway.service.")
            subprocess.run(["sudo", "/usr/bin/systemctl", "restart", "ibgateway.service"], check=False)
            write_failures(0)
            return 0
        else:
            print(f"[healthcheck] probe failed {n}/{FAIL_THRESHOLD}, not acting yet", file=sys.stderr)
            return 0


if __name__ == "__main__":
    sys.exit(main())
