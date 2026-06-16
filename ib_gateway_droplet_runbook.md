# IB Gateway on DigitalOcean Droplet — Runbook

_Reference for the headless, automated IB Gateway setup completed 2026-06-02 (Phase 3.5). Last updated 2026-06-15 (ib_async migration + daily→incremental rename)._
_This is the detailed companion to `project.md`. `project.md` stays synthetic for loading context into new chats; this file holds the full "how and why."_

---

## 1. What this delivers

A fully unattended Interactive Brokers paper-trading connection on the droplet:

- IB Gateway logs itself in (no human at the keyboard), survives reboots, and re-establishes its session daily/weekly on its own.
- The API listens on `127.0.0.1:4002` and is firewalled off from the public internet.
- A healthcheck probes it every 10 minutes, alerts via Telegram after 3 consecutive fails, and self-heals if it finds the gateway down or logged out.
- An **incremental**-pipeline timer is enabled and runs weekly 

Verified working after a **cold reboot** with zero manual steps: services came up, the gateway auto-logged-in, and an IB API client connected over localhost returning account `DUP678137`. _(Originally verified with `ib_insync`; the stack migrated to the drop-in `ib_async` on 2026-06-15.)_

---

## 2. The runtime stack

```
IBKR servers  (login + market data, outbound only)
      |
DigitalOcean droplet  (Ubuntu 24.04, UTC, ufw: only SSH open)
      |
  systemd (User=hugo, Restart=always)
      Xvfb :1            -> virtual display (Gateway is a Java GUI app)
        |
      IBC 3.23.0         -> fills login, accepts dialogs, sets API config
        |
      IB Gateway 10.45   -> headless, paper mode
        |
   API socket :4002      -> bound to all interfaces by IB, but localhost-only via ufw
        |
   Python clients        -> incremental pipeline / weekly rebalancer (same box)
```

---

## 3. Key facts, paths, versions

| Item | Value |
|------|-------|
| Droplet | DigitalOcean, IP `104.248.166.125`, user `hugo`, Ubuntu 24.04.4 LTS |
| Spec | 1 vCPU / **2 GB RAM** (resized up from 1 GB) / 25 GB disk |
| Swap | 2 GB swapfile, `vm.swappiness=10` (safety cushion, not primary) |
| Timezone | UTC (kept deliberately; market-time logic converts in code) |
| IB Gateway | **10.45**, offline `stable-standalone` installer, at `~/Jts/ibgateway/1045/` |
| Java | Bundled Azul Zulu 17 (came with the offline installer; no separate JDK) |
| IBC | **3.23.0**, at `~/ibc/` (outside the repo, owned by `hugo`) |
| IBC config | `~/ibc/config.ini`, `chmod 600` |
| Launch script | `~/ibc/gatewaystart.sh` (edited; see below) |
| API port | `4002` (paper), account `DUP678137`, login id `hugobrrss` |
| Alert secrets | `~/ibc/healthcheck.env`, `chmod 600` (Telegram bot token + chat id) |
| venv | `~/quant-research/venv` |
| Healthcheck reserved clientId | `99` (never reuse for pipelines) |

### `gatewaystart.sh` edits
- `TWS_MAJOR_VRSN=1045`
- `TRADING_MODE=paper`
- `IBC_PATH=/home/hugo/ibc`
- Launched with the `-inline` flag (see §6).

### `config.ini` settings
- `TradingMode=paper`
- `AcceptNonBrokerageAccountWarning=yes` (auto-dismisses the paper dialog that otherwise blocks API connections)
- `ExistingSessionDetectedAction=primaryoverride` (reclaims the session)
- `OverrideTwsApiPort=4002`
- `ReadOnlyApi=no` (so the API can place orders)
- `AcceptIncomingConnectionAction=accept` (interim; see §7 for the tighten-later note)
- `AutoRestartTime=05:00 AM` (UTC, since the box is UTC)
- `IbLoginId` / `IbPassword` = paper credentials (same login as Mac TWS)

---

## 4. systemd units

All in `/etc/systemd/system/`.

- **`xvfb.service`** — runs `Xvfb :1`, `Restart=always`, `ExecStartPre` removes a stale `/tmp/.X1-lock`.
- **`ibgateway.service`** — `ExecStart=/home/hugo/ibc/gatewaystart.sh -inline`, `User=hugo`, `Environment=DISPLAY=:1`, `Requires`/`After=xvfb.service`, `Restart=always`, `RestartSec=30`. **Enabled.**
- **`gateway-healthcheck.service` + `.timer`** — oneshot probe every 10 min (`OnUnitActiveSec=10min`, `OnBootSec=3min`). **Enabled.** (Acts on 3 consecutive failed probes)
- **`pipeline.service` + `.timer` + `pipeline-failure.service`** — Fri 21:30 UTC, `Persistent=true`.

Helper scripts: `src/execution/gateway_healthcheck.py` (probe) and `src/execution/notify.py` (reusable Telegram sender).

---

### 4.1 How the weekly pipeline runs on the Droplet
- two `systemd` units working as a pair: _alarm clock + task_
- `pipeline.timer` is the scheduler (`OnCalendar=Fri 21:30:00` set to Fri-21:30UTC, so after market close in the US). `Persistent=true` ensures the job runs as soon as Droplet is back in case it was down at the scheduled time
- `pipeline.service` is the task: when the timer fires, systemd starts this service that runs the `run_pipeline.py` script
- safety net (`OnFailure=`): if `pipeline.service` fails, systemd starts `pipeline-failure.service` which fires a Telegram alert

---

## 5. Operating it (common commands)

```bash
# status (use sudo to see service logs in the journal)
systemctl status ibgateway.service --no-pager
sudo journalctl -u ibgateway.service -n 50 --no-pager
sudo journalctl -u ibgateway.service -f          # live

# IBC's own diagnostic log
ls -lt ~/ibc/logs/ ; tail -n 60 ~/ibc/logs/ibc-3.23.0_GATEWAY-1045_*.txt

# is the API up?
ss -tlnp | grep 4002

# manual connection test (use a clientId that isn't 99)
source ~/quant-research/venv/bin/activate
python3 -c "from ib_async import IB; ib=IB(); ib.connect('127.0.0.1',4002,clientId=10,timeout=15); print(ib.isConnected(), ib.managedAccounts()); ib.disconnect()"

# restart / stop
sudo systemctl restart ibgateway.service
sudo systemctl stop ibgateway.service

# timers
systemctl list-timers --all --no-pager
```

---

## 6. Why the key decisions were made

- **Offline `stable-standalone` installer.** The Gateway has no self-updating build (unlike TWS), and the standalone installer gives a deterministic path (`~/Jts/ibgateway/1045`) and a pinned version. Reproducible and IBC-compatible.
- **`-inline` flag.** By default `gatewaystart.sh` wraps the gateway in an `xterm` and backgrounds it, so the launcher returns immediately — which would make systemd think the service died and restart-loop it. `-inline` `exec`s the launcher in the foreground so systemd supervises the real JVM.
- **Run as `hugo`, never root.** A Java GUI app holding broker credentials should not run as root. The repo and venv already live under `hugo`. (A dedicated service account is the cleaner answer for a *live* system; revisit at Phase 5.)
- **`ExistingSessionDetectedAction=primaryoverride`.** The droplet should always win and reclaim the session if something bumps it. **Consequence:** same login as Mac TWS, and IB allows one session per user — the droplet will bump a Mac TWS session on this account. For concurrent use, add a second IB username for the automated session (Phase 5).
- **`AutoRestartTime=05:00 AM` UTC.** Lets the session persist with a single weekly login (auto-restart, not auto-logoff). 05:00 UTC is clear of the US session (13:30–20:00 UTC) and IB's overnight reset. The weekly forced re-auth is handled automatically because it's paper (no 2FA).
- **localhost-only via `ufw`.** IB binds the API to all interfaces (`*:4002`); the firewall (`deny incoming`, allow only SSH) is what keeps it private. The loopback interface is never filtered, so local clients work.
- **Secrets split by tool.** IBC credentials live in `config.ini` because IBC is config-file-based (its documented, supported path). Telegram secrets use a systemd `EnvironmentFile` because the healthcheck reads env vars. Both files are `600`, owned by `hugo`, outside the repo.
- **Healthcheck self-heal.** `Restart=always` only catches a *crash*. The probe catches the *silent* failure (process alive but logged out / API unresponsive), alerts, and restarts. It exits 0 after remediating so the unit never sits in a "failed" state — the Telegram alert is the signal. 

---

## 7. IB Gateway version pinning & how to update

The `stable-standalone` build **never self-updates** — `1045` stays `1045` until you deliberately move it. This is intentional: you don't want a trading gateway silently changing version mid-week and breaking IBC or your code.

When you choose to update (or IBKR enforces a minimum version, which happens periodically):

1. Download the current `stable-standalone` installer and run it — it creates a new dir, e.g. `~/Jts/ibgateway/1046/`.
2. Edit `~/ibc/gatewaystart.sh`: `TWS_MAJOR_VRSN=1046`.
3. `sudo systemctl restart ibgateway.service`.
4. Verify: `ss -tlnp | grep 4002` and a Python connect test.
5. Once confirmed, delete the old version dir.

A controlled, tested upgrade — never automatic. (Could be wrapped in an `update_gateway.sh` helper later.)

---

## 8. Open items / follow-ups

- **Second IB username (Phase 5):** for running Mac TWS and the droplet concurrently without session conflict, and to keep live/real-money Mac trading isolated from the droplet's paper automation and its Telegram alerts.
- **Tighten API access (optional hardening):** add `127.0.0.1` to the Gateway's Trusted IPs and switch `AcceptIncomingConnectionAction` to `reject`.
- **`nbstripout` (hygiene):** strip notebook outputs on commit — keeps identifiers/data out of public history and makes notebooks read cleaner.

---

## 9. Security notes from this session

- Repo is **public by design** (CV goal). Secrets are gitignored (`.env`, `*.pem`, `*.key`, `credentials.json`) and confirmed off GitHub. History scan found no leaked credentials — only the paper account id in old committed notebook output (an identifier, not a secret; not worth rewriting history).
- `ufw` active: default deny incoming, only `22/tcp` (OpenSSH) allowed.
- **Telegram bot token was accidentally printed once and rotated** via @BotFather. Lesson: never `source`/echo a secrets file to the terminal; edit with an editor and verify with redaction (`sed -E 's/=.*/=<hidden>/'`).