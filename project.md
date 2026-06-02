# Quant Research Project - Context

## How We Work
- You guide and question me; I write the code. Iterative, back and forth.
- Challenge me — I'm aiming for quant shop best practices, not comfort.
- Claude Code handles tedious config/setup tasks.
- PyCharm + Jupyter notebooks for development and testing.
- Two machines (Mac: /Users/hugo/, Windows: C:\Users\hbourrou\) synced via GitHub.

## GitHub
https://github.com/hugobrrss/quant-research
(fetch latest files from here at the start of each chat)

## Infrastructure
- DigitalOcean Droplet: 104.248.166.125, user: hugo
- IB Paper account: DUP678137
- TWS (Mac): port 7497 | IB Gateway (Droplet): port 4002

## Data Assets
| File | Location | Description |
|------|----------|-------------|
| russell1000_production.parquet | Mac + Droplet | 3yr IB data, 1005 tickers |
| russell1000_research.parquet | Mac only | 3yr IB data, full features |
| crsp_clean.parquet | Mac only | 30yr CRSP daily data, ~16k securities |

## Droplet Current State

**Server:** DigitalOcean, IP 104.248.166.125, user: hugo, Ubuntu 24.04 LTS
**Disk:** 24GB, ~40% used
**Python:** 3.12, venv at ~/quant-research/venv
**Code:** cloned from GitHub at ~/quant-research
**Data:** russell1000_production.parquet and russell1000_research.parquet present at ~/quant-research/data/

**IB Gateway status: NOT running**
- Previous attempt used the auto-updating installer → wrong path structure, incompatible with IBC
- Correct approach: offline installer + IBC (Interactive Brokers Controller)
- Offline installer URL: https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64-offline.sh
- IBC installs to /opt/ibc, expects IB Gateway at ~/Jts/ibgateway/1037/
- IBC config.ini credentials must be set (IbLoginId, IbPassword, TradingMode=paper, ExistingSessionDetectedAction=disconnect, AcceptNonBrokerageAccountWarning=yes, ReadOnlyApi=no)
- Xvfb required for virtual display: sudo Xvfb :1 -screen 0 1024x768x24 &

**Next session exact steps:**
1. Download offline IB Gateway installer
2. Install to default location (~/Jts)
3. Reinstall IBC to /opt/ibc
4. Configure /root/ibc/config.ini
5. Run gatewaystart.sh and verify java process is running
6. Test Python connection on port 4002
7. Set up systemd service for auto-start on reboot

## Phase Progress
- ✅ Phase 1: Data pipelines (fetch, validate, process, features)
- ✅ Phase 2: ML models (baseline, trees, neural net)
- ✅ Phase 3: Live APIs, IB connection, backfill, daily pipeline
- 🔄 Phase 3.5: Droplet automation (IB Gateway startup + cron) ← CURRENT
- ⬜ Phase 4: Momentum strategy (signal → portfolio → execution → paper trading)
- ⬜ Phase 5: Managed futures + systematic global macro

## Next Session Starts Here
1. SSH into Droplet, run startup sequence above
2. Fix Read-Only API checkbox
3. Write `ib_gateway_login.sh` to automate the full login
4. Set up cron for daily pipeline
5. Then move to Phase 4