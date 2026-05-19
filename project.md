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

## Droplet Startup Sequence (MANUAL — not yet automated)
1. `sudo Xvfb :1 -screen 0 1024x768x24 &`
2. `sudo -E DISPLAY=:1 /opt/ibgateway/ibgateway &`
3. Log in manually via xdotool (see previous chat for coordinates)
4. Uncheck Read-Only API in Configure → Settings → API → Settings
5. Verify: `sudo ss -tlnp | grep java` should show port 4002

**First task next session:** automate steps 2-4 into a script.

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