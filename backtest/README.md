# Backtest and stress test of bot_mt5.py

`indicator_review_backtest.ipynb` holds the full study with charts: faithful replay of the bot,
raw signal edge, permutation test, walk-forward, Monte Carlo, stress tests, Deflated Sharpe,
the gaps found in the indicator and code, and improvement research with an untouched holdout.

Reproduce:

```bash
bash backtest/fetch_histdata.sh          # XAUUSD M1 2022 to 2026 from histdata.com (about 55 MB, not committed)
cd backtest
python build_notebook.py                 # regenerates the notebook from source
jupyter nbconvert --to notebook --execute --inplace indicator_review_backtest.ipynb
```

- `engine.py`: signal (same formulas as `bot_mt5.py`, checked bar by bar in the notebook),
  M1-resolved exits, the bot's money rules, metrics (PSR, Deflated Sharpe).
- Prices are HistData bid quotes. Their stamps are New York local time with DST (checked against the 08:30 data-release spike and the 18:00 reopen), converted to UTC. Spread is modelled at 0.30 USD.

Framework backtest (the real `goldbot` engine with funded-account rules):

- `framework_backtest.ipynb` (built by `build_framework_notebook.py`): parity with the research
  engine, funded-account replay, first-week pass/fail odds, Monte Carlo, cost stress.
- `orb.py`: opening range breakout research engine used to choose the strategy settings.
- `llm_replay.py`: evaluates the LLM committee on a sample of historical setups with your
  provider and key (billed; answers are cached).

Strategy search (7,488 variants with luck controls):

- `search.py`: 76 entry rules from 38 families x session windows x trend filter x 18 exits,
  numba simulator on 1-minute exits, random-direction and no-cost controls, Reality Check / SPA,
  weekly funded-challenge simulator.
- `strategy_search.ipynb` (built by `build_search_notebook.py`): results and charts. Runs in about 2 minutes.
