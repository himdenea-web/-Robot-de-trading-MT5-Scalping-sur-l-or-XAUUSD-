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
- Prices are HistData bid quotes in EST without DST, converted to UTC. Spread is modelled at 0.30 USD.
