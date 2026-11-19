#!/usr/bin/env bash
# Download free XAUUSD M1 bars (ASCII, EST without DST) from histdata.com.
# Usage: bash backtest/fetch_histdata.sh   -> zips in backtest/data/histdata/
set -e
OUT="$(dirname "$0")/data/histdata"; mkdir -p "$OUT"
CJ="$OUT/.cookies"
get() {  # $1 = page path suffix (YYYY or YYYY/M), $2 = date, $3 = datemonth
  local url="https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/xauusd/$1"
  local tk; tk=$(curl -s -A "Mozilla/5.0" -c "$CJ" "$url" | grep -oE 'name="tk" id="tk" value="[^"]+' | head -1 | sed 's/.*value="//')
  curl -s -A "Mozilla/5.0" -b "$CJ" -e "$url" -o "$OUT/XAUUSD_$3.zip" \
    --data "tk=$tk&date=$2&datemonth=$3&platform=ASCII&timeframe=M1&fxpair=XAUUSD" \
    https://www.histdata.com/get.php
  echo "$3 $(stat -c %s "$OUT/XAUUSD_$3.zip") bytes"
}
for y in 2022 2023 2024 2025; do get "$y" "$y" "$y"; sleep 2; done
for m in 1 2 3 4 5 6 7 8 9 10 11; do get "2026/$m" 2026 "2026$(printf %02d $m)"; sleep 2; done
