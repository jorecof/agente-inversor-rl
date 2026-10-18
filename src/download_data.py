"""Descarga reproducible con fechas cerradas; no requiere claves de API."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/btcusdt_daily.csv")
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("El CSV ya existe; se conserva la instantánea original.")
    start = int(dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc).timestamp()*1000)
    end = int(dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc).timestamp()*1000)-1
    rows, urls = [], []
    while start < end:
        params = urllib.parse.urlencode(dict(symbol="BTCUSDT", interval="1d",
                                             startTime=start, endTime=end, limit=1000))
        url = "https://data-api.binance.vision/api/v3/klines?"+params
        with urllib.request.urlopen(url, timeout=60) as response:
            batch = json.load(response)
        if not isinstance(batch, list):
            raise ValueError(batch)
        if not batch:
            break
        rows.extend(batch)
        urls.append(url)
        start = int(batch[-1][0])+86400000
    data = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume",
                                      "close_time", "quote_volume", "trades", "buy_base", "buy_quote", "ignore"])
    data["timestamp"] = pd.to_datetime(data.time, unit="ms", utc=True)
    data = data[["timestamp", "open", "high", "low", "close", "volume"]]
    if len(data) != 1827 or data.timestamp.duplicated().any():
        raise ValueError("Se esperaban exactamente 1827 velas diarias, 2020-2024 completos.")
    target.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(target, index=False)
    metadata = {"asset": "BTC/USDT", "exchange": "Binance", "interval": "1d",
                "timestamp_meaning": "Inicio de vela UTC; close disponible al terminar ese día",
                "period": "2020-01-01 a 2024-12-31 inclusive", "rows": len(data),
                "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "urls": urls, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "user_reference": "https://gym-trading-env.readthedocs.io/en/latest/download.html",
                "method": "REST público de Binance; no se utilizó gym_trading_env.downloader"}
    target.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Guardadas {len(data)} velas en {target}")


if __name__ == "__main__":
    main()
