# scripts/download_data.py
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

API_KEY = os.getenv("POLY_API")  # Set this in your environment
print("API KEY =", API_KEY)

BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={key}"

def fetch_minute_data(symbol, start_date, end_date):
    url = BASE_URL.format(
        symbol=symbol,
        start=start_date,
        end=end_date,
        key=API_KEY
    )
    print(f"Fetching {symbol} from {start_date} to {end_date}")
    r = requests.get(url)
    if r.status_code != 200:
        print("Error:", r.text)
        return None
    data = r.json().get("results", [])
    if not data:
        return None
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(columns={
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"
    })[["datetime", "open", "high", "low", "close", "volume"]]
    return df

def download_and_save(symbol, start_date, end_date, out_path):
    df = fetch_minute_data(symbol, start_date, end_date)
    if df is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path)
        print(f"Saved: {out_path}")
    else:
        print(f"No data for {symbol} on {start_date}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", type=str, default="2023-11-01")
    parser.add_argument("--end", type=str, default="2023-11-10")
    args = parser.parse_args()

    for symbol in args.symbols:
        out_file = f"data/minute/{symbol}_{args.start}_{args.end}.parquet"
        download_and_save(symbol, args.start, args.end, out_file)
        time.sleep(1)  # respect rate limits
