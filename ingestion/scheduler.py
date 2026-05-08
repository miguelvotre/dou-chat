"""
Daily scheduler for incremental DOU ingestion.

Runs every weekday at 8am and ingests the previous day's DOU.

Usage:
    python -m ingestion.scheduler          # runs in a continuous loop
    crontab: 0 8 * * 1-5 python -m ingestion.scheduler --once
"""

import argparse
from datetime import date, timedelta

import schedule
import time

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()


def ingest_yesterday() -> None:
    yesterday = date.today() - timedelta(days=1)
    # skip weekends
    if yesterday.weekday() >= 5:
        console.print(f"[dim]{yesterday} is a weekend, skipping.[/dim]")
        return
    console.print(f"[cyan]Starting incremental ingestion: {yesterday}[/cyan]")
    from ingestion.pipeline import run_pipeline
    run_pipeline(yesterday, yesterday)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    if args.once:
        ingest_yesterday()
        return

    schedule.every().day.at("08:00").do(ingest_yesterday)
    console.print("[green]Scheduler started. Daily ingestion at 08:00.[/green]")

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
