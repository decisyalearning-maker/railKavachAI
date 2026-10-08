from datetime import datetime
from pathlib import Path
import pandas as pd

LOG_PATH = Path("data/events.csv")

def log_event(data):
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), **data}
    df = pd.DataFrame([row])
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)

def read_events():
    if not LOG_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(LOG_PATH)
