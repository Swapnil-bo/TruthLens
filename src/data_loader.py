import os
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

# ── Raw GitHub URLs for FakeNewsNet (PolitiFact subset) ───────────────────────
FAKE_URL = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_fake.csv"
REAL_URL = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_real.csv"

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH  = os.path.join(DATA_DIR, "raw_news.csv")


def _download_csv(url: str, label: int) -> pd.DataFrame:
    """Download one CSV from GitHub and attach a label column."""
    print(f"  Downloading {'FAKE' if label == 0 else 'REAL'} news from FakeNewsNet …")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(response.text))

    # Keep only the columns we need; rename for consistency
    df = df[["title", "news_url"]].copy()
    df.rename(columns={"title": "headline", "news_url": "url"}, inplace=True)
    df["label"] = label          # 0 = fake, 1 = real
    df["body"]  = ""             # body text filled later via newspaper3k (optional)
    return df


def download_raw_data(force: bool = False) -> pd.DataFrame:
    """
    Download FakeNewsNet PolitiFact split and save to data/raw_news.csv.
    Skips download if file already exists (use force=True to re-download).
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(RAW_PATH) and not force:
        print(f"[data_loader] raw_news.csv already exists — loading from disk.")
        return pd.read_csv(RAW_PATH)

    print("[data_loader] Fetching FakeNewsNet dataset …")
    fake_df = _download_csv(FAKE_URL, label=0)
    real_df = _download_csv(REAL_URL, label=1)

    df = pd.concat([fake_df, real_df], ignore_index=True)
    df.drop_duplicates(subset=["headline"], inplace=True)
    df.dropna(subset=["headline"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(RAW_PATH, index=False)
    print(f"[data_loader] Saved {len(df)} articles → {RAW_PATH}")
    return df


def load_splits(test_size: float = 0.2, random_state: int = 42):
    """
    Returns (X_train, X_test, y_train, y_test) where X is a DataFrame
    of raw text columns and y is the label Series.
    """
    df = download_raw_data()

    X = df[["headline", "body", "url"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[data_loader] Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    download_raw_data(force=True)