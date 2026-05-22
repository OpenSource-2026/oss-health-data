import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BASE_URL = "https://api.github.com"

HEADERS = {
    "Accept": "application/vnd.github+json"
}

if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


API_URL_FIELDS = [
    "contributors_url",
    "languages_url",
    "tags_url",
    "events_url",
    "deployments_url",
    "subscribers_url",
]


def infer_dtype(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def is_github_api_url(value):
    return isinstance(value, str) and value.startswith("https://api.github.com")


def request_json(url, sleep_sec=0.2):
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 403:
        raise RuntimeError(
            "GitHub API rate limit or permission error. "
            "Check your GITHUB_TOKEN."
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"GitHub API request failed: {response.status_code} {url}"
        )

    time.sleep(sleep_sec)
    return response.json()


def flatten_value(path, key, value, depth=0):
    rows = []

    rows.append({
        "path": path,
        "key": key,
        "dtype": infer_dtype(value),
        "example_value": value,
        "is_github_api_url": is_github_api_url(value),
        "fetch_status": None,
        "error_message": None,
        "depth": depth,
    })

    return rows


def flatten_api_result(base_path, value, depth=1):
    rows = []

    if isinstance(value, list):
        for i, item in enumerate(value):
            path = f"{base_path}.__api__[{i}]"
            rows.extend(flatten_value(path, f"[{i}]", item, depth))

    elif isinstance(value, dict):
        for key, item in value.items():
            path = f"{base_path}.__api__.{key}"
            rows.extend(flatten_value(path, key, item, depth))

    else:
        path = f"{base_path}.__api__"
        rows.extend(flatten_value(path, "__api__", value, depth))

    return rows


def extract_repo_raw(full_name):
    repo_url = f"{BASE_URL}/repos/{full_name}"
    repo_data = request_json(repo_url)
    rows = []

    for key, value in repo_data.items():
        rows.extend(flatten_value(
            path=key,
            key=key,
            value=value,
            depth=0,
        ))

    return pd.DataFrame(rows)


def fetch_selected_api_urls(df, api_url_fields=None):
    if api_url_fields is None:
        api_url_fields = API_URL_FIELDS

    rows = df.to_dict("records")

    for field in api_url_fields:
        matched = df[df["path"] == field]

        if matched.empty:
            continue

        url = matched["example_value"].iloc[0]

        if not is_github_api_url(url):
            continue

        try:
            api_result = request_json(url)

            rows.append({
                "path": f"{field}.__api_fetch__",
                "key": "__api_fetch__",
                "dtype": infer_dtype(api_result),
                "example_value": url,
                "is_github_api_url": True,
                "fetch_status": "success",
                "error_message": None,
                "depth": 0,
            })

            rows.extend(flatten_api_result(field, api_result, depth=1))

        except Exception as e:
            rows.append({
                "path": f"{field}.__api_fetch__",
                "key": "__api_fetch__",
                "dtype": "error",
                "example_value": url,
                "is_github_api_url": True,
                "fetch_status": "failed",
                "error_message": str(e),
                "depth": 0,
            })

    return pd.DataFrame(rows)


def build_repo_dataframe(full_name):
    base_df = extract_repo_raw(full_name)
    df2 = fetch_selected_api_urls(base_df)
    return df2


def save_repo_dataframe(full_name, output_path):
    df = build_repo_dataframe(full_name)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    full_name = "pandas-dev/pandas"
    output_path = "data/processed/pandas_raw_flattened.csv"

    df = save_repo_dataframe(full_name, output_path)
    print(df.head())
    print(df.shape)