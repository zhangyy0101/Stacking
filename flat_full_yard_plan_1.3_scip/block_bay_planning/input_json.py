from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


INPUT_JSON_NAME = "input_data.json"


def input_json_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / INPUT_JSON_NAME


def has_input_json(data_dir: str | Path) -> bool:
    return input_json_path(data_dir).exists()


@lru_cache(maxsize=4)
def load_input_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def input_data(data_dir: str | Path) -> dict[str, Any]:
    return load_input_json(str(input_json_path(data_dir).resolve()))


def dataframe_from_split(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if not isinstance(value, dict):
        return pd.DataFrame(value)
    return pd.DataFrame(
        data=value.get("data", []),
        index=value.get("index"),
        columns=value.get("columns"),
    )


def input_dataframe(data_dir: str | Path, key: str, columns: list[str] | None = None) -> pd.DataFrame:
    frame = dataframe_from_split(input_data(data_dir).get(key))
    if columns is not None and not frame.empty:
        keep = [column for column in columns if column in frame.columns]
        return frame.loc[:, keep].copy()
    return frame


def input_value(data_dir: str | Path, key: str, default: Any = None) -> Any:
    return input_data(data_dir).get(key, default)


def vessel_container_ids(data_dir: str | Path) -> list[str]:
    vessels = input_data(data_dir).get("vessel_containers", {})
    return list(vessels) if isinstance(vessels, dict) else []


def vessel_doc_frame(data_dir: str | Path, voyage_id: object) -> pd.DataFrame:
    voyage = _voyage(voyage_id)
    vessels = input_data(data_dir).get("vessel_containers", {})
    if not isinstance(vessels, dict):
        return pd.DataFrame()
    content = vessels.get(voyage, {})
    if not isinstance(content, dict):
        return pd.DataFrame()
    return dataframe_from_split(content.get("doc_cntrs"))


def vessel_predict_cntrs(data_dir: str | Path, voyage_id: object) -> dict[str, Any]:
    voyage = _voyage(voyage_id)
    vessels = input_data(data_dir).get("vessel_containers", {})
    if not isinstance(vessels, dict):
        return {}
    content = vessels.get(voyage, {})
    if not isinstance(content, dict):
        return {}
    predict = content.get("predict_cntrs", {})
    return predict if isinstance(predict, dict) else {}


def _voyage(value: object) -> str:
    text = "" if value is None else str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text
