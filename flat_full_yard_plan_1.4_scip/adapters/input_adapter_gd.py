import json
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd


DEFAULT_ROUGH_ATTR = ["IYC_CSZ_CSIZECD", "IYC_POT_UNLDPORT"]
DEFAULT_DETAIL_ATTR = ["IYC_CSZ_CSIZECD", "IYC_POT_UNLDPORT", "IYC_CHEIGHTCD"]
DEFAULT_WEIGHT_LEVEL = [0, 10, 15, 20, 25, 30]


def normalize_voyage_id(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def list_or_default(value: Any, default: List[Any]) -> List[Any]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def voyage_rule_dict(value: Any, voyages: List[str], default: List[Any], *, fill_missing: bool = True) -> Dict[str, List[Any]]:
    normalized_voyages = [normalize_voyage_id(voyage) for voyage in voyages if normalize_voyage_id(voyage)]
    if isinstance(value, dict):
        out = {
            normalize_voyage_id(voyage): list_or_default(rules, default)
            for voyage, rules in value.items()
            if normalize_voyage_id(voyage)
        }
    elif value is None:
        out = {}
    else:
        shared = list_or_default(value, default)
        out = {voyage: list(shared) for voyage in normalized_voyages}
    if fill_missing:
        for voyage in normalized_voyages:
            out.setdefault(voyage, list(default))
    return out


def clean_for_json(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return None
        df_clean = obj.astype(object).where(pd.notna(obj), None)
        return df_clean.to_dict(orient="split")
    if isinstance(obj, pd.Series):
        if obj.empty:
            return None
        series_clean = obj.astype(object).where(pd.notna(obj), None)
        return series_clean.to_dict()
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (np.integer, np.signedinteger, np.unsignedinteger)):
        return int(obj)
    if isinstance(obj, (np.floating, np.complexfloating)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(item) for item in obj]
    return obj


def restore_dataframe_from_split(data: Optional[Dict]) -> Optional[pd.DataFrame]:
    if data is None:
        return None
    try:
        return pd.DataFrame(data=data.get("data", []), index=data.get("index"), columns=data.get("columns"))
    except Exception:
        return pd.DataFrame()


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return None if pd.isna(obj) else obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


class InputAdapterGd:
    """Guandong input adapter used by the standalone flat yard planner."""

    def __init__(self):
        self.take_over_vessel: Dict[str, List] = {}
        self.bay_slots_detail: pd.DataFrame = None
        self.tops_plan: pd.DataFrame = None
        self.area_function_info: pd.DataFrame = None
        self.vessel_berth_info: pd.DataFrame = None
        self.planning_time: pd.Timestamp = pd.Timestamp.now()
        self.history_plan_info: pd.DataFrame = None
        self.vessel_containers: Dict[str, Dict[str, pd.DataFrame | Dict]] = {}
        self.closed_area: Set[str] = set()
        self.berth_area_dist_matrix: pd.DataFrame = None
        self.voyage_predict: Dict = {}
        self.large_plan: Dict = {}
        self.adjust_plan_info: Dict = {}
        self.user_design: bool = True
        self.user_design_large_plan_area: List[str] = []
        self.rough_attr: Dict[str, List[str]] = {}
        self.detail_attr: Dict[str, List[str]] = {}
        self.bay_rules: Dict[str, List[str]] = {}
        self.row_rules: Dict[str, List[str]] = {}
        self.weight_level: Dict[str, List[int]] = {}
        self.is_data_local: bool = False
        self.local_path: str = None
        self.need_save_data: bool = True

    def to_dict(self) -> dict:
        return {
            "__class__": "InputAdapter",
            "take_over_vessel": self.take_over_vessel,
            "bay_slots_detail": clean_for_json(self.bay_slots_detail),
            "tops_plan": clean_for_json(self.tops_plan),
            "area_function_info": clean_for_json(self.area_function_info),
            "vessel_berth_info": clean_for_json(self.vessel_berth_info),
            "planning_time": self.planning_time,
            "history_plan_info": clean_for_json(self.history_plan_info),
            "vessel_containers": clean_for_json(self.vessel_containers),
            "closed_area": list(self.closed_area),
            "berth_area_dist_matrix": clean_for_json(self.berth_area_dist_matrix),
            "voyage_predict": clean_for_json(self.voyage_predict),
            "large_plan": clean_for_json(self.large_plan),
            "adjust_plan_info": clean_for_json(self.adjust_plan_info),
            "user_design": self.user_design,
            "user_design_large_plan_area": self.user_design_large_plan_area,
            "rough_attr": self.rough_attr,
            "detail_attr": self.detail_attr,
            "bay_rules": self.bay_rules,
            "row_rules": self.row_rules,
            "weight_level": self.weight_level,
            "is_data_local": self.is_data_local,
            "local_path": self.local_path,
            "need_save_data": self.need_save_data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InputAdapterGd":
        obj = cls()
        obj.take_over_vessel = data.get("take_over_vessel", {})
        obj.bay_slots_detail = restore_dataframe_from_split(data.get("bay_slots_detail"))
        obj.tops_plan = restore_dataframe_from_split(data.get("tops_plan"))
        obj.area_function_info = restore_dataframe_from_split(data.get("area_function_info"))
        obj.vessel_berth_info = restore_dataframe_from_split(data.get("vessel_berth_info"))

        pt = data.get("planning_time")
        obj.planning_time = pd.Timestamp(pt) if pt is not None else pd.NaT
        obj.history_plan_info = restore_dataframe_from_split(data.get("history_plan_info"))

        vessel_containers_raw = data.get("vessel_containers", {})
        vessel_containers_restored = {}
        for vid, content in vessel_containers_raw.items():
            new_content = {}
            for key, value in content.items():
                if key in {"doc_cntrs"}:
                    new_content[key] = restore_dataframe_from_split(value)
                else:
                    new_content[key] = value
            vessel_containers_restored[vid] = new_content
        obj.vessel_containers = vessel_containers_restored

        takeover_vessel_list = sum(obj.take_over_vessel.values(), [])
        obj.closed_area = set(data.get("closed_area", []))
        obj.berth_area_dist_matrix = restore_dataframe_from_split(data.get("berth_area_dist_matrix"))
        obj.voyage_predict = data.get("voyage_predict", {})
        obj.large_plan = data.get("large_plan", {})
        obj.adjust_plan_info = data.get("adjust_plan_info", {})
        user_design_value = data["user_design"] if "user_design" in data else obj.user_design
        obj.user_design = bool(user_design_value) if user_design_value is not None else False
        obj.user_design_large_plan_area = list_or_default(data.get("user_design_large_plan_area"), [])
        obj.rough_attr = voyage_rule_dict(data.get("rough_attr"), takeover_vessel_list, DEFAULT_ROUGH_ATTR)
        obj.detail_attr = voyage_rule_dict(data.get("detail_attr"), takeover_vessel_list, DEFAULT_DETAIL_ATTR)
        obj.bay_rules = voyage_rule_dict(data.get("bay_rules"), takeover_vessel_list, [], fill_missing=False)
        obj.row_rules = voyage_rule_dict(data.get("row_rules"), takeover_vessel_list, [], fill_missing=False)
        obj.weight_level = voyage_rule_dict(
            data.get("weight_level"),
            takeover_vessel_list,
            DEFAULT_WEIGHT_LEVEL,
            fill_missing=False,
        )
        obj.is_data_local = data.get("is_data_local", False)
        obj.local_path = data.get("local_path")
        obj.need_save_data = data.get("need_save_data", True)
        return obj

    def save_to_json(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, cls=SafeJSONEncoder, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_json(cls, filepath: str) -> "InputAdapterGd":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


if __name__ == "__main__":
    json_path = ""
    data = InputAdapterGd.load_from_json(json_path)
