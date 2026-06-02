import json
from decimal import Decimal
from typing import Dict, List, Set, Any, Union, Optional

from datetime import datetime
import pandas as pd
import numpy as np

def clean_for_json(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return None
        df_clean = obj.astype(object).where(pd.notna(obj), None)
        return df_clean.to_dict(orient='split')

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
        return pd.DataFrame(
            data=data.get('data', []),
            index=data.get('index'),
            columns=data.get('columns')
        )
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

class InputAdapterGd():
    """
    冠东输入适配器类
    """

    def __init__(self):
        self.take_over_vessel: Dict[str, List] = {}       # 接管的航次信息
        self.bay_slots_detail: pd.DataFrame               # 空间场箱位信息
        self.tops_plan: pd.DataFrame = None               # tops上已有的计划占位信息
        self.area_function_info: pd.DataFrame = None      # 箱区功能和负载信息
        self.vessel_berth_info: pd.DataFrame = None       # 船舶靠泊信息
        self.planning_time: pd.Timestamp = pd.Timestamp.now()     # 做计划时间
        self.history_plan_info: pd.DataFrame = None       # 接管航次的上一次计划信息
        self.vessel_containers: Dict[str, Dict[str, pd.DataFrame|Dict]] = {}      # 航次的箱信息
        # 结构信息
        # vessel_containers： {
        #    "452364" : {
        #         "doc_cntrs" : 航次对应的箱表 (pd.DataFrame),
        #         "predict_cntrs" : {
        #                   "20" : {
        #                       "total_volume" : 200,
        #                       "detail_info" : {
        #                               "DEHAM" : 20,
        #                           }
        #                   }
        #               },
        #         "work_lanes" : 3,
        #         "type" : "E"
        #   }
        #}
        self.closed_area: Set[str] = set()                # 关闭的箱区
        self.berth_area_dist_matrix: pd.DataFrame = None  # 泊位箱区距离矩阵
        self.voyage_predict: Dict = {}                    # 航次预测信息,已融合到vessel_containers中
        self.large_plan: Dict = {}                        # 大计划计算结果,大计划的计算结果，转成中计划输入所需的dict形式，存入这个成员变量，供中计划使用
        self.adjust_plan_info: Dict = {}                  # 计划的人工调整信息
        """
        {
        "large_plan":
            {
                "voyid":{
                    "add": [],  # 添加的箱区为必须使用箱区
                    "remove": []  # 删除的箱区为不能用的箱区
                },
            },
        "medium_plan":
            {
                "voyid":{
                    "20_port":{
                    "add": [],  # 添加的箱区为必须使用箱区
                    "remove": []  # 删除的箱区为不能用的箱区
                    },
                    "40_port":{
                    "add": [],  # 添加的箱区为必须使用箱区
                    "remove": []  # 删除的箱区为不能用的箱区
                },
            },
        "small_plan":{
            {
                "voyid":{
                    "20_port_hq":{
                    "add": [],  # 添加的箱区、贝为必须使用箱区
                    "remove": []  # 删除的箱区、贝为不能用的箱区
                    },
                    "40_port_hq":{
                    "add": [],  # 添加的箱区、贝为必须使用箱区
                    "remove": []  # 删除的箱区、贝为不能用的箱区
                },
            }
        """
        self.user_design: bool = True                     # 是否用户指定大计划区域，区域放在user_design_large_plan_area里
        self.user_design_large_plan_area: List[str] = []     # 用户指定的大计划区域列表
        self.is_data_local: bool = False                  # 是否本地加载信息
        self.local_path: str = None                       # 本地加载文件地址
        self.need_save_data: bool = True                  # 是否保存输入信息

    def to_dict(self) -> dict:
        """Convert entire object to JSON-safe dictionary (recursive cleaning)."""
        return {
            "__class__": "InputAdapter",
            "take_over_vessel": self.take_over_vessel,
            "bay_slots_detail": clean_for_json(self.bay_slots_detail),
            "tops_plan": clean_for_json(self.tops_plan),
            "area_function_info": clean_for_json(self.area_function_info),
            "vessel_berth_info": clean_for_json(self.vessel_berth_info),
            "planning_time": self.planning_time,  # Handled by SafeJSONEncoder
            "history_plan_info": clean_for_json(self.history_plan_info),
            "vessel_containers": clean_for_json(self.vessel_containers),
            "closed_area": list(self.closed_area),
            "berth_area_dist_matrix": clean_for_json(self.berth_area_dist_matrix),
            "voyage_predict": clean_for_json(self.voyage_predict),
            "large_plan": clean_for_json(self.large_plan),
            "adjust_plan_info": clean_for_json(self.adjust_plan_info),
            "user_design": self.user_design,
            "user_design_large_plan_area": clean_for_json(self.user_design_large_plan_area),
            "is_data_local": self.is_data_local,
            "local_path": self.local_path,
            "need_save_data": self.need_save_data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InputAdapter":
        """Reconstruct object from dictionary (with DataFrame restoration)."""
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
                    new_content[key] = value  # keep as dict/list/scalar
            vessel_containers_restored[vid] = new_content
        obj.vessel_containers = vessel_containers_restored

        obj.closed_area = set(data.get("closed_area", []))
        obj.berth_area_dist_matrix = restore_dataframe_from_split(data.get("berth_area_dist_matrix"))
        obj.voyage_predict = data.get("voyage_predict", {})
        obj.large_plan = data.get("large_plan", {})
        obj.adjust_plan_info = data.get("adjust_plan_info", {})
        obj.user_design = bool(data.get("user_design", obj.user_design))
        obj.user_design_large_plan_area = data.get("user_design_large_plan_area", obj.user_design_large_plan_area)
        obj.is_data_local = data.get("is_data_local", False)
        obj.local_path = data.get("local_path")
        obj.need_save_data = data.get("need_save_data", True)

        return obj

    def save_to_json(self, filepath: str):
        """Save to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                self.to_dict(),
                f,
                cls=SafeJSONEncoder,
                indent=2,
                ensure_ascii=False
            )

    @classmethod
    def load_from_json(cls, filepath: str) -> "InputAdapter":
        """Load from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

if __name__ == "__main__":
    json_path = ""
    data = InputAdapterGd.load_from_json(json_path)
