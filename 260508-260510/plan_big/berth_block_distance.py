import ast
import re
from pathlib import Path

import pandas as pd


# =========================
# 1. 参数区：按实际情况调整
# =========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "堆存计划测试数据20260508"

EXCEL_PATH = DATA_DIR / "of适放箱区列表.xlsx"
CLOSED_AREAS_PATH = DATA_DIR / "n_usefg_areas.txt"
DEFAULT_CLOSED_AREAS = {"20", "25", "4F"}   # 关闭的箱区

# 如果只想得到“归一化距离”，保持 1 即可。
# 如果要换算成米，可以把 X_UNIT_M 设置为一个通道宽度对应的米数；
# 由于希望先横向再纵向选择箱区，所以将Y_UNIT_M设置为一个较大的数，比如1000。
X_UNIT_M = 1.0
Y_UNIT_M = 1000.0

N_CHANNELS = 10 # 通道数量
N_BERTHS = 7    # 泊位数量

# 岸线从 10 箱区最左边到 A0 箱区最右边。
# 10 的中心为 x=0.5，因此其左边界为 0.0；
# A0箱区的右边界为 10.0。
SHORELINE_LEFT_X = 0.0
SHORELINE_RIGHT_X = 10.0

# 10 个垂直于岸线方向的箱区集合（竖直通道），从左到右
CHANNELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A"]

# 通道/箱区列中心线。10 的中心为 0.5，20 为 1.5，A0 为 9.5。
CHANNEL_X = {ch: i + 0.5 for i, ch in enumerate(CHANNELS)}

# 普通箱区的纵向层级。数值越大，越远离岸线。
# 这些值是根据蓝色示意图的相对位置给的近似值，可以后续用实测道路距离校准。
ROW_Y = {
    "8": 1.45,
    "9": 1.70,
    "A": 1.95,
    "B": 2.30,
    "C": 2.55,
    "D": 2.90,
    "E": 3.55,
    "F": 3.85,
    "G": 4.15,
    "H": 4.45,
    "J": 4.75,
    "K": 5.05,
}

# TODO:所有的ROW_Y在当前的基础上减去1.45，使得8层的纵坐标为0，防止横向距离和纵向距离计算失效
ROW_Y = {
    "8": 0.0,
    "9": 0.25,
    "A": 0.50,
    "B": 0.85,
    "C": 1.10,
    "D": 1.45,
    "E": 2.10,
    "F": 2.40,
    "G": 2.70,
    "H": 3.00,
    "J": 3.30,
    "K": 3.60,
}

# 下方 E1~EE 是两组横向箱区，不属于上方 1~A 的垂直通道列。
# 这里使用与 CHANNEL_X 相同坐标系下、从俯视图读取的组中心近似值。
E_GROUP_X = {
    "left": 1.20,   # E1~E7
    "right": 3.20,  # E8~EE
}

E_ROW_Y = {
    "1": 5.45,
    "2": 5.80,
    "3": 6.15,
    "4": 6.50,
    "5": 6.85,
    "6": 7.20,
    "7": 7.55,
    "8": 5.45,
    "9": 5.80,
    "A": 6.15,
    "B": 6.50,
    "C": 6.85,
    "D": 7.20,
    "E": 7.55,
}


# =========================
# 2. 泊位坐标
# =========================

def build_berths():
    """
    7 个泊位在 10 左边界到 A0 右边界之间均匀分布。
    返回的是每个泊位中心点坐标；B1 到 x=0 的距离与 B7 到 x=10 的距离相同。
    """
    berth_records = []
    berth_spacing = (SHORELINE_RIGHT_X - SHORELINE_LEFT_X) / N_BERTHS

    for i in range(N_BERTHS):
        berth_no = f"B{i + 1}"
        berth_x = SHORELINE_LEFT_X + (i + 0.5) * berth_spacing

        berth_records.append({
            "berth": berth_no,
            "berth_x": berth_x,
            "berth_y": 0.0,
        })

    return pd.DataFrame(berth_records)

# =========================
# 3. 箱区编号清洗
# =========================

def normalize_area_no(value):
    """
    把 Excel 中的箱区编号统一成字符串：
    19 -> "19"
    1A -> "1A"
    AB -> "AB"
    """
    if pd.isna(value):
        return None

    s = str(value).strip().upper()

    # 处理 pandas 可能读成 "19.0" 的情况
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]

    return s


# =========================
# 4. 箱区坐标转换
# =========================

def area_to_coord(area_no):
    """
    输入箱区编号，输出箱区中心点坐标。
    坐标单位是归一化单位；距离换算由 X_UNIT_M / Y_UNIT_M 控制。
    """
    area_no = normalize_area_no(area_no)

    if not area_no:
        raise ValueError("空箱区编号")

    # 特殊处理 E1~EE，返回对应坐标
    if area_no.startswith("E") and len(area_no) == 2:
        suffix = area_no[1] # 取第二个字符

        if suffix not in E_ROW_Y:
            raise ValueError(f"无法识别 E 区箱区编号：{area_no}")

        if suffix in ["1", "2", "3", "4", "5", "6", "7"]:
            x = E_GROUP_X["left"]
        else:
            x = E_GROUP_X["right"]

        y = E_ROW_Y[suffix]
        return x, y

    # 普通箱区，例如 19、28、4B、7D、AB
    if len(area_no) != 2:
        raise ValueError(f"箱区编号格式异常：{area_no}")

    channel = area_no[0]
    row = area_no[1]

    if channel not in CHANNEL_X:
        raise ValueError(f"无法识别通道：{area_no}")

    if row not in ROW_Y:
        raise ValueError(f"无法识别箱区纵向层级：{area_no}")

    x = CHANNEL_X[channel]
    y = ROW_Y[row]

    return x, y


# =========================
# 5. 读取 OF 适放箱区，并过滤关闭箱区
# =========================

def load_closed_areas(closed_areas_path): 
    closed_areas = set(DEFAULT_CLOSED_AREAS)

    if closed_areas_path.exists():
        text = closed_areas_path.read_text(encoding="utf-8").strip()

        if text:
            try:
                raw_values = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                raw_values = re.split(r"[\s,，;；]+", text)

            if isinstance(raw_values, (str, int, float)):
                raw_values = [raw_values]

            closed_areas.update(
                area
                for area in (normalize_area_no(value) for value in raw_values)
                if area
            )

    return closed_areas


def load_of_areas(excel_path, closed_areas=None):
    df = pd.read_excel(excel_path)

    if "area_no" in df.columns:
        col = "area_no"
    else:
        col = df.columns[0]

    areas = (
        df[col]
        .map(normalize_area_no)
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    closed_areas = {
        area
        for area in (normalize_area_no(value) for value in (closed_areas or []))
        if area
    }

    return [area for area in areas if area not in closed_areas]


# =========================
# 6. 计算泊位到箱区曼哈顿距离
# =========================

def manhattan_distance(ax, ay, bx, by):
    """
    曼哈顿距离：
    |x1-x2| * X_UNIT_M + |y1-y2| * Y_UNIT_M
    """
    return abs(ax - bx) * X_UNIT_M + abs(ay - by) * Y_UNIT_M


def build_area_coord_table(areas):
    records = []

    for area in areas:
        x, y = area_to_coord(area)
        records.append({
            "area_no": area,
            "area_x": x,
            "area_y": y,
        })

    return pd.DataFrame(records)


def build_distance_tables(area_coord_df, berth_df):
    """area_coord_df：箱区坐标表格，包含area_no, area_x, area_y。
    berth_df：泊位坐标表格，包含berth, berth_x, berth_y。"""
    long_records = []

    for _, area in area_coord_df.iterrows():
        for _, berth in berth_df.iterrows():
            d = manhattan_distance(
                area["area_x"],
                area["area_y"],
                berth["berth_x"],
                berth["berth_y"],
            )

            long_records.append({
                "area_no": area["area_no"],
                "area_x": area["area_x"],
                "area_y": area["area_y"],
                "berth": berth["berth"],
                "berth_x": berth["berth_x"],
                "berth_y": berth["berth_y"],
                "manhattan_distance": d,
            })

    long_df = pd.DataFrame(long_records)

    # 把长表转换成矩阵表
    matrix_df = (
        long_df
        .pivot(index="area_no", columns="berth", values="manhattan_distance")
        .reset_index()
    )

    # 保持 Excel 原始箱区顺序
    order = area_coord_df["area_no"].tolist()
    matrix_df["__order"] = matrix_df["area_no"].map({a: i for i, a in enumerate(order)})
    matrix_df = matrix_df.sort_values("__order").drop(columns="__order")

    berth_cols = [c for c in matrix_df.columns if c.startswith("B")]

    # 为每个箱区寻找最近泊位和最近的距离
    matrix_df["nearest_berth"] = matrix_df[berth_cols].idxmin(axis=1)
    matrix_df["nearest_distance"] = matrix_df[berth_cols].min(axis=1)

    return long_df, matrix_df


# =========================
# 7. 主程序
# =========================

def main():
    closed_areas = load_closed_areas(CLOSED_AREAS_PATH)
    areas = load_of_areas(EXCEL_PATH, closed_areas=closed_areas)

    berth_df = build_berths()
    area_coord_df = build_area_coord_table(areas)
    long_df, matrix_df = build_distance_tables(area_coord_df, berth_df)

    closed_area_df = pd.DataFrame({"area_no": sorted(closed_areas)})
    output_path = BASE_DIR / "of_适放箱区_泊位距离矩阵.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        berth_df.to_excel(writer, sheet_name="泊位坐标", index=False)
        area_coord_df.to_excel(writer, sheet_name="箱区坐标", index=False)
        closed_area_df.to_excel(writer, sheet_name="关闭箱区", index=False)
        matrix_df.to_excel(writer, sheet_name="距离矩阵", index=False)
        long_df.to_excel(writer, sheet_name="距离长表", index=False)

    print(f"已输出：{output_path.resolve()}")
    print(f"已过滤关闭箱区：{', '.join(sorted(closed_areas))}")
    print("\n箱区坐标示例：")
    print(area_coord_df.head())

    print("\n距离矩阵示例：")
    print(matrix_df.head())


if __name__ == "__main__":
    main()
