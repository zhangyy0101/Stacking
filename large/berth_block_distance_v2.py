import ast
import re
from pathlib import Path

import pandas as pd

from planning_large_main import discover_data_dir, resolve_output_path


# =========================
# 1. 参数区：按实际情况调整
# =========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = discover_data_dir(BASE_DIR, None)

EXCEL_PATH = DATA_DIR / "箱区功能.xlsx"
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
# A0 箱区的右边界为 10.0。
SHORELINE_LEFT_X = 0.0
SHORELINE_RIGHT_X = 10.0

# 10 个垂直于岸线方向的箱区集合（竖直通道），从左到右
CHANNELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A"]

# 通道/箱区列中心线。10 的中心为 0.5，20 为 1.5，A0 为 9.5。
CHANNEL_X = {ch: i + 0.5 for i, ch in enumerate(CHANNELS)}

# 普通箱区的纵向层级。数值越大，越远离岸线。
# 这些值是根据蓝色示意图的相对位置给的近似值，可以后续用实测道路距离校准。
# 当前版本将最靠近岸线的一排 10、20、30、...、A0 设为 y=0。
# 因此：
#   10、20、30、...、A0 的纵坐标为 0；
#   18、28、38、...、A8 不再作为 y=0，而是回到更远离岸线的位置。
#
# 注意：
#   1. L/M/N/P 是普通主通道向下延伸的层级，比如 2L、2M、2N、2P；
#   2. 图片中 2 通道没有 2Q、2R、2S，因此 Q/R/S 不放入普通 ROW_Y；
#   3. 5Q、5R、5S 属于图中可见的独立小箱区，放在 SPECIAL_AREA_COORDS 中单独处理。
ROW_Y = {
    "0": 0.00,
    "1": 0.25,
    "2": 0.50,
    "3": 0.75,
    "4": 1.00,
    "5": 1.15,
    "6": 1.30,
    "7": 1.40,

    # T 位于 17/27/... 与 18/28/... 之间，按示意图给一个中间近似值。
    "T": 1.43,

    "8": 1.45,
    "9": 1.70,
    "A": 1.95,
    "B": 2.30,
    "C": 2.55,
    "D": 2.90,

    # X 位于 D 与 E 之间，用于 1X、2X、... 等箱区。
    "X": 3.20,

    "E": 3.55,
    "F": 3.85,
    "G": 4.15,
    "H": 4.45,
    "J": 4.75,
    "K": 5.05,

    # L/M/N/P 是图片下半部分继续向下延伸的普通层级。
    # 例如 2L、2M、2N、2P。
    "L": 5.45,
    "M": 5.80,
    "N": 6.15,
    "P": 6.50,
}

# 下方 E1~EE 是两组横向箱区，不属于上方 1~A 的垂直通道列。
# 这里使用与 CHANNEL_X 相同坐标系下、从俯视图读取的组中心近似值。
E_GROUP_X = {
    "left": 1.20,   # E1~E7
    "right": 3.20,  # E8~EE
}

E_ROW_Y = {
    # E1~E7 位于左下方横向箱区组，整体在 2P/5P 等箱区下方。
    # 因此 E1 的 y 必须大于 P 层级的 6.50。
    "1": 7.05,
    "2": 7.35,
    "3": 7.65,
    "4": 7.95,
    "5": 8.25,
    "6": 8.55,
    "7": 8.85,

    # E8~EE 位于中下方横向箱区组，与 E1~E7 大致对应同一组纵向层级。
    "8": 7.05,
    "9": 7.35,
    "A": 7.65,
    "B": 7.95,
    "C": 8.25,
    "D": 8.55,
    "E": 8.85,
}


IGNORED_AREAS = {"C7"}


def evenly_spaced_values(start, end, count):
    """在两个纵坐标之间生成 count 个等间距数值"""
    if count == 1:
        return [round(start, 6)]

    step = (end - start) / (count - 1)
    return [round(start + i * step, 6) for i in range(count)]


def build_vertical_area_coords(area_names, x, start_y, end_y):
    """把一组箱区编号映射成坐标"""
    return {
        area: (x, y)
        for area, y in zip(area_names, evenly_spaced_values(start_y, end_y, len(area_names)))
    }


def midpoint(a, b):
    return round((a + b) / 2, 6)


XN_COORD = (10.25, 2.45)

X_GROUP_COORDS = build_vertical_area_coords(
    ["X1", "X2", "X3", "X4", "X5", "X6", "X7"],
    -0.75,
    E_ROW_Y["2"],
    E_ROW_Y["5"],
)

W_LEFT_GROUP_COORDS = build_vertical_area_coords(
    ["W5", "W6", "W7", "W8", "W9", "WA", "WB"],
    0.85,
    ROW_Y["L"],
    ROW_Y["P"],
)

S_GROUP_COORDS = build_vertical_area_coords(
    ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"],
    10.65,
    (ROW_Y["0"] + ROW_Y["1"]) / 2,
    (ROW_Y["A"] + ROW_Y["B"]) / 2,
)

D_GROUP_COORDS = {
    "DK": (CHANNEL_X["6"], E_ROW_Y["8"]),
    "DL": (CHANNEL_X["6"], E_ROW_Y["9"]),
    "DM": (CHANNEL_X["6"], E_ROW_Y["A"]),
    "DN": (CHANNEL_X["6"], E_ROW_Y["B"]),
    "DU": (CHANNEL_X["6"], E_ROW_Y["C"]),
    "DV": (CHANNEL_X["6"], E_ROW_Y["D"]),
    "DW": (CHANNEL_X["6"], E_ROW_Y["E"]),
    "DX": (CHANNEL_X["6"], round(E_ROW_Y["E"] + (E_ROW_Y["E"] - E_ROW_Y["D"]), 6)),

    "DF": (CHANNEL_X["7"], E_ROW_Y["8"]),
    "DG": (CHANNEL_X["7"], E_ROW_Y["9"]),
    "DH": (CHANNEL_X["7"], E_ROW_Y["A"]),
    "DJ": (CHANNEL_X["7"], E_ROW_Y["B"]),
    "DP": (CHANNEL_X["7"], E_ROW_Y["C"]),
    "DQ": (CHANNEL_X["7"], E_ROW_Y["D"]),
    "DR": (CHANNEL_X["7"], E_ROW_Y["E"]),
    "DS": (CHANNEL_X["7"], round(E_ROW_Y["E"] + (E_ROW_Y["E"] - E_ROW_Y["D"]), 6)),

    "DB": (CHANNEL_X["8"], E_ROW_Y["8"]),
    "DC": (CHANNEL_X["8"], E_ROW_Y["9"]),
    "DD": (CHANNEL_X["8"], E_ROW_Y["A"]),
    "DE": (CHANNEL_X["8"], E_ROW_Y["B"]),

    "D7": (CHANNEL_X["9"], E_ROW_Y["8"]),
    "D8": (CHANNEL_X["9"], E_ROW_Y["9"]),
    "D9": (CHANNEL_X["9"], E_ROW_Y["A"]),
    "DA": (CHANNEL_X["9"], E_ROW_Y["B"]),
}

FIVE_SMALL_AREA_COORDS = build_vertical_area_coords(
    ["5N", "5M", "5S", "5R", "5Q", "5P", "5W"],
    4.80,
    ROW_Y["L"],
    D_GROUP_COORDS["DX"][1],
)

RIGHT_W_GROUP_COORDS = {
    "W1": (CHANNEL_X["A"], ROW_Y["E"]),
    "W2": (CHANNEL_X["A"], ROW_Y["G"]),
    "W3": (CHANNEL_X["A"], ROW_Y["H"]),
    "W4": (CHANNEL_X["A"], ROW_Y["J"]),
    "WG": (CHANNEL_X["A"], ROW_Y["K"]),
}

D4_D6_X = midpoint(CHANNEL_X["9"], CHANNEL_X["A"])

C_GROUP_COORDS = {
    "C1": (CHANNEL_X["A"], FIVE_SMALL_AREA_COORDS["5M"][1]),
    "C2": (
        CHANNEL_X["A"],
        midpoint(FIVE_SMALL_AREA_COORDS["5M"][1], FIVE_SMALL_AREA_COORDS["5S"][1]),
    ),
    "C3": (
        midpoint(CHANNEL_X["A"], XN_COORD[0]),
        FIVE_SMALL_AREA_COORDS["5M"][1],
    ),
    "C4": (D4_D6_X, FIVE_SMALL_AREA_COORDS["5M"][1]),
    "C5": (D4_D6_X, FIVE_SMALL_AREA_COORDS["5N"][1]),
    "C6": (
        midpoint(CHANNEL_X["A"], D4_D6_X),
        FIVE_SMALL_AREA_COORDS["5M"][1],
    ),
}

D4_D6_COORDS = {
    "D4": (D4_D6_X, D_GROUP_COORDS["DN"][1]),
    "D5": (D4_D6_X, D_GROUP_COORDS["DU"][1]),
    "D6": (D4_D6_X, D_GROUP_COORDS["DV"][1]),
}

Y_GROUP_COORDS = {
    "Y1": (8.00, midpoint(D_GROUP_COORDS["DV"][1], D_GROUP_COORDS["DW"][1])),
    "Y3": (7.00, midpoint(D_GROUP_COORDS["DV"][1], D_GROUP_COORDS["DW"][1])),
    "Y4": (7.00, D_GROUP_COORDS["DX"][1]),
    "Y6": (8.00, D_GROUP_COORDS["DX"][1]),
}

OTHER_SMALL_AREA_COORDS = {
    "4P": (3.20, midpoint(ROW_Y["P"], E_ROW_Y["1"])),
    "7L": (6.55, C_GROUP_COORDS["C5"][1]),
    "7M": (6.55, C_GROUP_COORDS["C4"][1]),
    "7N": (6.00, midpoint(C_GROUP_COORDS["C5"][1], C_GROUP_COORDS["C4"][1])),
    "9L": (8.55, 5.75),
}

R_GROUP_COORDS = {
    "R1": (2.70, midpoint(ROW_Y["N"], ROW_Y["M"])),
    "R2": (2.70, midpoint(ROW_Y["M"], ROW_Y["L"])),
    "R3": (3.00, midpoint(ROW_Y["N"], ROW_Y["P"])),
    "R4": (2.00, midpoint(ROW_Y["L"], ROW_Y["M"])),
}


# 图片中还存在一些不属于 1~A 主通道规则的特殊箱区。
# 这些箱区不能用“首字符=横向通道、第二字符=纵向层级”的规则解析，
# 因此在这里直接给出从示意图估计的中心点坐标。
# 这些坐标是第一版近似值，后续可以按实测距离或更精确图纸继续校准。
SPECIAL_AREA_COORDS = {
    # 左侧 F 组
    "F9": (-0.60, 1.70),
    "FA": (-0.60, 1.95),
    "FB": (-0.60, 2.30),
    "FC": (-0.60, 2.55),
    "FD": (-0.60, 2.90),
    "FE": (-1.15, 3.55),
    "FF": (-1.15, 3.85),
    "FG": (-1.15, 4.15),
    "FH": (-1.15, 4.45),
    "FJ": (-1.15, 4.75),
    "FK": (-1.15, 5.05),

    # 左下 X 组
    **X_GROUP_COORDS,
    "XN": XN_COORD,

    # 左下与右侧 W 组
    **W_LEFT_GROUP_COORDS,
    **RIGHT_W_GROUP_COORDS,

    # 右上 S 组
    **S_GROUP_COORDS,

    # 右侧 C 组
    **C_GROUP_COORDS,

    # 中下部与右下部 D 组
    **D_GROUP_COORDS,
    **D4_D6_COORDS,

    # 底部右侧 Y 组
    **Y_GROUP_COORDS,

    # 5M、5N、5P、5Q、5R、5S 在图中是独立小箱区，
    # 不按普通“5 通道 + 后缀层级”规则处理。
    **FIVE_SMALL_AREA_COORDS,

    # 其他图中可见的独立小箱区
    **OTHER_SMALL_AREA_COORDS,

    # 独立小箱区
    **R_GROUP_COORDS,
    "BU": (10.20, ROW_Y["2"]),
    "J2": (10.50, -0.20),
}


EXPLICIT_AREA_ORDER = (
    [f"E{i}" for i in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E"]]
    + list(SPECIAL_AREA_COORDS)
)


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

    # 优先处理图片中补充的特殊箱区。
    if area_no in SPECIAL_AREA_COORDS:
        return SPECIAL_AREA_COORDS[area_no]

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
# 5. 读取箱区，并过滤不存在的箱区
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


def load_areas(excel_path, closed_areas=None):
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

    for area in EXPLICIT_AREA_ORDER:
        if area not in areas:
            areas.append(area)

    closed_areas = {
        area
        for area in (normalize_area_no(value) for value in (closed_areas or []))
        if area
    }

    # 距离矩阵要覆盖箱区功能表里的所有箱区；关闭箱区只记录在单独 sheet 中，
    # 不从距离矩阵里剔除。这里只过滤确认不存在的箱区。
    excluded_areas = IGNORED_AREAS

    return [area for area in areas if area not in excluded_areas]


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
    areas = load_areas(EXCEL_PATH, closed_areas=closed_areas)

    berth_df = build_berths()
    area_coord_df = build_area_coord_table(areas)
    long_df, matrix_df = build_distance_tables(area_coord_df, berth_df)

    closed_area_df = pd.DataFrame({"area_no": sorted(closed_areas)})
    output_path = resolve_output_path(Path("适放箱区_泊位距离矩阵.xlsx"), BASE_DIR)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        berth_df.to_excel(writer, sheet_name="泊位坐标", index=False)
        area_coord_df.to_excel(writer, sheet_name="箱区坐标", index=False)
        closed_area_df.to_excel(writer, sheet_name="关闭箱区", index=False)
        matrix_df.to_excel(writer, sheet_name="距离矩阵", index=False)
        long_df.to_excel(writer, sheet_name="距离长表", index=False)

    print(f"已输出：{output_path.resolve()}")
    print(f"已记录关闭箱区（距离矩阵不剔除）：{', '.join(sorted(closed_areas))}")
    print("\n箱区坐标示例：")
    print(area_coord_df.head())

    print("\n距离矩阵示例：")
    print(matrix_df.head())


if __name__ == "__main__":
    main()
