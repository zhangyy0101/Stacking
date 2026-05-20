from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from flat_yard_plan_data_io import find_area_function_file, normalize_code

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover - fallback keeps the script usable without scipy.
    ndimage = None


DEFAULT_BASE_IMAGE = Path("箱区俯视示意图（原始）.png")
DEFAULT_ALLOCATION = Path("outputs_large/latest_run/allocation.csv")
DEFAULT_OUTPUT_DIR = Path("outputs_large/yard_visualization")
ANNOTATION_Y_OFFSET = -10


def project_root_from(base_dir: Path) -> Path:
    for candidate in [base_dir, *base_dir.parents]:
        if (candidate / "large").is_dir() or (candidate / "medium_small").is_dir():
            return candidate
    return base_dir


def resolve_input_path(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def resolve_output_path(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def resolve_base_image(path: Path, base_dir: Path | None = None) -> Path:
    """
    功能：
        确定箱区俯视示意图底图路径。

    参数：
        path: 用户指定或默认的底图路径。

    返回：
        可读取的底图路径。

    异常：
        FileNotFoundError: 指定路径和自动候选路径均不存在时抛出。
    """

    base_dir = base_dir or Path(__file__).resolve().parent
    resolved = resolve_input_path(path, base_dir)
    if resolved.exists():
        return resolved

    search_roots: list[Path] = []
    for root in [Path.cwd().resolve(), base_dir.resolve(), project_root_from(base_dir)]:
        if root.is_dir() and root not in search_roots:
            search_roots.append(root)
    candidates = sorted(
        [candidate for root in search_roots for candidate in root.glob("*.png")],
        key=lambda p: (0 if "原始" in p.name else 1, p.name),
    )
    for candidate in candidates:
        if "示意" in candidate.name:
            return candidate
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Cannot find base image: {path}")


def build_area_positions() -> dict[str, tuple[int, int]]:
    """
    功能：
        构造箱区编号到图上标注坐标的映射。

    参数：
        无。

    返回：
        键为箱区号、值为 ``(x, y)`` 标注锚点的字典。

    说明：
        这些坐标基于当前 ``箱区俯视示意图.png`` 的版式人工模板化整理。
        若底图比例或排版变化，只需要调整这里的少量坐标模板。
    """

    positions: dict[str, tuple[int, int]] = {}

    # 主体 1/2/3/4/5/6/7/8/9/A 组。坐标取在每条箱区横排左侧偏内位置。
    group_x = {
        "1": 410,
        "2": 620,
        "3": 850,
        "4": 1080,
        "5": 1320,
        "6": 1590,
        "7": 1850,
        "8": 2120,
        "9": 2385,
        "A": 2635,
    }
    upper_y = {
        "0": 438,
        "1": 470,
        "2": 497,
        "3": 523,
        "4": 550,
        "5": 577,
        "6": 604,
        "7": 630,
    }
    lower_y = {
        "8": 690,
        "9": 724,
        "A": 752,
        "B": 780,
        "C": 812,
        "D": 846,
    }
    for prefix, x in group_x.items():
        for suffix, y in upper_y.items():
            positions[f"{prefix}{suffix}"] = (x, y)
        for suffix, y in lower_y.items():
            positions[f"{prefix}{suffix}"] = (x, y)

    # 个别在下方延展的箱区。
    positions.update(
        {
            "1E": (410, 924),
            "1H": (410, 978),
            "1J": (410, 1032),
            "1K": (410, 1060),
            "2E": (620, 924),
            "2H": (620, 978),
            "2J": (620, 1032),
            "2K": (620, 1060),
            "4E": (1080, 916),
            "4F": (1080, 944),
            "4G": (1080, 978),
            "4H": (1080, 1004),
            "4J": (1080, 1034),
            "4K": (1080, 1064),
            "5E": (1320, 916),
            "5F": (1320, 944),
            "5G": (1320, 978),
            "5H": (1320, 1004),
            "5J": (1320, 1034),
            "5K": (1320, 1064),
            "6E": (1590, 924),
            "6G": (1590, 978),
            "6H": (1590, 1004),
            "6J": (1590, 1034),
            "6K": (1590, 1064),
            "7E": (1850, 924),
            "7G": (1850, 978),
            "7H": (1850, 1004),
            "7J": (1850, 1034),
            "7K": (1850, 1064),
            "8E": (2120, 924),
            "8G": (2120, 978),
            "8H": (2120, 1004),
            "8J": (2120, 1034),
            "8K": (2120, 1064),
            "9E": (2385, 924),
            "9H": (2385, 978),
            "9K": (2385, 1034),
            "AB": (2635, 780),
            "AC": (2635, 812),
            "AD": (2635, 846),
        }
    )

    # E 组长条箱区。
    for idx, area in enumerate(["E1", "E2", "E3", "E4", "E5", "E6", "E7"]):
        positions[area] = (430, 1328 + idx * 53)
    for idx, area in enumerate(["E8", "E9", "EA", "EB", "EC", "ED", "EE"]):
        positions[area] = (920, 1328 + idx * 53)

    return positions


def remove_hand_drawn_red_marks(image: Image.Image, *, min_component_area: int = 180) -> Image.Image:
    """
    功能：
        从底图中清理手画红色勾选痕迹，生成空白底图。

    参数：
        image: 原始 RGBA/RGB 底图。
        min_component_area: 需要清理的红色连通块最小面积；小于该面积的小红圈会保留。

    返回：
        清理后的底图。
    """

    rgba = image.convert("RGBA")
    arr = np.asarray(rgba).copy()
    rgb = arr[:, :, :3].astype(np.int16)
    red_mask = (
        (rgb[:, :, 0] > 145)
        & (rgb[:, :, 1] < 155)
        & (rgb[:, :, 2] < 170)
        & ((rgb[:, :, 0] - np.maximum(rgb[:, :, 1], rgb[:, :, 2])) > 35)
    )
    if not red_mask.any():
        return rgba

    if ndimage is not None:
        labels, count = ndimage.label(red_mask)
        clean_mask = np.zeros(red_mask.shape, dtype=bool)
        objects = ndimage.find_objects(labels)
        for label_index, slc in enumerate(objects, start=1):
            if slc is None:
                continue
            component = labels[slc] == label_index
            area = int(component.sum())
            height = slc[0].stop - slc[0].start
            width = slc[1].stop - slc[1].start
            if area >= min_component_area and width >= 8 and height >= 8:
                clean_mask[slc] |= component
        clean_mask = ndimage.binary_dilation(clean_mask, iterations=2)
    else:
        clean_mask = red_mask

    if ndimage is not None and clean_mask.any():
        filled = arr.copy()
        _, nearest = ndimage.distance_transform_edt(clean_mask, return_indices=True)
        filled[clean_mask] = arr[tuple(index[clean_mask] for index in nearest)]
        return Image.fromarray(filled, mode="RGBA")

    median = rgba.filter(ImageFilter.MedianFilter(size=17))
    mask_img = Image.fromarray((clean_mask.astype(np.uint8) * 255), mode="L")
    result = rgba.copy()
    result.paste(median, mask=mask_img)
    return result


def read_allocation_summary(allocation_path: Path) -> dict[str, list[dict[str, Any]]]:
    """
    功能：
        读取求解结果并按箱区、航次汇总计划箱量。

    参数：
        allocation_path: ``allocation.csv`` 路径。

    返回：
        键为箱区号、值为航次计划列表的字典。

    异常：
        FileNotFoundError: 求解结果文件不存在时抛出。
        KeyError: 求解结果缺少必需列时抛出。
    """

    df = pd.read_csv(allocation_path, dtype={"voy_id": str, "area_no": str, "flow": str, "size": str})
    required = {"voy_id", "area_no", "planned_qty"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Allocation file is missing columns: {sorted(missing)}")
    df["planned_qty"] = pd.to_numeric(df["planned_qty"], errors="coerce").fillna(0.0)
    grouped = (
        df.groupby(["area_no", "voy_id"], as_index=False)["planned_qty"]
        .sum()
        .sort_values(["area_no", "voy_id"])
    )
    summary: dict[str, list[dict[str, Any]]] = {}
    for _, row in grouped.iterrows():
        qty = int(round(float(row["planned_qty"])))
        if qty <= 0:
            continue
        summary.setdefault(str(row["area_no"]), []).append(
            {"voy_id": str(row["voy_id"]), "planned_qty": qty}
        )
    return summary


def read_area_work_type_labels(base_dir: Path, data_dir: Path | None) -> dict[str, str]:
    """
    功能：
        从箱区功能 Excel 中读取每个箱区的作业类型标签。

    参数：
        base_dir: 当前脚本所在基础目录。
        data_dir: 业务数据目录；为 ``None`` 时自动发现默认数据目录。

    返回：
        键为箱区号、值为按 Excel 顺序拼接的作业类型标签，例如 ``OF/OZ/IF/IZ/T``。

    异常：
        FileNotFoundError: 数据目录或箱区功能表未找到时抛出。
        KeyError: 箱区功能表缺少必需列时抛出。
    """

    resolved_data_dir = data_dir.resolve() if data_dir else base_dir.resolve()
    area_file = find_area_function_file(resolved_data_dir)
    df = pd.read_excel(area_file).copy()
    required = {"area_no", "cntr_type"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Area function file is missing columns: {sorted(missing)}")

    df["area_no"] = df["area_no"].map(normalize_code)
    df = df[df["area_no"].notna()].drop_duplicates("area_no", keep="first")

    labels: dict[str, str] = {}
    for _, row in df.iterrows():
        area = row["area_no"]
        if not area:
            continue
        flows: list[str] = []
        for part in re.split(r"[,/;，、\s]+", str(row["cntr_type"])):
            flow = normalize_code(part)
            if flow and flow not in flows:
                flows.append(flow)
        labels[area] = "/".join(flows)
    return labels


def generate_yard_visualization(
    *,
    base_image: Path = DEFAULT_BASE_IMAGE,
    data_dir: Path | None = None,
    allocation: Path = DEFAULT_ALLOCATION,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    base_dir = base_dir or Path(__file__).resolve().parent
    base_image_path = resolve_base_image(base_image, base_dir)
    allocation_path = resolve_input_path(allocation, base_dir)
    resolved_output_dir = resolve_output_path(output_dir, base_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    blank = Image.open(base_image_path).convert("RGBA")
    allocation_by_area = read_allocation_summary(allocation_path)
    area_work_type_labels = read_area_work_type_labels(base_dir, data_dir)
    area_positions = build_area_positions()
    annotated, unmapped = draw_annotations(blank, allocation_by_area, area_positions, area_work_type_labels)

    blank_path = resolved_output_dir / "yard_base_original.png"
    annotated_path = resolved_output_dir / "yard_allocation_annotated.png"
    blank.save(blank_path)
    annotated.save(annotated_path)
    check_path = resolved_output_dir / "visualization_allocation_check.csv"
    write_visualization_check(check_path, allocation_by_area, area_positions, area_work_type_labels)
    unmapped_path = resolved_output_dir / "unmapped_areas.csv"
    if unmapped:
        write_unmapped_areas(unmapped_path, unmapped, allocation_by_area)
    elif unmapped_path.exists():
        unmapped_path.unlink()

    return {
        "base_image": str(base_image_path),
        "allocation": str(allocation_path),
        "output_dir": str(resolved_output_dir),
        "base_image_copy": str(blank_path),
        "annotated_image": str(annotated_path),
        "check_csv": str(check_path),
        "allocation_area_count": len(allocation_by_area),
        "area_function_label_count": len(area_work_type_labels),
        "unmapped_areas": unmapped,
    }


def label_fill_color(work_type_label: str) -> tuple[int, int, int, int]:
    """
    功能：
        根据作业类型标签返回箱区功能段的底色。

    参数：
        work_type_label: 箱区作业类型标签。

    返回：
        RGBA 颜色。
    """

    flows = set(work_type_label.split("/")) if work_type_label else set()
    if "IF" in flows:
        return (72, 164, 92, 242)
    if "OF" in flows:
        return (202, 150, 40, 242)
    if flows & {"OZ", "IZ"}:
        return (154, 174, 196, 242)
    if "T" in flows:
        return (196, 188, 176, 242)
    return (255, 255, 240, 0)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    功能：
        加载适合图上标注的字体。

    参数：
        size: 字号。

    返回：
        Pillow 字体对象。
    """

    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    功能：
        加载适合图上加粗标注的字体。

    参数：
        size: 字号。

    返回：
        Pillow 字体对象。
    """

    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_annotations(
    image: Image.Image,
    allocation_by_area: dict[str, list[dict[str, Any]]],
    area_positions: dict[str, tuple[int, int]],
    area_work_type_labels: dict[str, str],
) -> tuple[Image.Image, list[str]]:
    """
    功能：
        将航次-箱区计划箱量标注到空白箱区图上。

    参数：
        image: 已清理的空白底图。
        allocation_by_area: 按箱区汇总的计划结果。
        area_positions: 箱区号到图上坐标的映射。
        area_work_type_labels: 箱区功能标签映射。

    返回：
        二元组 ``(annotated_image, unmapped_areas)``，第二项记录缺少坐标的箱区。
    """

    annotated = image.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(13)
    bold_font = load_bold_font(13)
    unmapped: list[str] = []

    vessel_colors = {
        "453334": (255, 244, 140, 232),
        "453400": (156, 220, 255, 232),
        "453886": (185, 255, 185, 232),
        "454063": (255, 190, 220, 232),
    }
    border = (28, 36, 46, 230)

    for area, rows in sorted(allocation_by_area.items()):
        if area not in area_positions:
            unmapped.append(area)
            continue
        x, y = area_positions[area]
        y += ANNOTATION_Y_OFFSET
        work_type_label = area_work_type_labels.get(area, "")
        area_label = f"{area} {work_type_label}".strip()
        segments = [(area_label, label_fill_color(work_type_label), bold_font)] + [
            (
                f"{row['voy_id']}:{row['planned_qty']}",
                vessel_colors.get(row["voy_id"], (230, 230, 230, 232)),
                font,
            )
            for row in rows
        ]
        segment_bboxes = [draw.textbbox((0, 0), text, font=segment_font) for text, _, segment_font in segments]
        text_width = sum(bbox[2] - bbox[0] for bbox in segment_bboxes) + 5 * (len(segments) - 1)
        text_height = max(bbox[3] - bbox[1] for bbox in segment_bboxes)
        box_w = text_width + 10
        box_h = text_height + 8
        x2 = min(x + box_w, annotated.width - 4)
        y2 = min(y + box_h, annotated.height - 4)
        x1 = max(4, x2 - box_w)
        y1 = max(4, y2 - box_h)

        draw.rounded_rectangle((x1, y1, x2, y2), radius=4, fill=(255, 255, 240, 226), outline=border, width=1)
        xx = x1 + 5
        yy = y1 + 3
        for (text, fill, segment_font), bbox in zip(segments, segment_bboxes):
            segment_w = bbox[2] - bbox[0]
            if fill[3] > 0:
                draw.rounded_rectangle((xx - 2, yy, xx + segment_w + 2, yy + text_height + 2), radius=2, fill=fill)
            draw.text((xx, yy), text, fill=(0, 0, 0, 255), font=segment_font)
            xx += segment_w + 5

    combined = Image.alpha_composite(annotated, overlay)
    return combined, unmapped


def write_unmapped_areas(path: Path, unmapped: list[str], allocation_by_area: dict[str, list[dict[str, Any]]]) -> None:
    """
    功能：
        将缺少坐标模板的箱区写成 CSV，方便后续补坐标。

    参数：
        path: 输出 CSV 路径。
        unmapped: 缺少坐标的箱区列表。
        allocation_by_area: 按箱区汇总的计划结果。

    返回：
        无。
    """

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["area_no", "voy_id", "planned_qty"])
        for area in unmapped:
            for row in allocation_by_area.get(area, []):
                writer.writerow([area, row["voy_id"], row["planned_qty"]])


def write_visualization_check(
    path: Path,
    allocation_by_area: dict[str, list[dict[str, Any]]],
    area_positions: dict[str, tuple[int, int]],
    area_work_type_labels: dict[str, str],
) -> None:
    """
    Write the exact area/voyage quantities that the visualization is expected to display.
    """

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["area_no", "work_type", "has_position", "voy_id", "planned_qty"])
        for area in sorted(allocation_by_area):
            rows = allocation_by_area[area]
            for row in rows:
                writer.writerow(
                    [
                        area,
                        area_work_type_labels.get(area, ""),
                        int(area in area_positions),
                        row["voy_id"],
                        row["planned_qty"],
                    ]
                )


def parse_args() -> argparse.Namespace:
    """
    功能：
        解析命令行参数。

    参数：
        无。

    返回：
        argparse 命名空间。
    """

    parser = argparse.ArgumentParser(description="Generate a yard allocation map from the solved plan.")
    parser.add_argument("--base-image", type=Path, default=DEFAULT_BASE_IMAGE)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--allocation", type=Path, default=DEFAULT_ALLOCATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """
    功能：
        生成空白箱区示意图和带规划结果标注的箱区图。

    参数：
        无。

    返回：
        无。
    """

    args = parse_args()
    result = generate_yard_visualization(
        base_image=args.base_image,
        data_dir=args.data_dir,
        allocation=args.allocation,
        output_dir=args.output_dir,
        base_dir=Path(__file__).resolve().parent,
    )

    print(f"Base image: {result['base_image']}")
    print(f"Allocation rows grouped into areas: {result['allocation_area_count']}")
    print(f"Area function labels read: {result['area_function_label_count']}")
    print(f"Base image copy written to: {result['base_image_copy']}")
    print(f"Annotated image written to: {result['annotated_image']}")
    if result["unmapped_areas"]:
        print(f"Unmapped areas: {', '.join(result['unmapped_areas'])}")


if __name__ == "__main__":
    main()
