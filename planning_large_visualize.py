from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover - fallback keeps the script usable without scipy.
    ndimage = None


DEFAULT_BASE_IMAGE = Path("箱区俯视示意图（原始）.png")
DEFAULT_ALLOCATION = Path("outputs_large/latest_run/allocation.csv")
DEFAULT_OUTPUT_DIR = Path("outputs_large/yard_visualization")


def resolve_base_image(path: Path) -> Path:
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

    if path.exists():
        return path
    candidates = sorted(Path.cwd().glob("*.png"), key=lambda p: p.stat().st_size)
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


def draw_annotations(
    image: Image.Image,
    allocation_by_area: dict[str, list[dict[str, Any]]],
    area_positions: dict[str, tuple[int, int]],
) -> tuple[Image.Image, list[str]]:
    """
    功能：
        将航次-箱区计划箱量标注到空白箱区图上。

    参数：
        image: 已清理的空白底图。
        allocation_by_area: 按箱区汇总的计划结果。
        area_positions: 箱区号到图上坐标的映射。

    返回：
        二元组 ``(annotated_image, unmapped_areas)``，第二项记录缺少坐标的箱区。
    """

    annotated = image.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(13)
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
        segments = [(area, (255, 255, 240, 0))] + [
            (f"{row['voy_id']}:{row['planned_qty']}", vessel_colors.get(row["voy_id"], (230, 230, 230, 232)))
            for row in rows
        ]
        segment_bboxes = [draw.textbbox((0, 0), text, font=font) for text, _ in segments]
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
        for (text, fill), bbox in zip(segments, segment_bboxes):
            segment_w = bbox[2] - bbox[0]
            if fill[3] > 0:
                draw.rounded_rectangle((xx - 2, yy, xx + segment_w + 2, yy + text_height + 2), radius=2, fill=fill)
            draw.text((xx, yy), text, fill=(0, 0, 0, 255), font=font)
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
    base_image_path = resolve_base_image(args.base_image)
    allocation_path = args.allocation
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    blank = Image.open(base_image_path).convert("RGBA")
    allocation_by_area = read_allocation_summary(allocation_path)
    annotated, unmapped = draw_annotations(blank, allocation_by_area, build_area_positions())

    blank_path = output_dir / "yard_base_original.png"
    annotated_path = output_dir / "yard_allocation_annotated.png"
    blank.save(blank_path)
    annotated.save(annotated_path)
    if unmapped:
        write_unmapped_areas(output_dir / "unmapped_areas.csv", unmapped, allocation_by_area)

    print(f"Base image: {base_image_path}")
    print(f"Allocation rows grouped into areas: {len(allocation_by_area)}")
    print(f"Base image copy written to: {blank_path}")
    print(f"Annotated image written to: {annotated_path}")
    if unmapped:
        print(f"Unmapped areas: {', '.join(unmapped)}")


if __name__ == "__main__":
    main()
