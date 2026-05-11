from __future__ import annotations

import csv
import math
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

from .models import AreaOperation, Bay, BigPlanRow, BoxGroup, ProblemData, Unit, VoyageSchedule


DEFAULT_PLANNING_TIME = datetime(2026, 5, 8, 9, 30)


def _norm(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = text[:-2]
    return text or default


def _voyage(value: object, fallback: str = "") -> str:
    text = _norm(value, fallback)
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def parse_datetime(value: object) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def _read_one_column_xlsx(path: Path, header: str = "area_no") -> set[str]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ElementTree.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))
        sheet = ElementTree.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    values: list[str] = []
    for cell in sheet.findall(".//a:sheetData/a:row/a:c", ns):
        raw = cell.find("a:v", ns)
        if raw is None:
            continue
        value = raw.text or ""
        if cell.attrib.get("t") == "s":
            value = shared[int(value)]
        values.append(value.strip())
    return {v for v in values if v and v != header}


def read_closed_areas(data_dir: str | Path) -> set[str]:
    path = Path(data_dir) / "n_usefg_areas.txt"
    text = path.read_text(encoding="utf-8").strip()
    return set(re.findall(r"[A-Za-z0-9]+", text))


def read_export_areas(data_dir: str | Path) -> set[str]:
    # 这个文件是出口 OF 适放箱区列表。不同环境下中文文件名可能出现编码差异，
    # 因此优先按关键词自动查找，而不是依赖一个硬编码文件名。
    data_path = Path(data_dir)
    candidates = [
        path for path in data_path.glob("*.xlsx")
        if not path.name.startswith("~$") and path.name.lower().startswith("of")
    ]
    if not candidates:
        raise FileNotFoundError(f"No OF area xlsx file found under {data_path}")
    return _read_one_column_xlsx(candidates[0])


def read_vessel_schedules(data_dir: str | Path) -> dict[str, VoyageSchedule]:
    # vessel_berth_info.csv can support several planning decisions. In the current
    # medium/small solver, area-level choices are fixed by the supplied big plan,
    # so the schedule fields are currently consumed for capacity timing: whether
    # containers already in the yard still occupy capacity at the planning timestamp.
    frame = pd.read_csv(Path(data_dir) / "vessel_berth_info.csv")
    schedules: dict[str, VoyageSchedule] = {}
    for row in frame.to_dict("records"):
        if _norm(row.get("VOY_IEFG")) != "E":
            continue
        voyage_id = _voyage(row.get("VOY_ID"))
        receive_start = parse_datetime(row.get("SCD_RCVSTDT"))
        receive_end = parse_datetime(row.get("SCD_RCVEDDT"))
        berth_time = parse_datetime(row.get("VBT_ABTHDT")) or parse_datetime(row.get("VBT_PBTHDT"))
        departure_time = parse_datetime(row.get("VBT_ADPTDT")) or parse_datetime(row.get("VBT_PDPTDT"))
        berth_no = _norm(row.get("VBT_BTH_ABTHNO")) or _norm(row.get("VBT_BTH_PBTHNO"))
        if not (voyage_id and receive_start and receive_end and berth_time and departure_time):
            continue
        schedules[voyage_id] = VoyageSchedule(
            voyage_id=voyage_id,
            receive_start=receive_start,
            receive_end=receive_end,
            berth_no=berth_no,
            berth_time=berth_time,
            departure_time=departure_time,
        )
    return schedules


def select_upcoming_opening_voyages(
    data_dir: str | Path,
    planning_time: datetime = DEFAULT_PLANNING_TIME,
    horizon_hours: float = 24.0,
    count: int = 2,
) -> list[str]:
    horizon_end = planning_time + timedelta(hours=horizon_hours)
    candidates = [
        schedule
        for schedule in read_vessel_schedules(data_dir).values()
        if planning_time <= schedule.receive_start < horizon_end
    ]
    candidates.sort(key=lambda item: (item.receive_start, item.berth_time, item.voyage_id))
    return [item.voyage_id for item in candidates[:count]]


def read_big_plan(path: str | Path) -> list[BigPlanRow]:
    """璇诲彇澶ц鍒掔粨鏋滐紝骞跺敖閲忎繚鐣?20/40 灏哄閰嶉銆?
    鐩墠鏀寔涓ょ鏉ユ簮锛?    1. 涓皬璁″垝鑷繁鐨勬爣鍑嗘牸寮忥細`voyage_id, area_no, planned_boxes`锛?       濡傛灉棰濆甯︽湁 `size` 鎴?`size_mode` 鍒楋紝浼氳鍙栧昂瀵革紱鍚﹀垯璁颁负 `ALL`銆?    2. 澶ц鍒掓槑缁嗘牸寮忥細`voy_id, area_no, size, planned_qty`銆?
    瀵逛簬澶ц鍒掓槑缁嗭紝45 灏轰笉浼氬崟鐙嚭鐜帮紱濡傛灉澶栭儴鏂囦欢浼犲叆浜?45锛岃繖閲屼篃浼?    褰掑苟涓?40锛屼互淇濇寔鍜屽ぇ璁″垝妯″瀷涓€鑷淬€?    """

    counter: Counter[tuple[str, str, str]] = Counter()
    rows: list[BigPlanRow] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        required = {"voyage_id", "area_no", "planned_boxes"}
        missing = required.difference(reader.fieldnames or [])
        if missing and {"voy_id", "area_no", "planned_qty"}.issubset(reader.fieldnames or []):
            for row in reader:
                boxes = int(round(float(row["planned_qty"])))
                if boxes > 0:
                    size_mode = _big_plan_size_mode(row.get("size"))
                    counter[(_voyage(row["voy_id"]), _norm(row["area_no"]), size_mode)] += boxes
            rows = [
                BigPlanRow(voyage_id, area_no, boxes, size_mode)
                for (voyage_id, area_no, size_mode), boxes in sorted(counter.items())
                if boxes > 0
            ]
        elif missing:
            raise ValueError(f"big plan file missing columns: {sorted(missing)}")
        else:
            size_field = "size_mode" if "size_mode" in (reader.fieldnames or []) else "size"
            for row in reader:
                boxes = int(round(float(row["planned_boxes"])))
                if boxes > 0:
                    size_mode = _big_plan_size_mode(row.get(size_field)) if size_field in row else "ALL"
                    rows.append(BigPlanRow(_voyage(row["voyage_id"]), _norm(row["area_no"]), boxes, size_mode))
    if not rows:
        raise ValueError("big plan file contains no positive planned_boxes")
    return rows


def load_box_groups(
    data_dir: str | Path,
    voyage_ids: set[str],
    planned_by_voyage: dict[str, int],
    planned_by_voyage_size: dict[tuple[str, str], int] | None = None,
) -> list[BoxGroup]:
    """读取箱明细并缩放成中小计划需求组。

    箱量目标完全来自大计划，不再按 70% 或其他比例缩放。
    如果大计划提供 20/40 尺寸拆分，则按航次和尺寸分别对齐；45ft
    明细归入 40ft 目标。
    """
    data_path = Path(data_dir)
    counters: Counter[tuple] = Counter()
    for path in sorted(data_path.glob("container_info_*.parquet")):
        file_voyage = path.stem.replace("container_info_", "")
        frame = pd.read_parquet(path)
        for row in frame.to_dict("records"):
            voyage_id = _voyage(row.get("IYC_EVOY_ID"), file_voyage)
            if voyage_id not in voyage_ids:
                continue
            ctype = _norm(row.get("IYC_CTYPECD"), "UNK")
            status = _norm(row.get("IYC_STS_CSTATUSCD"), "UNK")
            key = (
                voyage_id,
                _norm(row.get("IYC_CSZ_CSIZECD"), "40"),
                _norm(row.get("IYC_CHEIGHTCD"), "UNK"),
                status,
                _norm(row.get("IYC_POT_UNLDPORT"), "UNK"),
                _norm(row.get("IYC_CST_COPERCD"), "UNK"),
                ctype,
                bool(ctype == "RF" or not pd.isna(row.get("IYC_SETTMPT"))),
                bool(not pd.isna(row.get("IYC_DTP_DNGGCD"))),
                bool(not pd.isna(row.get("IYC_OVLMTCD"))),
                tuple(sorted(_business_special_codes(row))),
            )
            counters[key] += 1

    raw_by_voyage: defaultdict[str, list[tuple[tuple, int]]] = defaultdict(list)
    for key, count in counters.items():
        raw_by_voyage[key[0]].append((key, count))

    groups: list[BoxGroup] = []
    for voyage_id in sorted(voyage_ids):
        raw = raw_by_voyage.get(voyage_id, [])
        if planned_by_voyage_size:
            size_targets = {
                size_mode: qty
                for (v, size_mode), qty in planned_by_voyage_size.items()
                if v == voyage_id and qty > 0
            }
            group_index = 1
            for size_mode in ("20", "40"):
                target = size_targets.get(size_mode, 0)
                if target <= 0:
                    continue
                raw_for_size = [
                    item for item in raw
                    if _big_plan_size_mode(item[0][1]) == size_mode
                ]
                if not raw_for_size:
                    raw_for_size = [(_generic_box_key(voyage_id, size_mode), 1)]
                group_index = _append_scaled_groups(groups, raw_for_size, target, group_index)
            continue

        if not raw:
            raw = [(_generic_box_key(voyage_id, "40"), 1)]
        _append_scaled_groups(groups, raw, planned_by_voyage[voyage_id], 1)
    return groups


def _business_special_codes(row: dict) -> set[str]:
    codes: set[str] = set()
    ctype = _norm(row.get("IYC_CTYPECD"))
    if ctype in {"OT", "FR", "TK", "BU"}:
        codes.add(f"TYPE_{ctype}")
    dng = _norm(row.get("IYC_DTP_DNGGCD"))
    if dng:
        codes.add(f"DG_{dng}")
    ov = _norm(row.get("IYC_OVLMTCD"))
    if ov:
        codes.add(f"OV_{ov}")
    return codes


def _largest_remainder_scale(counts: list[int], source_total: int, target_total: int) -> list[int]:
    if target_total <= 0:
        return [0 for _ in counts]
    if source_total <= 0:
        out = [0 for _ in counts]
        out[0] = target_total
        return out
    raw = [c * target_total / source_total for c in counts]
    base = [int(math.floor(v)) for v in raw]
    remain = target_total - sum(base)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:remain]:
        base[i] += 1
    return base


def _big_plan_size_mode(value: object) -> str:
    """杞崲涓哄ぇ璁″垝灏哄鍙ｅ緞銆?
    澶ц鍒掑彧鍖哄垎 20 鍜?40锛屽叾涓?45 灏哄綊鍏?40 灏恒€傚鏋滄棫鏂囦欢娌℃湁灏哄鍒楋紝
    杩斿洖 `ALL`锛屽悗缁細閫€鍥炲埌鍙寜鑸-绠卞尯鎬婚噺绾︽潫銆?    """
    size = _norm(value).upper()
    if not size:
        return "ALL"
    if size == "20":
        return "20"
    if size in {"40", "45"}:
        return "40"
    return "40"


def _generic_box_key(voyage_id: str, size_mode: str) -> tuple:
    """鏋勯€犲厹搴曞睘鎬х粍銆?
    褰撴煇鑸鏈夊ぇ璁″垝閰嶉锛屼絾绠辨槑缁嗕腑鎵句笉鍒板搴斿昂瀵哥殑绠卞瓙鏃讹紝鐢ㄤ竴涓櫘閫?    灞炴€х粍鎵挎帴杩欓儴鍒嗚鍒掗噺锛屼繚璇佹ā鍨嬩粛鑳界粰鍑哄彲妫€鏌ョ殑缁撴灉銆?    """
    return (voyage_id, size_mode, "UNK", "OF", "UNK", "UNK", "GP", False, False, False, ())


def _append_scaled_groups(
    groups: list[BoxGroup],
    raw: list[tuple[tuple, int]],
    target: int,
    start_index: int,
) -> int:
    """鎶婂師濮嬪睘鎬х粍鎸夌洰鏍囩閲忕缉鏀惧悗杩藉姞鍒?`groups`銆?
    杩斿洖涓嬩竴涓彲鐢ㄧ殑缁勭紪鍙凤紝渚夸簬鍚屼竴鑸鎸?20/40 鍒嗘《缂╂斁鍚庝粛淇濇寔缁勫彿鍞竴銆?    """
    total_raw = sum(c for _, c in raw)
    scaled_counts = _largest_remainder_scale([c for _, c in raw], total_raw, target)
    group_index = start_index
    for key, demand in zip((item[0] for item in raw), scaled_counts):
        if demand <= 0:
            continue
        v, size, height, status, port, operator, ctype, reefer, dangerous, over_limit, special_codes = key
        groups.append(
            BoxGroup(
                group_id=f"{v}_G{group_index:03d}",
                voyage_id=v,
                size=size,
                height=height,
                status=status,
                port=port,
                operator=operator,
                ctype=ctype,
                reefer=reefer,
                dangerous=dangerous,
                over_limit=over_limit,
                special_codes=special_codes,
                demand=demand,
            )
        )
        group_index += 1
    return group_index


def build_bays(
    data_dir: str | Path,
    allowed_areas: set[str],
    closed_areas: set[str],
    planning_time: datetime = DEFAULT_PLANNING_TIME,
    vessel_schedules: dict[str, VoyageSchedule] | None = None,
) -> dict[str, Bay]:
    data_path = Path(data_dir)
    base = pd.read_parquet(data_path / "bay_slots_detail.parquet")
    # 拆分文件已经分别表示 20ft 和 40ft 可用视图。中计划和小计划
    # 都按大计划口径处理尺寸：45ft 箱统一消耗 40ft 容量。
    cap40 = _capacity_by_bay(data_path / "bay_slots_detail_40.parquet")
    physical_cap = _physical_capacity_by_bay(base)
    vessel_schedules = vessel_schedules or read_vessel_schedules(data_dir)
    released_cap = _released_capacity_by_bay(base, vessel_schedules, planning_time)
    for key, count in released_cap["40"].items():
        cap40[key] = cap40.get(key, 0) + count
    for key, count in released_cap["physical"].items():
        physical_cap[key] = physical_cap.get(key, 0) + count
    # 大计划使用的是 20ft 等价容量。为了让中小计划与大计划口径一致，
    # 20ft 箱在贝位层也允许使用普通物理空位；最终仍由 physical_capacity
    # 限制同一贝位内 20/40 总占用不超量。
    cap20 = dict(physical_cap)

    bay_numbers = (
        base[["YAA_AREANO", "YBY_BAYNO"]]
        .drop_duplicates()
        .assign(
            YAA_AREANO=lambda x: x["YAA_AREANO"].map(_norm),
            YBY_BAYNO=lambda x: x["YBY_BAYNO"].map(_bay_no),
        )
    )
    bay_order: dict[tuple[str, str], int] = {}
    block_by_bay: dict[tuple[str, str], tuple[int, tuple[str, ...], bool]] = {}
    for area_no, area_df in bay_numbers.groupby("YAA_AREANO"):
        ordered = sorted(area_df["YBY_BAYNO"].tolist(), key=_bay_sort_key)
        for idx, bay_no in enumerate(ordered):
            bay_order[(area_no, bay_no)] = idx
        big_bay_starts = {
            bay_no
            for bay_no in ordered
            if cap40.get((area_no, bay_no), 0) > 0
        }
        for block_index, members, adjusted in _make_yard_blocks(ordered, big_bay_starts):
            for bay_no in members:
                block_by_bay[(area_no, bay_no)] = (block_index, members, adjusted)

    bays: dict[str, Bay] = {}
    for (area_no, bay_no), order in bay_order.items():
        if area_no in closed_areas or area_no not in allowed_areas:
            continue
        block_index, block_members, block_adjusted = block_by_bay.get((area_no, bay_no), (order + 1, (bay_no,), False))
        bay_key = f"{area_no}-{bay_no}"
        bays[bay_key] = Bay(
            area_no=area_no,
            bay_no=bay_no,
            bay_key=bay_key,
            block_id=f"{area_no}-B{block_index:02d}",
            block_bays=block_members,
            block_bay_count=len(block_members),
            block_boundary_adjusted=block_adjusted,
            bay_order=order,
            cap_by_size={
                "20": cap20.get((area_no, bay_no), 0),
                "40": cap40.get((area_no, bay_no), 0),
            },
            physical_capacity=physical_cap.get((area_no, bay_no), 0),
        )

    occupied = base[base["HAS_CONTAINER"] == 1]
    for row in occupied.to_dict("records"):
        if not _is_occupied_at_planning_time(row, vessel_schedules, planning_time):
            continue
        area_no = _norm(row.get("YAA_AREANO"))
        bay_no = _bay_no(row.get("YBY_BAYNO"))
        bay = bays.get(f"{area_no}-{bay_no}")
        if bay is None:
            continue
        size = _norm(row.get("IYC_CSZ_CSIZECD"))
        if size:
            bay.existing_size_modes.add(_size_mode(size))
        height = _norm(row.get("IYC_CHEIGHTCD"))
        if height:
            bay.existing_heights.add(height)
        port = _norm(row.get("IYC_POT_UNLDPORT"))
        if port:
            bay.existing_ports.add(port)
        status = _norm(row.get("IYC_STS_CSTATUSCD"))
        ctype = _norm(row.get("IYC_CTYPECD"))
        if status.endswith("E") or status in {"IE", "OE", "TE", "RE"}:
            bay.fallback_reasons.add("existing_empty_container")
        if ctype == "RF" or not pd.isna(row.get("IYC_SETTMPT")):
            bay.fallback_reasons.add("existing_reefer_container")
        sig = _existing_special_signature(row)
        if sig != "NORMAL":
            bay.existing_special_signatures.add(sig)

    return bays


def build_area_operations(
    data_dir: str | Path,
    vessel_schedules: dict[str, VoyageSchedule],
) -> dict[str, list[AreaOperation]]:
    base = pd.read_parquet(Path(data_dir) / "bay_slots_detail.parquet")
    occupied = base[base["HAS_CONTAINER"] == 1]
    seen: set[tuple[str, str]] = set()
    operations: defaultdict[str, list[AreaOperation]] = defaultdict(list)
    for row in occupied.to_dict("records"):
        voyage_id = _voyage(row.get("IYC_EVOY_ID"))
        schedule = vessel_schedules.get(voyage_id)
        if schedule is None:
            continue
        area_no = _norm(row.get("YAA_AREANO"))
        if not area_no:
            continue
        key = (area_no, voyage_id)
        if key in seen:
            continue
        seen.add(key)
        operations[area_no].append(
            AreaOperation(
                area_no=area_no,
                voyage_id=voyage_id,
                start_time=schedule.berth_time,
                end_time=schedule.departure_time,
            )
        )
    for items in operations.values():
        items.sort(key=lambda item: (item.start_time, item.end_time, item.voyage_id))
    return dict(operations)


def _released_capacity_by_bay(
    base: pd.DataFrame,
    vessel_schedules: dict[str, VoyageSchedule],
    planning_time: datetime,
) -> dict[str, Counter[tuple[str, str]]]:
    released = {"20": Counter(), "40": Counter(), "physical": Counter()}
    occupied = base[base["HAS_CONTAINER"] == 1]
    for row in occupied.to_dict("records"):
        if _is_occupied_at_planning_time(row, vessel_schedules, planning_time):
            continue
        area_no = _norm(row.get("YAA_AREANO"))
        bay_no = _bay_no(row.get("YBY_BAYNO"))
        if not area_no or not bay_no:
            continue
        released["physical"][(area_no, bay_no)] += 1
        enabled_sizes = _enabled_size_modes(row.get("YBY_ENABLECSIZECD"))
        if not enabled_sizes:
            enabled_sizes = {_size_mode(row.get("IYC_CSZ_CSIZECD"))}
        for size_mode in enabled_sizes:
            released[size_mode][(area_no, bay_no)] += 1
    return released


def _is_occupied_at_planning_time(
    row: dict,
    vessel_schedules: dict[str, VoyageSchedule],
    planning_time: datetime,
) -> bool:
    if row.get("HAS_CONTAINER") != 1:
        return False
    voyage_id = _voyage(row.get("IYC_EVOY_ID"))
    if not voyage_id:
        return True
    schedule = vessel_schedules.get(voyage_id)
    if schedule is None:
        return True
    return schedule.departure_time > planning_time


def _capacity_by_bay(path: Path, required_size: str | None = None) -> dict[tuple[str, str], int]:
    """鎸夎礉浣嶇粺璁＄┖浣嶅閲忋€?
    `bay_slots_detail_20.parquet` 鍜?`bay_slots_detail_40.parquet` 鏈韩宸茬粡鏄?    鎸夊昂瀵告媶鍑烘潵鐨勮鍥撅紝鍥犳璇诲彇 20/40 瀹归噺鏃朵笉鑳藉啀寮哄埗瑕佹眰
    `YBY_ENABLECSIZECD` 閲屽嚭鐜板悓涓€涓昂瀵搞€傜鍖?38 鐨?20 灏虹┖浣嶅氨鏄竴涓緥瀛愶細
    瀹冨湪 20 灏鸿鍥句腑瀛樺湪锛屼絾鍚敤灏哄瀛楁鏄剧ず涓?`40, 45`锛屽鏋滃啀娆¤繃婊や細
    琚敊璇墸鎺夛紝瀵艰嚧涓鍒掕涓哄ぇ璁″垝缁欏嚭鐨?20 灏洪厤棰濇棤鍖哄潡鍙斁銆?
    褰撳墠姝ｅ紡姹傝В涓嶅啀鍗曠嫭浣跨敤 45 灏哄閲忥紱淇濈暀 `required_size` 鍙傛暟鍙槸涓轰簡
    鍏煎鍙兘鐨勪复鏃舵鏌ャ€?    """
    frame = pd.read_parquet(path)
    empty = frame[frame["HAS_CONTAINER"] == 0].copy()
    if required_size is not None:
        empty = empty[empty["YBY_ENABLECSIZECD"].apply(lambda value: _slot_allows_size(value, required_size))]
    empty["YAA_AREANO"] = empty["YAA_AREANO"].astype(str)
    empty["YBY_BAYNO"] = empty["YBY_BAYNO"].map(_bay_no)
    counts = empty.groupby(["YAA_AREANO", "YBY_BAYNO"]).size()
    return {(str(a), str(b)): int(v) for (a, b), v in counts.items()}


def _physical_capacity_by_bay(base: pd.DataFrame) -> dict[tuple[str, str], int]:
    empty = base[base["HAS_CONTAINER"] == 0].copy()
    empty["YAA_AREANO"] = empty["YAA_AREANO"].astype(str)
    empty["YBY_BAYNO"] = empty["YBY_BAYNO"].map(_bay_no)
    counts = empty.groupby(["YAA_AREANO", "YBY_BAYNO"]).size()
    return {(str(a), str(b)): int(v) for (a, b), v in counts.items()}


def _enabled_size_modes(value: object) -> set[str]:
    text = _norm(value)
    if not text:
        return set()
    modes = set()
    for part in re.findall(r"\d+", text):
        if part == "20":
            modes.add("20")
        elif part in {"40", "45"}:
            modes.add("40")
    return modes


def _slot_allows_size(value: object, required_size: str) -> bool:
    enabled = _enabled_size_modes(value)
    if enabled:
        return required_size in enabled
    return required_size in {"20", "40"}


def _make_yard_blocks(
    ordered_bays: list[str],
    big_bay_starts: set[str],
    target_bay_count: int = 6,
) -> list[tuple[int, tuple[str, ...], bool]]:
    """鏋勯€犱腑璁″垝浣跨敤鐨勫尯鍧椼€?
    鐩爣鏄瘡涓尯鍧楀寘鍚?6 涓繛缁皬璐濅綅銆傚鏋滆竟鐣屼細鍒囧紑涓€涓?40ft 澶ц礉浣嶅锛?    灏卞悜闄勮繎绉诲姩杈圭晫銆?5ft 鍦ㄦ湰妯″瀷涓寜 40ft 澶勭悊锛屽洜姝や笉鍐嶅崟鐙垽鏂?45ft銆?    """

    blocks: list[tuple[int, tuple[str, ...], bool]] = []
    start = 0
    block_index = 1
    while start < len(ordered_bays):
        if len(ordered_bays) - start <= target_bay_count:
            end = len(ordered_bays)
            adjusted = False
        else:
            target_end = start + target_bay_count
            end = _nearest_safe_block_end(
                ordered_bays,
                big_bay_starts,
                start=start,
                target_end=target_end,
            )
            adjusted = end != target_end
        members = tuple(ordered_bays[start:end])
        blocks.append((block_index, members, adjusted))
        block_index += 1
        start = end
    return blocks


def _nearest_safe_block_end(
    ordered_bays: list[str],
    big_bay_starts: set[str],
    start: int,
    target_end: int,
) -> int:
    # 如果 end-1 位置的小贝位可以作为 40ft 大贝位起点，则边界会切开
    # 推断的大贝位对，属于不安全边界。
    min_end = start + 1
    max_end = len(ordered_bays)
    for offset in range(0, len(ordered_bays) + 1):
        probes = [target_end] if offset == 0 else [target_end + offset, target_end - offset]
        for end in probes:
            if not (min_end <= end <= max_end):
                continue
            if end == len(ordered_bays) or ordered_bays[end - 1] not in big_bay_starts:
                return end
    return len(ordered_bays)


def _size_mode(value: object) -> str:
    size = _norm(value, "40")
    return "20" if size == "20" else "40"


def _existing_special_signature(row: dict) -> str:
    marks = []
    if _norm(row.get("IYC_CTYPECD")) == "RF" or not pd.isna(row.get("IYC_SETTMPT")):
        marks.append("RF")
    if not pd.isna(row.get("IYC_DTP_DNGGCD")):
        marks.append("DG")
    if not pd.isna(row.get("IYC_OVLMTCD")):
        marks.append("OV")
    return "+".join(marks) if marks else "NORMAL"


def _bay_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        nums = re.findall(r"\d+", value)
        return (int(nums[0]) if nums else 9999), value


def _bay_no(value: object) -> str:
    """缁熶竴璐濅綅鍙锋牸寮忋€?
    涓嶅悓婧愭枃浠堕噷鍚屼竴涓礉浣嶅彲鑳藉啓鎴?`5`銆乣5.0` 鎴?`05`銆傚閲忕粺璁″拰璐濅綅瀵硅薄
    蹇呴』浣跨敤鍚屼竴涓敭锛屽惁鍒欎細鍑虹幇鈥滄槑鏄庢湁绌轰綅锛屼絾鍖哄潡瀹归噺涓?0鈥濈殑鍋囪薄銆?    """
    text = _norm(value)
    if text.isdigit() and len(text) == 1:
        return f"0{text}"
    return text


def make_units(groups: list[BoxGroup]) -> list[Unit]:
    units: list[Unit] = []
    uid = 0
    for group in groups:
        for _ in range(group.demand):
            units.append(Unit(uid, group))
            uid += 1
    return units


def build_problem(
    data_dir: str | Path,
    big_plan: list[BigPlanRow],
    planning_time: datetime = DEFAULT_PLANNING_TIME,
    horizon_hours: float = 24.0,
    small_plan_threshold: int = 10,
) -> ProblemData:
    """鏋勯€犱腑璁″垝/灏忚鍒掔殑瀹屾暣杈撳叆銆?
    澶ц鍒掑湪杩欓噷琚綋鎴愮‖杈圭晫锛?    - 鑸鍙兘浣跨敤澶ц鍒掑凡缁忛€夊嚭鐨勭鍖猴紱
    - 姣忎釜鑸-绠卞尯鐨勬€荤閲忓繀椤荤瓑浜庡ぇ璁″垝锛?    - 濡傛灉澶ц鍒掓彁渚涗簡灏哄琛岋紝鍒欐瘡涓埅娆?绠卞尯-灏哄鐨勭閲忎篃蹇呴』绛変簬澶ц鍒掋€?    """

    closed = read_closed_areas(data_dir)
    export_areas = read_export_areas(data_dir)
    planned_by_voyage: defaultdict[str, int] = defaultdict(int)
    planned_by_voyage_size: defaultdict[tuple[str, str], int] = defaultdict(int)
    area_quota: dict[tuple[str, str], int] = {}
    area_size_quota: dict[tuple[str, str, str], int] = {}
    assigned_areas: defaultdict[str, set[str]] = defaultdict(set)
    cleaned_plan: list[BigPlanRow] = []
    vessel_schedules = read_vessel_schedules(data_dir)
    # 中计划和小计划承接大计划结果，箱量必须完全等于大计划，
    # 不再按 70% 或其他滚动比例自行缩放。
    input_plan = big_plan
    has_size_quota = any(row.size_mode != "ALL" for row in input_plan)
    for row in input_plan:
        if row.area_no in closed:
            raise ValueError(f"big plan uses closed area {row.area_no} for voyage {row.voyage_id}")
        if export_areas and row.area_no not in export_areas:
            raise ValueError(
                f"big plan uses area {row.area_no} for voyage {row.voyage_id}, "
                "but it is not in of閫傛斁绠卞尯鍒楄〃.xlsx"
            )
        cleaned_plan.append(row)
        planned_by_voyage[row.voyage_id] += row.planned_boxes
        area_quota[(row.voyage_id, row.area_no)] = area_quota.get((row.voyage_id, row.area_no), 0) + row.planned_boxes
        if row.size_mode != "ALL":
            planned_by_voyage_size[(row.voyage_id, row.size_mode)] += row.planned_boxes
            area_size_quota[(row.voyage_id, row.area_no, row.size_mode)] = (
                area_size_quota.get((row.voyage_id, row.area_no, row.size_mode), 0)
                + row.planned_boxes
            )
        assigned_areas[row.voyage_id].add(row.area_no)
    if not cleaned_plan:
        raise ValueError("no big-plan rows remain after Guandong export-area and closed-area filtering")

    groups = load_box_groups(
        data_dir,
        set(planned_by_voyage),
        dict(planned_by_voyage),
        dict(planned_by_voyage_size) if has_size_quota else None,
    )
    units = make_units(groups)
    voyage_windows = _build_voyage_windows(sorted(planned_by_voyage), vessel_schedules, horizon_hours, planning_time)
    # `planning_time` 是当前运行分配算法的时刻，也是判断堆场已有箱是否仍占用
    # 容量的快照时刻。不能用开港窗口开始时间覆盖它，否则诊断输出和容量判断
    # 都会偏到第一个航次的 receive_start。
    capacity_snapshot_time = planning_time
    bays = build_bays(
        data_dir,
        export_areas,
        closed,
        planning_time=capacity_snapshot_time,
        vessel_schedules=vessel_schedules,
    )
    area_operations = build_area_operations(data_dir, vessel_schedules)
    target_voyages = sorted(planned_by_voyage)
    return ProblemData(
        groups=groups,
        units=units,
        bays=bays,
        big_plan=cleaned_plan,
        assigned_areas=dict(assigned_areas),
        area_quota=area_quota,
        area_size_quota=area_size_quota,
        small_plan_threshold=small_plan_threshold,
        business_special_codes=_collect_business_special_codes(groups),
        planning_time=capacity_snapshot_time,
        horizon_hours=horizon_hours,
        voyage_windows=voyage_windows,
        area_operations=area_operations,
        target_voyages=target_voyages,
    )


def _build_voyage_windows(
    voyage_ids: list[str],
    schedules: dict[str, VoyageSchedule],
    horizon_hours: float,
    fallback_start: datetime,
) -> dict[str, tuple[datetime, datetime]]:
    windows: dict[str, tuple[datetime, datetime]] = {}
    for voyage_id in voyage_ids:
        start = schedules[voyage_id].receive_start if voyage_id in schedules else fallback_start
        windows[voyage_id] = (start, start + timedelta(hours=horizon_hours))
    return windows


def _collect_business_special_codes(groups: list[BoxGroup]) -> set[str]:
    out: set[str] = set()
    for group in groups:
        out.update(group.special_codes)
    return out
