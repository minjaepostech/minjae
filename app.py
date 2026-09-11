import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 1. 페이지 설정
st.set_page_config(page_title="FET-Analysis_Minjae", layout="wide")
st.title("FET-Analysis_Minjae")

# ✅ 드래그바(슬라이더)의 선, 원, 위에 뜨는 숫자까지 모두 검은색/기본색으로 완벽 통일하는 CSS
st.markdown("""
<style>
/* 1. 슬라이더 손잡이(원) 검은색 */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: black !important;
    border-color: black !important;
}
/* 2. 슬라이더 채워진 선(트랙) 검은색으로 강제 덮어쓰기 */
div[data-testid="stSlider"] div[data-testid="stSliderTrack"] > div:nth-child(1) {
    background-color: black !important;
}
/* 3. 슬라이더 위에 뜨는 작은 숫자 말풍선 배경 투명하게, 글씨는 기본색(테마색)으로 */
div[data-testid="stSlider"] div[role="slider"] > div {
    color: var(--text-color) !important;
    background-color: transparent !important;
}
/* 만약 인라인 스타일로 칠해지는 기본 빨간색이 있다면 모두 검은색으로 차단 */
div[data-testid="stSlider"] div[style*="rgb(255, 75, 75)"],
div[data-testid="stSlider"] div[style*="#ff4b4b"] {
    background-color: black !important;
}
/* Scientific notation: uppercase display only for the TLM minimum-current input.
   Keep Streamlit's supported %e formatter; do not change fonts/colours/layout. */
.st-key-tlm_min_id input,
[data-testid="stSidebar"] input[aria-label^="Minimum "] {
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# TLM-only helpers. The original FET branch is preserved separately below.
# All resistances are fit internally as R_total * W_um versus L_um.
# Thus the internal slope is R_sheet [ohm/sq] without an extra scale factor.
# ============================================================================
from io import BytesIO
import hashlib
import html
import re


_TLM_VOLTAGE_ATOL = 1e-6  # V; tolerate float32 programmed-voltage storage.
_TLM_BASE_COLUMNS = ("DrainI", "DrainV", "GateV")
_TLM_COL_RE = re.compile(
    r"^(DrainI|DrainV|GateV)(?:\((\d+)\)|[._](\d+))?$", re.IGNORECASE
)
_TLM_COL_NAMES = {name.lower(): name for name in _TLM_BASE_COLUMNS}
_TLM_COLORS = [
    "#2E60AB", "#A23B72", "#F18F01", "#18A558", "#5B5F97", "#F05650",
    "#6FADCF", "#98633D", "#8F6BC4", "#118F8B", "#6F7644", "#555555",
]
_TLM_SYMBOLS = [
    "circle", "square", "diamond", "triangle-up", "triangle-down", "cross",
    "x", "triangle-left", "triangle-right", "star", "pentagon", "hexagon",
]


def _tlm_column_id(value):
    """Recognize both the supplied wide triplets and ordinary long-format headers."""
    name = re.sub(r"\s+", "", str(value)).lstrip("\ufeff")
    match = _TLM_COL_RE.fullmatch(name)
    if match is None:
        return None
    kind = _TLM_COL_NAMES[match.group(1).lower()]
    suffix = match.group(2) or match.group(3) or ""
    return kind, suffix


def _tlm_find_header(raw):
    """Find a header in the first 25 rows; do not infer unknown column meanings."""
    for i in range(min(25, len(raw))):
        found = {}
        for column_index, value in enumerate(raw.iloc[i].tolist()):
            key = _tlm_column_id(value)
            if key:
                kind, suffix = key
                if kind in found.setdefault(suffix, {}):
                    raise ValueError("같은 이름의 측정 열이 중복되어 열을 구분할 수 없습니다.")
                found[suffix][kind] = column_index
        complete = {
            suffix: cols for suffix, cols in found.items()
            if all(kind in cols for kind in _TLM_BASE_COLUMNS)
        }
        if complete:
            return i, complete, found
    raise ValueError(
        "DrainI(n), DrainV(n), GateV(n) 또는 DrainI, DrainV, GateV 열을 찾지 못했습니다."
    )


def _tlm_split_sweeps(vd):
    """Split by actual acquisition direction; never average forward/backward sweeps."""
    values = np.asarray(vd, dtype=float)
    if len(values) < 2:
        return [(0, len(values))]
    segments, start, previous_direction = [], 0, 0
    for edge, difference in enumerate(np.diff(values)):
        direction = 0 if abs(difference) <= 1e-10 else (1 if difference > 0 else -1)
        if direction == 0:
            continue
        if previous_direction and direction != previous_direction:
            # The turning point belongs to both adjoining sweeps.
            segments.append((start, edge + 1))
            start = edge
        previous_direction = direction
    segments.append((start, len(values)))
    return segments


def _tlm_parse_sheet(raw, sheet_name):
    """Return {nominal_VG: [monotonic acquisition passes]} and parsing notes.

    Current must already be in A; voltage must already be in V. No guessed
    current scaling, gate interpolation, or forward/backward averaging is done.
    """
    header_index, complete, found = _tlm_find_header(raw)
    curves, notes = {}, []
    for suffix in found:
        if suffix not in complete:
            notes.append(f"{sheet_name}: 불완전한 측정 열 묶음 ({suffix})은 읽지 않았습니다.")
    for suffix, columns in complete.items():
        data = raw.iloc[header_index + 1:]
        vectors = {
            kind: pd.to_numeric(data.iloc[:, col], errors="coerce").to_numpy(dtype=float)
            for kind, col in columns.items()
        }
        gate = vectors["GateV"]
        drain = vectors["DrainV"]
        current = vectors["DrainI"]
        finite = np.isfinite(gate) & np.isfinite(drain) & np.isfinite(current)
        nonempty = data.iloc[:, list(columns.values())].notna().any(axis=1).to_numpy()
        discarded = int(np.sum(nonempty & ~finite))
        if discarded:
            notes.append(
                f"{sheet_name}, 묶음 {suffix or '기본'}: 숫자가 아닌 값/결측값 "
                f"{discarded}개 행을 제외했습니다. 그 공백을 가로질러 보간하지 않습니다."
            )
        # Split at missing rows and at actual VG changes. Keep repeated gate
        # sweeps as separate passes in their original acquisition order.
        start = 0
        while start < len(gate):
            if not finite[start]:
                start += 1
                continue
            stop = start + 1
            nominal_gate = float(gate[start])
            while (stop < len(gate) and finite[stop]
                   and abs(float(gate[stop]) - nominal_gate) <= _TLM_VOLTAGE_ATOL):
                stop += 1
            matched_gate = next(
                (g for g in curves if abs(g - nominal_gate) <= _TLM_VOLTAGE_ATOL), None
            )
            if matched_gate is None:
                matched_gate = float(np.round(np.median(gate[start:stop]), 6))
                curves.setdefault(matched_gate, [])
            local_vd = drain[start:stop]
            for seg_start, seg_stop in _tlm_split_sweeps(local_vd):
                a, b = start + seg_start, start + seg_stop
                if b <= a:
                    continue
                curves[matched_gate].append({
                    "sheet": str(sheet_name), "group": suffix or "base",
                    "vg": matched_gate,
                    "vd": drain[a:b].copy(), "id": current[a:b].copy(),
                    "rows": np.arange(a, b, dtype=int) + header_index + 2,
                    "vg_actual": gate[a:b].copy(),
                })
            start = stop
    if not curves:
        raise ValueError("숫자로 읽을 수 있는 output 데이터가 없습니다.")
    return {"curves": curves, "notes": notes, "header_row": header_index + 1}


def _tlm_settings_info(raw):
    """Only read clearly labelled Settings metadata; never apply it to every sheet."""
    if raw is None or raw.empty:
        return {}
    terminal_column = None
    for row in raw.itertuples(index=False, name=None):
        if len(row) and str(row[0]).strip().lower() == "device terminal":
            terminal_column = next(
                (j for j, x in enumerate(row) if str(x).strip().lower() == "drain"), None
            )
            break
    info = {}
    if terminal_column is not None:
        for row in raw.itertuples(index=False, name=None):
            if not row or terminal_column >= len(row):
                continue
            if str(row[0]).strip().lower() == "compliance":
                try:
                    limit = float(row[terminal_column])
                    if np.isfinite(limit) and limit > 0:
                        info["latest_drain_compliance_A"] = limit
                except (TypeError, ValueError):
                    pass
    return info


def _tlm_load_excel(file_bytes):
    """Read the user's workbook without writing or modifying it."""
    engine = "xlrd" if file_bytes[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" else "openpyxl"
    result = {"sheets": {}, "errors": {}, "metadata": {}}
    with pd.ExcelFile(BytesIO(file_bytes), engine=engine) as book:
        candidates = [
            s for s in book.sheet_names
            if s.strip().lower() == "data" or s.strip().lower().startswith("append")
        ]
        result["sheet_names"] = list(book.sheet_names)
        for sheet in candidates:
            raw = pd.read_excel(book, sheet_name=sheet, header=None)
            try:
                result["sheets"][sheet] = _tlm_parse_sheet(raw, sheet)
            except (ValueError, TypeError, IndexError) as exc:
                result["errors"][sheet] = str(exc)
        settings = next((s for s in book.sheet_names if s.strip().lower() == "settings"), None)
        if settings:
            result["metadata"] = _tlm_settings_info(
                pd.read_excel(book, sheet_name=settings, header=None)
            )
    return result


def _tlm_common_gates(parsed_sheets, sheet_names):
    """Use only measured VG values present in every selected device/sheet."""
    if not sheet_names:
        return []
    first = parsed_sheets[sheet_names[0]]["curves"]
    return [
        float(g) for g in first
        if all(any(abs(g - h) <= _TLM_VOLTAGE_ATOL
                   for h in parsed_sheets[s]["curves"]) for s in sheet_names[1:])
    ]


def _tlm_get_curve(parsed, vg, pass_index):
    matches = [g for g in parsed["curves"] if abs(float(g) - vg) <= _TLM_VOLTAGE_ATOL]
    if len(matches) != 1:
        raise ValueError("해당 V_G가 없거나 서로 구별할 수 없는 V_G가 중복되어 있습니다.")
    passes = parsed["curves"][matches[0]]
    if pass_index >= len(passes):
        raise ValueError(f"선택한 Sweep {pass_index + 1}이 없습니다.")
    return passes[pass_index]


def _tlm_sample_at_vd(curve, target_vd, allow_interpolation=True,
                      min_current=0.0, compliance=0.0):
    """Extract a signed current at one VD; no extrapolation or zero-bias ratio.

    Exact duplicate VD points within the same monotonic pass are averaged,
    explicitly recorded. An exact match tolerates float32 voltage roundoff.
    """
    if not np.isfinite(target_vd) or target_vd == 0.0:
        raise ValueError("V_D = 0 V에서는 V_D/I_D 저항을 구할 수 없습니다.")
    v = np.asarray(curve["vd"], dtype=float)
    i = np.asarray(curve["id"], dtype=float)
    if len(v) == 0:
        raise ValueError("선택한 sweep에 데이터가 없습니다.")
    # Average exact duplicate voltages only within this one acquisition pass.
    uv, inverse = np.unique(v, return_inverse=True)
    counts = np.bincount(inverse)
    ui = np.bincount(inverse, weights=i) / counts
    tolerance = max(1e-8, abs(target_vd) * 1e-6)
    close = np.where(np.abs(uv - target_vd) <= tolerance)[0]
    if len(close):
        index = int(close[np.argmin(np.abs(uv[close] - target_vd))])
        used_vd, used_id = float(uv[index]), float(ui[index])
        vl = vh = used_vd
        il = ih = used_id
        sample_count = int(counts[index])
        method = "measured" if sample_count == 1 else "duplicate_mean_same_pass"
        contributing_currents = i[v == used_vd]
    else:
        if target_vd < uv[0] or target_vd > uv[-1]:
            raise ValueError(
                f"V_D={target_vd:g} V는 측정 범위 [{uv[0]:g}, {uv[-1]:g}] V 밖입니다."
            )
        if not allow_interpolation:
            raise ValueError("정확한 V_D 측정점이 없습니다. 필요하면 보간을 허용하세요.")
        hi = int(np.searchsorted(uv, target_vd))
        lo = hi - 1
        if lo < 0 or hi >= len(uv) or uv[hi] <= uv[lo]:
            raise ValueError("선택한 V_D를 둘러싼 서로 다른 두 측정점이 없습니다.")
        vl, vh, il, ih = float(uv[lo]), float(uv[hi]), float(ui[lo]), float(ui[hi])
        if il * ih < 0:
            raise ValueError("보간 양 끝의 전류 부호가 달라 영점 부근 저항을 신뢰하기 어렵습니다.")
        used_id = il + (ih - il) * (target_vd - vl) / (vh - vl)
        used_vd, method = float(target_vd), "linear_interpolation"
        sample_count = int(counts[lo] + counts[hi])
        contributing_currents = i[(v == vl) | (v == vh)]
    if not np.isfinite(used_id) or abs(used_id) <= min_current:
        raise ValueError("I_D가 0이거나 설정한 최소 |I_D| 이하입니다.")
    if compliance > 0 and np.max(np.abs(contributing_currents)) >= 0.99 * compliance:
        raise ValueError("선택점 또는 보간 끝점이 입력한 current compliance의 99% 이상입니다.")
    resistance = abs(used_vd / used_id)
    if not np.isfinite(resistance) or resistance <= 0:
        raise ValueError("유한한 양의 총저항을 계산할 수 없습니다.")
    return {
        "VD_used_V": used_vd, "ID_A": used_id, "Rtotal_ohm": resistance,
        "Extraction": method, "N_contributing_points": sample_count,
        "VD_lower_V": vl, "VD_upper_V": vh, "ID_lower_A": il, "ID_upper_A": ih,
        "VD_ID_same_sign": bool(used_vd * used_id > 0),
    }


def _tlm_fit_line(lengths_um, rw_ohm_um):
    """Unconstrained ordinary least squares. Uncertainties are residual-based SEs."""
    x = np.asarray(lengths_um, dtype=float)
    y = np.asarray(rw_ohm_um, dtype=float)
    if (len(x) != len(y) or len(x) < 2 or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y)) or np.any(x <= 0)):
        raise ValueError("피팅에는 유효한 양의 길이와 유한한 저항 데이터가 필요합니다.")
    if len(np.unique(x)) < 2:
        raise ValueError("서로 다른 채널 길이가 최소 2개 필요합니다.")
    xc = x - x.mean()
    sxx = float(np.dot(xc, xc))
    slope = float(np.dot(xc, y - y.mean()) / sxx)
    intercept = float(y.mean() - slope * x.mean())
    fitted = slope * x + intercept
    residual = y - fitted
    sse = float(np.dot(residual, residual))
    sst = float(np.dot(y - y.mean(), y - y.mean()))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    n, dof = len(x), len(x) - 2
    slope_se = intercept_se = np.nan
    if dof > 0:
        variance = sse / dof
        slope_se = float(np.sqrt(variance / sxx))
        intercept_se = float(np.sqrt(variance * (1.0 / n + x.mean() ** 2 / sxx)))
    flags = []
    if len(np.unique(x)) < 3:
        flags.append("only_2_distinct_lengths")
    if slope <= 0:
        flags.append("nonpositive_Rsheet")
    if intercept < 0:
        flags.append("negative_intercept")
    if np.isfinite(intercept_se) and abs(intercept) <= 2 * intercept_se:
        flags.append("intercept_not_resolved_within_2SE")
    if not np.isfinite(r2):
        flags.append("R2_undefined_constant_y")
    elif r2 < 0.98:
        flags.append("R2_below_0.98_review")
    return {
        "N_devices": n, "N_distinct_lengths": len(np.unique(x)), "DOF": dof,
        "Rsheet_ohm_sq": slope, "Rsheet_SE_ohm_sq": slope_se,
        "Intercept_total_RcW_ohm_um": intercept,
        "Intercept_total_RcW_SE_ohm_um": intercept_se,
        "Intercept_total_RcW_ohm_cm": intercept * 1e-4,
        "Intercept_total_RcW_SE_ohm_cm": intercept_se * 1e-4,
        "RcW_one_contact_ohm_um": intercept / 2.0,
        "RcW_one_contact_SE_ohm_um": intercept_se / 2.0,
        "RcW_one_contact_ohm_cm": intercept * 1e-4 / 2.0,
        "RcW_one_contact_SE_ohm_cm": intercept_se * 1e-4 / 2.0,
        "R2": r2, "RMSE_ohm_um": float(np.sqrt(sse / n)),
        "Fit_status": "; ".join(flags) if flags else "OK",
    }


def _tlm_analyze(parsed_sheets, mapping, gate_values, pass_index, target_vd,
                 allow_interpolation=True, min_current=0.0, compliance=0.0):
    """Extract every selected device at every VG. Never silently fit a smaller subset."""
    records, parameters, raw_output = [], [], []
    for vg in gate_values:
        gate_records = []
        for device in mapping:
            record = {
                "Sheet": device["Sheet"], "Length_um": float(device["Length_um"]),
                "Width_um": float(device["Width_um"]), "VG_V": float(vg),
                "Sweep": pass_index + 1, "VD_requested_V": float(target_vd),
                "Used_in_fit": False, "Status": "OK",
            }
            try:
                curve = _tlm_get_curve(parsed_sheets[device["Sheet"]], vg, pass_index)
                for row, vd, current in zip(curve["rows"], curve["vd"], curve["id"]):
                    raw_output.append({
                        "Sheet": device["Sheet"], "Length_um": device["Length_um"],
                        "Width_um": device["Width_um"], "VG_V": vg,
                        "Sweep": pass_index + 1, "Column_group": curve["group"],
                        "Excel_row": int(row), "DrainV_V": float(vd), "DrainI_A": float(current),
                    })
                record.update(_tlm_sample_at_vd(
                    curve, target_vd, allow_interpolation, min_current, compliance
                ))
                if device["Width_um"] <= 0 or not np.isfinite(device["Width_um"]):
                    raise ValueError("Width는 0보다 커야 합니다.")
                record["RW_ohm_um"] = record["Rtotal_ohm"] * device["Width_um"]
                record["RW_ohm_cm"] = record["RW_ohm_um"] * 1e-4
                if device["Length_um"] <= 0 or not np.isfinite(device["Length_um"]):
                    raise ValueError("Length를 입력해야 합니다 (0보다 큰 값).")
            except (ValueError, KeyError, IndexError) as exc:
                record["Status"] = str(exc)
            gate_records.append(record)
        summary = {
            "VG_V": float(vg), "VD_requested_V": float(target_vd), "Sweep": pass_index + 1,
            "N_selected_devices": len(mapping), "N_devices": 0,
        }
        if len(gate_records) >= 2 and all(r["Status"] == "OK" for r in gate_records):
            try:
                fit = _tlm_fit_line(
                    [r["Length_um"] for r in gate_records], [r["RW_ohm_um"] for r in gate_records]
                )
                summary.update(fit)
                for record in gate_records:
                    record["Used_in_fit"] = True
                    prediction = fit["Rsheet_ohm_sq"] * record["Length_um"] + fit["Intercept_total_RcW_ohm_um"]
                    record["RW_fit_ohm_um"] = prediction
                    record["Residual_ohm_um"] = record["RW_ohm_um"] - prediction
            except ValueError as exc:
                summary["Fit_status"] = "NOT_FITTED: " + str(exc)
        else:
            summary["Fit_status"] = (
                "NOT_FITTED: 최소 2개 시트/서로 다른 길이가 필요합니다."
                if len(gate_records) < 2 else
                "NOT_FITTED: 선택한 소자 중 입력/전류 추출 오류가 있습니다. 아래 상세 표를 확인하세요."
            )
        records.extend(gate_records)
        parameters.append(summary)
    return pd.DataFrame(records), pd.DataFrame(parameters), pd.DataFrame(raw_output)



# Presentation-only names: internal keys and the input-file header parser stay ASCII.
# Dataframe/CSV headers use full words because they do not render HTML subscripts.
_TLM_TABLE_LABELS = {
    "Sheet": "Sheet", "Length_um": "Length (μm)", "Width_um": "Width (μm)",
    "VG_V": "Gate Voltage (V)", "VD_requested_V": "Reference Drain Voltage (V)",
    "VD_used_V": "Used Drain Voltage (V)", "ID_A": "Drain Current (A)",
    "Rtotal_ohm": "Total Resistance (Ω)", "RW_ohm_um": "Total Resistance × Width (Ω·μm)",
    "RW_ohm_cm": "Total Resistance × Width (Ω·cm)",
    "RW_fit_ohm_um": "Fitted Resistance × Width (Ω·μm)",
    "Residual_ohm_um": "Residual (Ω·μm)",
    "Extraction": "Extraction Method", "N_contributing_points": "Contributing Points",
    "VD_lower_V": "Lower Drain Voltage (V)", "VD_upper_V": "Upper Drain Voltage (V)",
    "ID_lower_A": "Current at Lower Voltage (A)", "ID_upper_A": "Current at Upper Voltage (A)",
    "VD_ID_same_sign": "Drain Voltage / Current Same Sign", "Used_in_fit": "Used in Fit",
    "N_selected_devices": "Selected Devices", "N_devices": "Fitted Devices",
    "N_distinct_lengths": "Distinct Lengths", "DOF": "Degrees of Freedom",
    "Rsheet_ohm_sq": "Sheet Resistance (Ω/sq)",
    "Rsheet_SE_ohm_sq": "Sheet Resistance SE (Ω/sq)",
    "Intercept_total_RcW_ohm_um": "Total Contact Resistance × Width (Ω·μm)",
    "Intercept_total_RcW_ohm_cm": "Total Contact Resistance × Width (Ω·cm)",
    "Intercept_total_RcW_SE_ohm_um": "Total Contact Resistance × Width SE (Ω·μm)",
    "Intercept_total_RcW_SE_ohm_cm": "Total Contact Resistance × Width SE (Ω·cm)",
    "RcW_one_contact_ohm_um": "One-Contact Resistance × Width (Ω·μm)",
    "RcW_one_contact_ohm_cm": "One-Contact Resistance × Width (Ω·cm)",
    "RcW_one_contact_SE_ohm_um": "One-Contact Resistance × Width SE (Ω·μm)",
    "RcW_one_contact_SE_ohm_cm": "One-Contact Resistance × Width SE (Ω·cm)",
    "R2": "R²", "RMSE_ohm_um": "Fit RMSE (Ω·μm)", "Fit_status": "Fit Status",
    "Gate_values_V": "Measured Gate Voltages (V)", "Column_group": "Column Group",
    "Excel_row": "Excel Row", "DrainV_V": "Drain Voltage (V)", "DrainI_A": "Drain Current (A)",
    "Mean": "Arithmetic Mean", "N_gate_values": "Averaged Gate Voltages",
    "Included_gates_V": "Included Gate Voltages (V)",
    "N_selected_gates": "Selected Gate Voltages", "Parameter": "Parameter",
}
_TLM_STATUS_LABELS = {
    "measured": "Measured point",
    "linear_interpolation": "Linear interpolation within one output sweep",
    "duplicate_mean_same_pass": "Mean of duplicate voltage points within one sweep",
    "only_2_distinct_lengths": "Only 2 distinct lengths",
    "nonpositive_Rsheet": "Nonpositive sheet resistance",
    "negative_intercept": "Negative intercept",
    "intercept_not_resolved_within_2SE": "Intercept magnitude is within 2 SE of zero",
    "R2_undefined_constant_y": "R² undefined (constant resistance × width)",
    "R2_below_0.98_review": "R² below 0.98 (review flag only)",
    "NOT_FITTED": "Not fitted",
}
_TLM_AVERAGE_OPTION = "__average_of_selected_gate_fits__"
_TLM_AVERAGE_FIELDS = (
    "Intercept_total_RcW_ohm_um", "Intercept_total_RcW_ohm_cm",
    "RcW_one_contact_ohm_um", "RcW_one_contact_ohm_cm", "Rsheet_ohm_sq", "R2",
)


def _tlm_upper_exponents(text):
    """Uppercase numerical scientific notation, not words or identifiers."""
    return re.sub(r"(?<=\d)e(?=[+-]?\d)", "E", str(text))


def _tlm_plain_text(text):
    """Readable text for tables/exports, without changing stored numerical data."""
    text = str(text)
    for old, new in _TLM_STATUS_LABELS.items():
        text = text.replace(old, new)
    for old, new in (("V_D", "Drain voltage"), ("I_D", "Drain current"),
                     ("V_G", "Gate voltage")):
        text = text.replace(old, new)
    return _tlm_upper_exponents(text)


def _tlm_markdown_text(text):
    """Format diagnostic messages originating in unchanged numerical functions."""
    text = str(text)
    for old, new in _TLM_STATUS_LABELS.items():
        text = text.replace(old, new)
    for old, new in (("V_D", r"$V_{\mathrm{D}}$"), ("I_D", r"$I_{\mathrm{D}}$"),
                     ("V_G", r"$V_{\mathrm{G}}$")):
        text = text.replace(old, new)
    return _tlm_upper_exponents(text)


def _tlm_table_label(column):
    if column in _TLM_TABLE_LABELS:
        return _TLM_TABLE_LABELS[column]
    match = re.fullmatch(r"(RW|Fit)_VG_(.+)V_ohm_(cm|um)", str(column))
    if match:
        prefix, tag, unit = match.groups()
        value = tag.replace("m", "-").replace("p", "+").replace("d", ".")
        try:
            value = f"{float(value):.8G}"
        except ValueError:
            pass
        quantity = "Resistance × Width" if prefix == "RW" else "Fit"
        return f"{quantity} | Gate {value} V (Ω·{'μm' if unit == 'um' else 'cm'})"
    return str(column).replace("_", " ")


def _tlm_present_frame(frame):
    """Use display names only on a copy. Numeric dtypes and precision survive."""
    shown = frame.copy()
    for column in ("Status", "Fit_status", "Extraction"):
        if column in shown:
            shown[column] = shown[column].map(
                lambda value: _tlm_plain_text(value) if isinstance(value, str) else value
            )
    if "Parameter" in shown:
        shown["Parameter"] = shown["Parameter"].map(_tlm_table_label)
    return shown.rename(columns={c: _tlm_table_label(c) for c in shown})


def _tlm_show_table(frame):
    shown = _tlm_present_frame(frame)
    formats = {
        col: (lambda value: f"{float(value):.9G}" if np.isfinite(value) else "N/A")
        for col in shown.select_dtypes(include=["floating"]).columns
    }
    # Styler changes displayed strings only; sorting/exported numbers remain numeric.
    st.dataframe(shown.style.format(formats, na_rep="N/A"),
                 use_container_width=True, hide_index=True)


def _tlm_store_widget(widget_key, saved_key):
    st.session_state[saved_key] = st.session_state[widget_key]


def _tlm_checkbox_list(label, options, prefix, format_func=str):
    """Stable per-item keys keep choices when available sheets/gates change."""
    selected = []
    with st.sidebar.expander(label, expanded=True):
        for option in options:
            token = hashlib.sha256(str(option).encode("utf-8")).hexdigest()[:16]
            key = f"{prefix}_{token}"
            saved = f"saved_{key}"
            if saved not in st.session_state:
                st.session_state[saved] = bool(st.session_state.get(key, True))
            st.session_state[key] = st.session_state[saved]
            checked = st.checkbox(format_func(option), key=key,
                                  on_change=_tlm_store_widget, args=(key, saved))
            st.session_state[saved] = checked
            if checked:
                selected.append(option)
    return selected


def _tlm_geometry_input(label, *, key, value, **kwargs):
    """Retain numeric settings when a sheet/workbook temporarily hides a widget."""
    saved = f"saved_{key}"
    if saved not in st.session_state:
        st.session_state[saved] = float(st.session_state.get(key, value))
    st.session_state[key] = st.session_state[saved]
    result = st.sidebar.number_input(label, key=key, on_change=_tlm_store_widget,
                                     args=(key, saved), **kwargs)
    st.session_state[saved] = result
    return result


def _tlm_average_parameters(parameters):
    """Unweighted mean of finite, separately fitted gate parameters; never refit.

    A review flag (e.g. a negative intercept) does not silently exclude a fit.
    Failed fits are unavailable. SEs are deliberately NOT averaged.
    """
    fitted = parameters[
        parameters.get("Rsheet_ohm_sq", pd.Series(np.nan, index=parameters.index)).notna()
    ]
    result, rows = {}, []
    for field in _TLM_AVERAGE_FIELDS:
        values = pd.to_numeric(fitted.get(field, pd.Series(np.nan, index=fitted.index)),
                               errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=float))
        result[field] = float(values[finite].mean()) if np.any(finite) else np.nan
        included = fitted.loc[finite, "VG_V"] if "VG_V" in fitted else []
        rows.append({"Parameter": field, "Mean": result[field],
                     "N_gate_values": int(np.sum(finite)), "N_selected_gates": len(parameters),
                     "Included_gates_V": ", ".join(f"{g:.8G}" for g in included)})
    result["N_gate_fits"] = len(fitted)
    result["N_R2_values"] = rows[-1]["N_gate_values"]
    return result, pd.DataFrame(rows)


def _tlm_csv_bytes(frame):
    # Keep numeric columns numeric. Escape only potentially executable text cells.
    safe = _tlm_present_frame(frame)
    for col in safe.select_dtypes(include=["object", "string"]).columns:
        safe[col] = safe[col].map(
            lambda x: "'" + x if isinstance(x, str) and x.startswith(("=", "+", "-", "@")) else x
        )
    return safe.to_csv(index=False, float_format="%.15G", lineterminator="\n").encode("utf-8-sig")


def _tlm_export_points(extracted, mapping, gate_values, unit):
    """Origin XY table only; no fitted coordinates, parameters or pooled averages.

    Sort measured devices by length; retain separate rows for duplicate lengths.
    Export every available extracted RW, even when a gate could not be fitted.
    Missing/failed extractions are blank cells, never fabricated zeros.
    """
    scale = 1e-4 if unit == "Ω·cm" else 1.0
    devices = sorted(mapping, key=lambda r: (r["Length_um"], r["Sheet"]))
    points = pd.DataFrame({"Channel Length (μm)": [d["Length_um"] for d in devices]})
    for vg in gate_values:
        subset = extracted[extracted["VG_V"] == vg]
        values = []
        for device in devices:
            found = subset[subset["Sheet"] == device["Sheet"]]
            value = found["RW_ohm_um"].iloc[0] if len(found) and "RW_ohm_um" in found else np.nan
            values.append(float(value) * scale if np.isfinite(value) else np.nan)
        points[f"RtotalW ({unit}) | Gate {float(vg):.8G} V"] = values
    return points


def _tlm_card(title, value, color):
    # Slightly larger TLM-only text; retain the original palette and spacing.
    return f"""
    <div style='text-align: left; padding: 5px 0;'>
        <p style='font-size: 22px; margin-bottom: 5px; color: #555;'>{title}</p>
        <p style='font-size: 30px; font-weight: bold; color: {color}; margin: 0; line-height: 1.2;'>{value}</p>
    </div>
    """


def _tlm_format(value, unit=""):
    if not np.isfinite(value):
        return "N/A"
    number = f"{value:.2f}" if value == 0 or 0.01 <= abs(value) < 1e5 else f"{value:.3E}"
    return number + (" " + unit if unit else "")


def _tlm_style_figure(fig, x_title, y_title, height=650):
    # Deliberately use the original Plotly template/axis/font/legend settings.
    common_axis_params = dict(
        ticks="outside", tickwidth=1.5, tickcolor='black', ticklen=8,
        showline=True, linewidth=1.5, linecolor='black', mirror=True,
        showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='dot',
        zeroline=False, layer='below traces',
        title_font=dict(size=22), tickfont=dict(size=15)
    )
    fig.update_layout(
        width=1000, height=height, autosize=False, template="plotly_white",
        margin=dict(t=70, b=85, l=110, r=65),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1,
                    xanchor="left", yanchor="top", x=0.015, y=0.985,
                    font=dict(color="black", size=14)),
        hovermode="closest",
    )
    fig.update_xaxes(title_text=x_title, **common_axis_params)
    fig.update_yaxes(title_text=y_title, exponentformat="power", **common_axis_params)
    return fig


def _tlm_main_figure(extracted, parameters, gate_values, unit, extend_to_zero):
    scale = 1e-4 if unit == "Ω·cm" else 1.0
    fig = go.Figure()
    for j, vg in enumerate(gate_values):
        color, symbol = _TLM_COLORS[j % len(_TLM_COLORS)], _TLM_SYMBOLS[j % len(_TLM_SYMBOLS)]
        data = extracted[extracted["VG_V"] == vg].copy()
        if "RW_ohm_um" not in data:
            continue
        data = data[np.isfinite(data["RW_ohm_um"]) & (data["Length_um"] > 0)].sort_values("Length_um")
        if data.empty:
            continue
        group = f"gate_{vg}"
        fig.add_trace(go.Scatter(
            x=data["Length_um"], y=data["RW_ohm_um"] * scale, mode="markers",
            name=f"V<sub>G</sub> = {vg:G} V", legendgroup=group,
            marker=dict(color=color, symbol=symbol, size=11, line=dict(width=1, color=color)),
            customdata=np.column_stack([
                data["Sheet"].map(lambda name: html.escape(str(name))),
                data["ID_A"].map(lambda value: f"{value:.6G}"),
                data["VD_used_V"].map(lambda value: f"{value:.6G}"),
                data["Length_um"].map(lambda value: f"{value:.6G}"),
                (data["RW_ohm_um"] * scale).map(lambda value: f"{value:.6G}"),
            ]),
            hovertemplate=("L = %{customdata[3]} μm<br>R<sub>total</sub>W = %{customdata[4]} " + unit +
                           "<br>Sheet: %{customdata[0]}<br>I<sub>D</sub> = %{customdata[1]} A" +
                           "<br>V<sub>D</sub> = %{customdata[2]} V<extra>%{fullData.name}</extra>"),
        ))
        result = parameters[parameters["VG_V"] == vg].iloc[0]
        if pd.notna(result.get("Rsheet_ohm_sq")):
            xmin, xmax = float(data["Length_um"].min()), float(data["Length_um"].max())
            xx = np.linspace(xmin, xmax, 101)
            yy = (result["Rsheet_ohm_sq"] * xx + result["Intercept_total_RcW_ohm_um"]) * scale
            fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", line=dict(color=color, width=1.5),
                                     name=f"Fit V<sub>G</sub> = {vg:G} V", legendgroup=group,
                                     showlegend=False, hoverinfo="skip"))
            if extend_to_zero and xmin > 0:
                xx = np.array([0.0, xmin])
                yy = (result["Rsheet_ohm_sq"] * xx + result["Intercept_total_RcW_ohm_um"]) * scale
                fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines",
                                         line=dict(color=color, width=1.2, dash="dot"),
                                         name="Extrapolation", legendgroup=group,
                                         showlegend=False, hoverinfo="skip"))
    return _tlm_style_figure(fig, "Channel Length (μm)", f"R<sub>total</sub>W ({unit})")


def _tlm_preview_outputs(parsed_sheets, selected_sheets, mapping, gate_values,
                         pass_index, target_vd, file_key):
    with st.expander(r"Output Curves : 선택한 $V_{\mathrm{D}}$와 선형 영역 확인", expanded=False):
        preview_sheet = st.selectbox("Preview sheet", selected_sheets, key=f"tlm_preview_{file_key}")
        fig = go.Figure()
        for j, vg in enumerate(gate_values):
            try:
                curve = _tlm_get_curve(parsed_sheets[preview_sheet], vg, pass_index)
            except ValueError:
                continue
            fig.add_trace(go.Scatter(
                x=curve["vd"], y=curve["id"], mode="lines+markers",
                line=dict(color=_TLM_COLORS[j % len(_TLM_COLORS)], width=1.5),
                marker=dict(size=4), name=f"V<sub>G</sub> = {vg:G} V",
                customdata=np.column_stack([
                    [f"{value:.6G}" for value in curve["vd"]],
                    [f"{value:.6G}" for value in curve["id"]],
                ]),
                hovertemplate=("V<sub>D</sub> = %{customdata[0]} V<br>"
                               "I<sub>D</sub> = %{customdata[1]} A<extra>%{fullData.name}</extra>"),
            ))
        if np.isfinite(target_vd):
            fig.add_vline(x=target_vd, line_dash="dash", line_color="#555", line_width=1.5)
        _tlm_style_figure(fig, "Drain Voltage (V)", "Drain Current (A)", height=550)
        st.plotly_chart(fig, use_container_width=False, key=f"tlm_output_plot_{file_key}")
        st.caption(r"원본의 부호를 유지한 $I_{\mathrm{D}}$–$V_{\mathrm{D}}$입니다. 회색 점선은 선택한 기준 $V_{\mathrm{D}}$입니다. 드래그하여 확대할 수 있습니다.")


def _tlm_saved_selectbox(label, options, *, key, sidebar=False, **kwargs):
    """Retain a choice when switching workbooks temporarily removes its widget."""
    options = list(options)
    if not options:
        raise ValueError("선택 가능한 항목이 없습니다.")
    saved = f"saved_{key}"
    previous = st.session_state.get(saved, st.session_state.get(key))
    if previous not in options:
        previous = options[0]
    st.session_state[key] = previous
    ui = st.sidebar if sidebar else st
    result = ui.selectbox(label, options, key=key, on_change=_tlm_store_widget,
                          args=(key, saved), **kwargs)
    st.session_state[saved] = result
    return result


def _tlm_saved_checkbox(label, *, key):
    saved = f"saved_{key}"
    st.session_state[key] = bool(st.session_state.get(saved, st.session_state.get(key, False)))
    result = st.sidebar.checkbox(label, key=key, on_change=_tlm_store_widget,
                                 args=(key, saved))
    st.session_state[saved] = result
    return result


def _tlm_select_uploaded_file(uploaded_files):
    """Select one workbook. Never combine sheets from different uploaded files.

    Key by filename AND content, not filename alone. Identically named uploads
    remain selectable; two identical uploads also receive separate session keys.
    """
    choices, occurrences = {}, {}
    name_counts = {}
    for uploaded in uploaded_files:
        name_counts[uploaded.name] = name_counts.get(uploaded.name, 0) + 1
    name_indices = {}
    for uploaded in uploaded_files:
        data_bytes = uploaded.getvalue()
        digest = hashlib.sha256(data_bytes).hexdigest()
        base = hashlib.sha256((uploaded.name + "\0" + digest).encode("utf-8")).hexdigest()[:24]
        occurrence = occurrences.get(base, 0)
        occurrences[base] = occurrence + 1
        token = base if occurrence == 0 else f"{base}_{occurrence + 1}"
        name_indices[uploaded.name] = name_indices.get(uploaded.name, 0) + 1
        label = uploaded.name
        if name_counts[uploaded.name] > 1:
            label += f" (File {name_indices[uploaded.name]})"
        choices[token] = (uploaded, data_bytes, digest, label)
    if len(choices) > 1:
        token = _tlm_saved_selectbox(
            "📁 Select Excel File", list(choices), key="tlm_selected_workbook", sidebar=True,
            format_func=lambda item: choices[item][3],
            help="현재 선택한 파일의 Data / Append 시트만 분석합니다. 서로 다른 파일의 데이터는 합치지 않습니다."
        )
    else:
        token = next(iter(choices))
        st.session_state["saved_tlm_selected_workbook"] = token
    uploaded, data_bytes, digest, _ = choices[token]
    return uploaded, data_bytes, digest, token


def _tlm_parameter_ui_style():
    """Scope typography to this selectbox, including its portalled dropdown menu.

    st-key-* selectors handle keyed Streamlit widgets; :has(input[aria-label])
    supplies a fallback. No global selectbox/paragraph/Plotly style is changed.
    """
    st.markdown("""
<style>
:is([class*="st-key-tlm_summary_gate_"],
    [data-testid="stSelectbox"]:has(input[aria-label^="Parameter display"]))
    [data-testid="stWidgetLabel"] p {
    font-size: 22px !important;
    font-weight: 600 !important;
}
:is([class*="st-key-tlm_summary_gate_"],
    [data-testid="stSelectbox"]:has(input[aria-label^="Parameter display"]))
    [data-testid="stWidgetLabel"] .katex {
    font-size: 1em !important;
}
:is([class*="st-key-tlm_summary_gate_"],
    [data-testid="stSelectbox"]:has(input[aria-label^="Parameter display"]))
    [data-baseweb="select"] :is(div, input, span) {
    font-size: 20px !important;
}
:is([class*="st-key-tlm_summary_gate_"],
    [data-testid="stSelectbox"]:has(input[aria-label^="Parameter display"]))
    [data-baseweb="select"] > div {
    min-height: 48px;
}
/* BaseWeb renders the options outside the widget. Enlarge only while this
   specific combobox is open; all sidebar and FET dropdowns stay unchanged. */
body:has(:is([class*="st-key-tlm_summary_gate_"],
    [data-testid="stSelectbox"]:has(input[aria-label^="Parameter display"]))
    [aria-expanded="true"])
    :is([data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"])
    :is([role="option"], [role="option"] div, [role="option"] span,
        [data-testid="stMarkdownContainer"] p) {
    font-size: 20px !important;
}
</style>
""", unsafe_allow_html=True)


def run_tlm_analysis():
    _tlm_parameter_ui_style()
    st.sidebar.header("TLM Information")
    st.sidebar.markdown("**Output Curve → TLM**")
    common_width = st.sidebar.number_input(
        "Width (μm)", min_value=0.001, value=1000.0, step=50.0, format="%.3f", key="tlm_common_width"
    )
    unit = st.sidebar.selectbox(r"$R_{\mathrm{total}}W$ / $R_{\mathrm{c}}W$ Unit", ["Ω·cm", "Ω·μm"], key="tlm_unit")
    st.sidebar.markdown("---")
    st.markdown("<h3 style='color: #333;'>📊 TLM Fitting (Output Curves)</h3>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Data / Append 시트에 서로 다른 채널 길이의 Output Curve를 저장한 Excel 파일을 업로드하세요",
        type=["xls", "xlsx"], accept_multiple_files=True, key="tlm_file_upload"
    )
    if not uploaded_files:
        st.info("Excel 파일을 업로드한 뒤, 왼쪽에서 사용할 파일·시트와 각 시트의 Length를 입력하세요. 여러 파일도 업로드할 수 있습니다.")
        st.markdown(
            "예: **Data → 2 μm / Append1 → 2.5 μm / Append2 → 5 μm**. "
            "각 시트 안의 여러 $V_{\\mathrm{G}}$ output 곡선은 자동으로 구분합니다."
        )
        return
    uploaded, data_bytes, digest, file_key = _tlm_select_uploaded_file(uploaded_files)
    # Session-local in-memory parsing cache; no file is written to the server.
    cached = st.session_state.get("tlm_parsed_workbook")
    if cached is None or cached["digest"] != digest:
        try:
            with st.spinner("Output 측정 시트를 읽는 중입니다..."):
                workbook = _tlm_load_excel(data_bytes)
        except ImportError as exc:
            st.error("Excel 읽기 패키지가 없습니다. requirements.txt의 xlrd와 openpyxl을 설치하세요.")
            st.code("python -m pip install xlrd openpyxl", language="bash")
            st.caption(str(exc))
            return
        except Exception as exc:
            st.error(f"Excel 파일을 읽지 못했습니다: {exc}")
            return
        st.session_state["tlm_parsed_workbook"] = {"digest": digest, "workbook": workbook}
    else:
        workbook = cached["workbook"]
    parsed_sheets = workbook["sheets"]
    if workbook["errors"]:
        with st.expander("읽지 못한 측정 시트", expanded=True):
            for sheet, error in workbook["errors"].items():
                st.warning(_tlm_markdown_text(f"{sheet}: {error}"))
    if not parsed_sheets:
        st.error("분석 가능한 Data / Append 시트가 없습니다. Calc와 Settings는 TLM 길이 데이터로 사용하지 않습니다.")
        return
    selected_sheets = _tlm_checkbox_list(
        "📂 Select Data Sheets", list(parsed_sheets), f"tlm_sheet_check_{file_key}"
    )
    if not selected_sheets:
        st.info("왼쪽에서 분석할 시트를 선택하세요.")
        return
    individual_width = _tlm_saved_checkbox("시트마다 다른 Width 입력", key=f"tlm_width_mode_{file_key}")
    st.sidebar.markdown("**Sheet → Channel Length**")
    st.sidebar.caption("Length의 0은 미입력 상태입니다. 실제 측정 길이를 직접 입력하세요.")
    mapping = []
    for sheet in selected_sheets:
        sheet_key = hashlib.sha256(sheet.encode("utf-8")).hexdigest()[:12]
        length = _tlm_geometry_input(
            f"{sheet} — Length (μm)", min_value=0.0, value=0.0, step=0.1,
            format="%.3f", key=f"tlm_L_{file_key}_{sheet_key}"
        )
        width = common_width
        if individual_width:
            width = _tlm_geometry_input(
                f"{sheet} — Width (μm)", min_value=0.001, value=float(common_width), step=50.0,
                format="%.3f", key=f"tlm_W_{file_key}_{sheet_key}"
            )
        mapping.append({"Sheet": sheet, "Length_um": float(length), "Width_um": float(width)})
    gates = _tlm_common_gates(parsed_sheets, selected_sheets)
    st.sidebar.markdown("---")
    if not gates:
        st.error("선택한 모든 시트에 공통으로 측정된 $V_{\\mathrm{G}}$가 없습니다. 게이트 전압은 임의로 보간하지 않습니다.")
        with st.expander("시트별 측정 $V_{\\mathrm{G}}$", expanded=True):
            for sheet in selected_sheets:
                st.write(sheet, ", ".join(f"{g:G} V" for g in parsed_sheets[sheet]["curves"]))
        return
    selected_gates = _tlm_checkbox_list(
        r"Gate Voltages $V_{\mathrm{G}}$ (V)", gates, f"tlm_gate_check_{file_key}",
        format_func=lambda g: f"{g:G} V"
    )
    if not selected_gates:
        st.info("왼쪽에서 피팅할 $V_{\\mathrm{G}}$를 하나 이상 선택하세요.")
        return
    max_passes = max(
        len(parsed_sheets[s]["curves"][g]) for s in selected_sheets
        for g in parsed_sheets[s]["curves"] if any(abs(g - h) <= _TLM_VOLTAGE_ATOL for h in selected_gates)
    )
    pass_index = _tlm_saved_selectbox(
        "Output Sweep", list(range(max_passes)), sidebar=True,
        format_func=lambda j: f"Sweep {j + 1}" + (" (First Acquired)" if j == 0 else ""),
        key=f"tlm_pass_{file_key}",
        help=r"같은 $V_{\mathrm{G}}$에서 취득한 드레인 전압 스캔의 순서입니다. 예: 0 → −5 → 0 V이면 Sweep 1은 0 → −5 V, Sweep 2는 −5 → 0 V입니다. 왕복/반복 sweep을 자동으로 평균하지 않습니다."
    )
    curves_for_reference = []
    for sheet in selected_sheets:
        for vg in selected_gates:
            try:
                curves_for_reference.append(_tlm_get_curve(parsed_sheets[sheet], vg, pass_index))
            except ValueError:
                pass  # The extraction table will report this missing pass explicitly.
    if not curves_for_reference:
        st.error("선택한 sweep의 output 데이터가 없습니다.")
        return
    lower = max(float(c["vd"].min()) for c in curves_for_reference)
    upper = min(float(c["vd"].max()) for c in curves_for_reference)
    first_vd = np.unique(curves_for_reference[0]["vd"])
    candidates = [float(v) for v in first_vd if abs(v) > 1e-9 and lower - 1e-8 <= v <= upper + 1e-8]
    default_vd = float(f"{min(candidates, key=abs):.7g}") if candidates else -0.1
    target_vd = _tlm_geometry_input(
        r"Reference $V_{\mathrm{D}}$ (V)", value=default_vd, step=0.001, format="%.3f",
        key=f"tlm_vd_{file_key}", help="0 V는 사용할 수 없습니다. 모든 소자의 저전압 선형영역에서 선택하세요."
    )
    st.sidebar.caption(f"선택 sweep의 공통 $V_{{\\mathrm{{D}}}}$ 범위: {lower:G} ~ {upper:G} V")
    interpolate = st.sidebar.checkbox(
        "측정점 사이 선형 보간 허용", value=True, key="tlm_interpolate",
        help=r"선택한 $V_{\mathrm{D}}$에 측정점이 없을 때, 같은 시트·같은 $V_{\mathrm{G}}$·같은 sweep의 양옆 두 전압점으로 $I_{\mathrm{D}}$를 추정합니다. 채널 길이별 TLM 직선 피팅과는 별개입니다. 측정 범위 밖 외삽은 하지 않습니다."
    )
    extend_to_zero = st.sidebar.checkbox("피팅선을 L = 0까지 표시", value=True, key="tlm_extend_zero")
    with st.sidebar.expander("Quality checks", expanded=False):
        min_current = st.number_input(
            r"Minimum $|I_{\mathrm{D}}|$ (A)", min_value=0.0, value=0.0, step=1e-12, format="%.3e", key="tlm_min_id",
            help=r"사용할 최소 전류 크기입니다. $|I_{\mathrm{D}}|$가 이 값 이하인 점은 사용할 수 없습니다. 0이면 전류가 정확히 0인 점만 제외합니다."
        )
        compliance = st.number_input(
            "Current compliance (A; 0 = 미설정)", min_value=0.0, value=0.0, step=0.001,
            format="%.6f", key="tlm_compliance",
            help="입력 시 선택점/보간 끝점이 한계의 99% 이상이면 피팅에서 사용하지 않습니다. 모든 선택 시트에 공통 적용됩니다."
        )
        limit = workbook["metadata"].get("latest_drain_compliance_A")
        if limit is not None:
            st.caption(
                f"Settings에 기재된 Drain compliance: {limit:G} A. 최신 실행의 설정일 수 있으므로 "
                "모든 Append에 자동 적용하지 않습니다."
            )
    st.markdown(
        f"**File:** {html.escape(uploaded.name)} · **Selected sheets:** {len(selected_sheets)} · "
        f"**V<sub>D</sub>:** {target_vd:.3f} V · **Sweep:** {pass_index + 1}", unsafe_allow_html=True
    )
    st.info(
        "각 $V_{\\mathrm{G}}$에서 $R_{\\mathrm{total}}$ = |$V_{\\mathrm{D}}$ / $I_{\\mathrm{D}}$|를 구한 뒤 $R_{\\mathrm{total}}$ × W 대 L을 피팅합니다. "
        "모든 길이에서 저전압 선형영역인지 Output Curves에서 확인하세요. "
        "여기서 구하는 값은 선택한 $V_{\\mathrm{D}}$에 대한 저항이며, 비선형 구간에서는 겉보기 저항입니다."
    )
    _tlm_preview_outputs(parsed_sheets, selected_sheets, mapping, selected_gates, pass_index, target_vd, file_key)
    notes = [note for s in selected_sheets for note in parsed_sheets[s]["notes"]]
    gate_sets_differ = any(len(parsed_sheets[s]["curves"]) != len(gates) for s in selected_sheets)
    with st.expander("Sheet Mapping : 인식된 데이터 확인", expanded=any(d["Length_um"] <= 0 for d in mapping)):
        inspection = []
        for d in mapping:
            inspection.append({**d, "Gate_values_V": ", ".join(f"{g:G}" for g in parsed_sheets[d["Sheet"]]["curves"])})
        _tlm_show_table(pd.DataFrame(inspection))
        if gate_sets_differ:
            st.warning("시트별 $V_{\\mathrm{G}}$ 목록이 달라 모든 선택 시트에 공통인 $V_{\\mathrm{G}}$만 선택창에 표시했습니다.")
        for note in notes:
            st.warning(_tlm_markdown_text(note))
    if any(d["Length_um"] <= 0 for d in mapping):
        st.warning("왼쪽에서 각 시트의 실제 Length를 입력하세요. 0은 미입력 상태이며, 길이를 임의로 가정하지 않습니다.")
    if len(selected_sheets) < 2:
        st.warning("현재는 측정 시트가 1개입니다. 전류/저항 확인은 가능하지만 TLM 피팅에는 서로 다른 길이의 시트가 최소 2개 필요합니다.")
    else:
        n_lengths = len({d["Length_um"] for d in mapping if d["Length_um"] > 0})
        if n_lengths < 2:
            st.warning("유효한 서로 다른 채널 길이가 최소 2개 필요합니다. 각 시트의 Length를 확인하세요.")
        elif n_lengths == 2:
            st.warning("서로 다른 길이가 2개뿐이면 직선성 검증이 어렵습니다. 3개 이상 길이로 측정하는 것을 권합니다.")
    extracted, parameters, raw_output = _tlm_analyze(
        parsed_sheets, mapping, selected_gates, pass_index, target_vd,
        interpolate, min_current, compliance
    )
    valid_fits = parameters[
        parameters.get("Rsheet_ohm_sq", pd.Series(np.nan, index=parameters.index)).notna()
    ]
    failed = extracted[extracted["Status"] != "OK"]
    if len(failed):
        with st.expander("입력 / 전류 추출 확인 필요", expanded=True):
            _tlm_show_table(failed[["Sheet", "VG_V", "Length_um", "Status"]])
    if "VD_ID_same_sign" in extracted and (extracted["VD_ID_same_sign"] == False).any():
        st.warning("일부 점에서 $V_{\\mathrm{D}}$와 $I_{\\mathrm{D}}$의 부호가 다릅니다. 저항 크기는 |$V_{\\mathrm{D}}$/$I_{\\mathrm{D}}$|로 계산했으며 원래 부호는 Extracted Data에서 확인할 수 있습니다.")
    if len(valid_fits):
        st.markdown("<h4 style='color: #6FADCF; font-size: 28px;'>Gate-dependent TLM Fitting</h4>", unsafe_allow_html=True)
        st.plotly_chart(
            _tlm_main_figure(extracted, parameters, selected_gates, unit, extend_to_zero),
            use_container_width=False, key=f"tlm_main_plot_{file_key}"
        )
        st.caption("측정값: 심볼 / 선형 피팅: 실선 / L=0 방향 외삽: 점선. 음수 절편이나 기울기는 강제로 보정하지 않습니다.")
        summary_options = [_TLM_AVERAGE_OPTION] + valid_fits["VG_V"].tolist()
        summary_key = f"tlm_summary_gate_{file_key}"
        summary_saved = f"saved_{summary_key}"
        previous_summary = st.session_state.get(summary_saved, st.session_state.get(summary_key))
        if previous_summary not in summary_options:
            previous_summary = summary_options[0]
        st.session_state[summary_key] = previous_summary
        chosen_vg = st.selectbox(
            r"Parameter display $V_{\mathrm{G}}$ (V)",
            summary_options,
            format_func=lambda g: ("Average (All Selected Gate Voltages)"
                                   if g == _TLM_AVERAGE_OPTION else f"{g:G} V"),
            key=summary_key, on_change=_tlm_store_widget, args=(summary_key, summary_saved)
        )
        st.session_state[summary_saved] = chosen_vg
        is_average = chosen_vg == _TLM_AVERAGE_OPTION
        if is_average:
            result, _ = _tlm_average_parameters(parameters)
        else:
            result = valid_fits[valid_fits["VG_V"] == chosen_vg].iloc[0]
        unit_tag = "ohm_cm" if unit == "Ω·cm" else "ohm_um"
        heading = "TLM Parameters (Average)" if is_average else "TLM Parameters"
        st.markdown(f"<h4 style='color: #6FADCF; font-size: 28px;'>{heading}</h4>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(_tlm_card("Intercept (R<sub>c,total</sub>W)",
                    _tlm_format(result[f"Intercept_total_RcW_{unit_tag}"], unit), "#2E60AB"), unsafe_allow_html=True)
        c2.markdown(_tlm_card("R<sub>c</sub>W (One Contact)",
                    _tlm_format(result[f"RcW_one_contact_{unit_tag}"], unit), "#A23B72"), unsafe_allow_html=True)
        c3.markdown(_tlm_card("Sheet Resistance (R<sub>sheet</sub>)",
                    _tlm_format(result["Rsheet_ohm_sq"], "Ω/sq"), "#F18F01"), unsafe_allow_html=True)
        c4.markdown(_tlm_card("Mean R²" if is_average else "Linear Fit (R²)",
                    f"{result['R2']:.5f}" if np.isfinite(result["R2"]) else "N/A", "#18A558"), unsafe_allow_html=True)
        st.markdown(
            "**절편 전체 = (R<sub>S</sub> + R<sub>D</sub>)W. "
            "두 접촉이 대칭일 때만 단일 접촉의 R<sub>c</sub>W = 절편 / 2입니다.**", unsafe_allow_html=True
        )
        if is_average:
            included = ", ".join(f"{g:G}" for g in valid_fits["VG_V"])
            st.caption(
                f"- 선택한 게이트 전압 {len(selected_gates)}개 중 피팅된 {len(valid_fits)}개의 산술평균입니다. "
                f"포함 전압: {included} V. 체크 해제한 전압과 피팅 실패한 전압은 포함하지 않습니다."
            )
            st.caption(
                "- 서로 다른 게이트 조건의 요약값이며 반복 측정 평균이나 새로운 TLM 피팅 결과가 아닙니다. "
                "평균 R²도 개별 R²의 산술평균입니다."
            )
            if result["N_R2_values"] < len(valid_fits):
                st.caption(f"R²가 정의된 {result['N_R2_values']}개 전압만 평균 R²에 포함했습니다.")
            flagged = valid_fits[valid_fits["Fit_status"] != "OK"]
            if len(flagged):
                st.warning(
                    f"검토 표시가 있는 {len(flagged)}개 게이트 피팅도 평균에 포함되어 있습니다. "
                    "전체 피팅 결과를 확인하고, 제외하려는 게이트 전압은 왼쪽에서 체크 해제하세요."
                )
        else:
            if result["Fit_status"] != "OK":
                st.warning("피팅 검토 항목: " + _tlm_markdown_text(result["Fit_status"]))
    else:
        st.info("유효한 길이/전압/전류 조건이 갖춰지면 $V_{\\mathrm{G}}$별 피팅 그래프와 파라미터가 표시됩니다.")
    with st.expander("All $V_{\\mathrm{G}}$ Fit Parameters : 전체 피팅 결과", expanded=bool(len(valid_fits))):
        _tlm_show_table(parameters)
    with st.expander("Extracted Data : 실제 사용한 $V_{\\mathrm{D}}$, $I_{\\mathrm{D}}$, $R_{\\mathrm{total}}$ W", expanded=False):
        _tlm_show_table(extracted)
    with st.expander("Model : 적용 모델", expanded=False):
        st.latex(r"R_{\mathrm{tot}}W=R_{\mathrm{sheet}}L+(R_S+R_D)W")
        st.markdown(
            "W 정규화는 **$R_{\\mathrm{total}}$을 W로 나누는 것이 아니라 곱하는 것**입니다. "
            "내부 계산은 Ω·μm 대 μm로 수행하므로 기울기가 $R_{\\mathrm{sheet}}$(Ω/sq)입니다. "
            "Ω·cm 대 μm 그래프에서는 표시 기울기 × 10⁴가 $R_{\\mathrm{sheet}}$입니다. "
            "비접촉저항(Ω·cm²), transfer length, contact-corrected mobility는 추가 가정 없이 계산하지 않습니다."
        )
        st.markdown(
            "같은 $V_{\\mathrm{G}}$, 동일한 접촉 구조 및 비교 가능한 채널 특성을 전제로 합니다. "
            "R²가 높아도 선형 접촉이나 정확한 접촉저항이 입증되는 것은 아닙니다. "
            "길이별 $V_{\\mathrm{th}}$ 차이·트랩·접근/배선 저항·compliance 영향을 따로 확인하세요. "
            "보간은 현재 sweep 안에서만 수행하고, $V_{\\mathrm{G}}$는 보간하지 않습니다."
        )
    st.markdown("---")
    st.markdown("<h4 style='color: #333;'>Export to CSV file</h4>", unsafe_allow_html=True)
    points = _tlm_export_points(extracted, mapping, selected_gates, unit)
    no_geometry = any(not np.isfinite(d["Length_um"]) or d["Length_um"] <= 0 for d in mapping)
    source_stem = re.split(r"[\\/]", uploaded.name)[-1].rsplit(".", 1)[0]
    st.download_button(
        "Channel Length & RₜₒₜₐₗW CSV", _tlm_csv_bytes(points),
        file_name=f"{source_stem}_TLM_Origin.csv", mime="text/csv",
        key="tlm_dl_points", disabled=no_geometry
    )
    st.caption(
        "첫 열은 Channel Length (μm), 나머지 열은 체크한 각 게이트 전압의 RₜₒₜₐₗW입니다. "
        "단위는 왼쪽에서 선택한 Ω·cm 또는 Ω·μm을 사용합니다. Origin에서 첫 열을 X, 나머지 열을 Y로 지정하세요. "
        "길이는 오름차순이며 같은 길이의 반복 측정은 별도 행으로 유지합니다. 추출할 수 없는 값은 빈칸입니다."
    )


# The FET numerical code and styles below are unchanged; only visible symbol labels are updated.
# TLM renders its own mode and stops before the original FET code executes.
analysis_mode = st.sidebar.radio(
    "Analysis Mode", ["FET Parameter Analysis", "TLM Fitting"], key="main_analysis_mode"
)
st.sidebar.markdown("---")
if analysis_mode == "TLM Fitting":
    run_tlm_analysis()
    st.stop()

# 2. 소자 파라미터 
st.sidebar.header("Device Information")

# 🌟 Operating Mode 선택 기능 추가 (사이드바 최상단)
operating_mode = st.sidebar.radio("Operating Mode", ["Linear", "Saturation"])
st.sidebar.markdown("---")

W = st.sidebar.number_input("Width (μm)", value=1000.0, step=50.0, format="%.2f")
L = st.sidebar.number_input("Length (μm)", value=100.0, step=10.0, format="%.2f")

# Capacitance 입력 (직접 입력 또는 HfO2 k/두께 계산)
st.sidebar.markdown("**Capacitance (nF/cm²)**")
use_hfo2 = st.sidebar.checkbox("Calculate from dielectric constant (k) and thickness")

if use_hfo2:
    hfo2_k = st.sidebar.number_input("Dielectric constant (k)", value=25.0, step=0.1, format="%.2f")
    hfo2_t_nm = st.sidebar.number_input("Thickness (nm)", value=30.0, step=1.0, format="%.2f")
    eps0 = 8.854e-12
    Cox_nf = (eps0 * hfo2_k / (hfo2_t_nm * 1e-9)) * 1e5
    st.sidebar.markdown(
        f'''
        <div style="
            font-size:16px;
            font-weight:bold;
            color:#333;
            margin-top:10px;
            margin-bottom:15px;
        ">
            Calculated Capacitance: {Cox_nf:.2f} nF/cm²
        </div>
        <hr style="
            border:0;
            border-top:1px solid #cccccc;
            margin:10px 0 15px 0;
        ">
        ''',
        unsafe_allow_html=True
    )
else:
    Cox_nf = st.sidebar.number_input("Capacitance (nF/cm²)", value=34.5, format="%.2f")

Cox = Cox_nf * 1e-9

# 무한대(inf) 값을 0이 아닌 '앞뒤의 정상적인 값'으로 채워 넣는 함수
def fix_inf(gm_array):
    gm_series = pd.Series(gm_array).replace([np.inf, -np.inf], np.nan)
    return gm_series.ffill().bfill().values

# SS 계산 함수 정의
def calculate_ss(id_vals, vg_vals):
    log_id = np.log10(np.abs(id_vals) + 1e-15)
    d_log_id = np.abs(np.gradient(log_id, vg_vals))
    d_log_id_smooth = np.convolve(d_log_id, np.ones(3)/3, mode='same')
    valid_slopes = d_log_id_smooth[np.isfinite(d_log_id_smooth) & (d_log_id_smooth > 0)]
    return (1.0 / np.max(valid_slopes)) * 1000 if len(valid_slopes) > 0 else np.inf

# 큰 글자 카드 UI 함수
def make_card(title, value, color):
    return f"""
    <div style='text-align: left; padding: 5px 0;'>
        <p style='font-size: 20px; margin-bottom: 5px; color: #555;'>{title}</p>
        <p style='font-size: 26px; font-weight: bold; color: {color}; margin: 0; line-height: 1.2;'>{value}</p>
    </div>
    """

# 파라미터 추출 헬퍼 함수 (모드 분기 추가)
def extract_parameters_from_sheet(df, file_id, sheet_name, w, l, cox, mode):
    vg = df['GateV']
    id_raw = df['DrainI']
    vd = df['DrainV'].iloc[0]
    
    if abs(vg.max() - vg.iloc[0]) > abs(vg.min() - vg.iloc[0]):
        peak_idx = vg.idxmax()
    else:
        peak_idx = vg.idxmin()
        
    vg_fwd, id_fwd = vg[:peak_idx+1].reset_index(drop=True), id_raw[:peak_idx+1].reset_index(drop=True)
    vg_bwd, id_bwd = vg[peak_idx:].reset_index(drop=True), id_raw[peak_idx:].reset_index(drop=True)
    
    # 🌟 모드별 수식 분기
    if mode == "Linear":
        gm_fwd_raw = fix_inf(np.gradient(id_fwd.values, vg_fwd.values))
        mobility_fwd_raw = (abs(gm_fwd_raw) * l) / (w * cox * abs(vd))
        
        gm_bwd_raw = fix_inf(np.gradient(id_bwd.values, vg_bwd.values))
        mobility_bwd_raw = (abs(gm_bwd_raw) * l) / (w * cox * abs(vd))
    else: # Saturation
        sqrt_id_fwd = np.sqrt(np.abs(id_fwd.values))
        gm_fwd_raw = fix_inf(np.gradient(sqrt_id_fwd, vg_fwd.values))
        mobility_fwd_raw = (2 * l / (w * cox)) * (gm_fwd_raw ** 2)
        
        sqrt_id_bwd = np.sqrt(np.abs(id_bwd.values))
        gm_bwd_raw = fix_inf(np.gradient(sqrt_id_bwd, vg_bwd.values))
        mobility_bwd_raw = (2 * l / (w * cox)) * (gm_bwd_raw ** 2)

    # 모드별로 세션 상태 분리
    key_fwd = f"val_fwd_{file_id}_{sheet_name}_{mode}"
    key_bwd = f"val_bwd_{file_id}_{sheet_name}_{mode}"
    
    if key_fwd in st.session_state:
        target_vg_fwd = st.session_state[key_fwd]
    else:
        abs_gm_f = np.abs(gm_fwd_raw)
        idx_f_auto = np.argmax(abs_gm_f[2:-2]) + 2 if len(abs_gm_f) > 5 else np.argmax(abs_gm_f)
        target_vg_fwd = float(vg_fwd.iloc[idx_f_auto])

    if key_bwd in st.session_state:
        target_vg_bwd = st.session_state[key_bwd]
    else:
        abs_gm_b = np.abs(gm_bwd_raw)
        idx_b_auto = np.argmax(abs_gm_b[2:-2]) + 2 if len(abs_gm_b) > 5 else np.argmax(abs_gm_b)
        target_vg_bwd = float(vg_bwd.iloc[idx_b_auto])

    # 가장 가까운 전압 값 매칭
    vg_max_gm_fwd = float(vg_fwd.loc[(vg_fwd - target_vg_fwd).abs().idxmin()])
    vg_max_gm_bwd = float(vg_bwd.loc[(vg_bwd - target_vg_bwd).abs().idxmin()])
    
    idx_f = vg_fwd[vg_fwd == vg_max_gm_fwd].index[0]
    idx_b = vg_bwd[vg_bwd == vg_max_gm_bwd].index[0]

    # Vth 및 Mobility 계산
    if mode == "Linear":
        vth_fwd = -id_fwd.iloc[idx_f] / gm_fwd_raw[idx_f] + vg_max_gm_fwd
        vth_bwd = -id_bwd.iloc[idx_b] / gm_bwd_raw[idx_b] + vg_max_gm_bwd
    else:
        vth_fwd = -np.sqrt(abs(id_fwd.iloc[idx_f])) / gm_fwd_raw[idx_f] + vg_max_gm_fwd
        vth_bwd = -np.sqrt(abs(id_bwd.iloc[idx_b])) / gm_bwd_raw[idx_b] + vg_max_gm_bwd

    peak_mu_fwd = mobility_fwd_raw[idx_f]
    peak_mu_bwd = mobility_bwd_raw[idx_b]
    
    hysteresis = abs(vth_fwd - vth_bwd)
    
    onoff_ratio = id_raw.abs().max() / id_raw.abs().min()
    ss_fwd = calculate_ss(id_fwd.values, vg_fwd.values)
    ss_bwd = calculate_ss(id_bwd.values, vg_bwd.values)
    
    return {
        'mu_fwd': peak_mu_fwd, 'vth_fwd': vth_fwd, 'gm_max_fwd': vg_max_gm_fwd, 'ss_fwd': ss_fwd,
        'mu_bwd': peak_mu_bwd, 'vth_bwd': vth_bwd, 'gm_max_bwd': vg_max_gm_bwd, 'ss_bwd': ss_bwd,
        'onoff': onoff_ratio, 'hysteresis': hysteresis,
        'vg_fwd': vg_fwd, 'id_fwd': id_fwd, 'gm_fwd_raw': gm_fwd_raw, 'mobility_fwd_raw': mobility_fwd_raw,
        'vg_bwd': vg_bwd, 'id_bwd': id_bwd, 'gm_bwd_raw': gm_bwd_raw, 'mobility_bwd_raw': mobility_bwd_raw,
        'vg_full': vg,
        'vd': vd 
    }

# 3. 파일 업로드
uploaded_files = st.file_uploader(
    "측정된 엑셀 파일을 업로드하세요",
    type=["xlsx", "xls"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 1:
        selected_file_name = st.sidebar.selectbox(
            "📁 Select Excel File",
            [f.name for f in uploaded_files]
        )
        uploaded_file = next(f for f in uploaded_files if f.name == selected_file_name)
    else:
        uploaded_file = uploaded_files[0]

    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    target_sheets = [s for s in sheet_names if s == 'Data' or s.lower().startswith('append')]
    
    if not target_sheets:
        st.error("분석할 수 있는 시트('Data' 또는 'Append...')가 없습니다.")
    else:
        # 최초 1회 세션 초기화 로직
        for s_name in target_sheets:
            key_f_init = f"val_fwd_{file_id}_{s_name}_{operating_mode}"
            key_b_init = f"val_bwd_{file_id}_{s_name}_{operating_mode}"
            
            if key_f_init not in st.session_state:
                temp_df = pd.read_excel(uploaded_file, sheet_name=s_name)
                temp_vg = temp_df['GateV']
                temp_id = temp_df['DrainI']
                if abs(temp_vg.max() - temp_vg.iloc[0]) > abs(temp_vg.min() - temp_vg.iloc[0]):
                    p_idx = temp_vg.idxmax()
                else: p_idx = temp_vg.idxmin()
                temp_fwd_vg, temp_fwd_id = temp_vg[:p_idx+1].reset_index(drop=True), temp_id[:p_idx+1].reset_index(drop=True)
                temp_bwd_vg, temp_bwd_id = temp_vg[p_idx:].reset_index(drop=True), temp_id[p_idx:].reset_index(drop=True)
                
                if operating_mode == "Linear":
                    gm_f_init = np.abs(fix_inf(np.gradient(temp_fwd_id.values, temp_fwd_vg.values)))
                    gm_b_init = np.abs(fix_inf(np.gradient(temp_bwd_id.values, temp_bwd_vg.values)))
                else:
                    gm_f_init = np.abs(fix_inf(np.gradient(np.sqrt(np.abs(temp_fwd_id.values)), temp_fwd_vg.values)))
                    gm_b_init = np.abs(fix_inf(np.gradient(np.sqrt(np.abs(temp_bwd_id.values)), temp_bwd_vg.values)))
                
                idx_f_init = np.argmax(gm_f_init[2:-2]) + 2 if len(gm_f_init) > 5 else np.argmax(gm_f_init)
                idx_b_init = np.argmax(gm_b_init[2:-2]) + 2 if len(gm_b_init) > 5 else np.argmax(gm_b_init)
                
                st.session_state[key_f_init] = float(temp_fwd_vg.iloc[idx_f_init])
                st.session_state[key_b_init] = float(temp_bwd_vg.iloc[idx_b_init])

        st.sidebar.markdown("---")
        options = target_sheets + ["Average (All Sheets)"]
        selected_sheet = st.sidebar.selectbox("📂 Select Data Sheet", options)
        
        # =====================================================================
        # [모드 1] Average (All Sheets) 선택 시 로직
        # =====================================================================
        if selected_sheet == "Average (All Sheets)":
            st.markdown(f"<h3 style='color: #333;'>📊 Statistics ({operating_mode} - Average of {len(target_sheets)} sheets)</h3>", unsafe_allow_html=True)
            st.info("해당 값은 각 시트에서 추출된(수정된 $V_{g}$ 포인트가 반영된) 파라미터의 평균(± 표준편차)입니다.")
            
            results = []
            for sheet in target_sheets:
                df = pd.read_excel(uploaded_file, sheet_name=sheet)
                if 'GateV' in df.columns and 'DrainI' in df.columns:
                    res = extract_parameters_from_sheet(df, file_id, sheet, W, L, Cox, operating_mode)
                    results.append(res)
                    
            if not results:
                st.error("유효한 데이터가 있는 시트가 없습니다.")
            else:
                df_res = pd.DataFrame(results)
                
                def format_stat(col, unit, is_log=False):
                    mean_val = df_res[col].mean()
                    std_val = df_res[col].std()
                    if is_log:
                        exp = int(np.floor(np.log10(mean_val)))
                        coef = mean_val / (10 ** exp)
                        return f"{coef:.2f}E{exp}" 
                    
                    if not np.isfinite(mean_val): return "N/A"
                    return f"{mean_val:.2f} ± {std_val:.2f} {unit}"
                
                st.markdown("<h4 style='color: #6FADCF;'>Forward Sweep Parameters (Avg)</h4>", unsafe_allow_html=True)
                f1, f2, f3, f4 = st.columns(4)
                f1.markdown(make_card(f"{operating_mode} Mobility (@ Peak)", format_stat('mu_fwd', 'cm²/V·s'), "#2E60AB"), unsafe_allow_html=True)
                f2.markdown(make_card("Threshold Voltage (Vₜₕ)", format_stat('vth_fwd', 'V'), "#A23B72"), unsafe_allow_html=True)
                f3.markdown(make_card("Peak Point (V<sub>g</sub>)", format_stat('gm_max_fwd', 'V'), "#F18F01"), unsafe_allow_html=True)
                f4.markdown(make_card("SS (Subthreshold Swing)", format_stat('ss_fwd', 'mV/dec'), "#18A558"), unsafe_allow_html=True)

                st.markdown("<h4 style='color: #F05650; margin-top: 20px;'>Backward Sweep Parameters (Avg)</h4>", unsafe_allow_html=True)
                b1, b2, b3, b4 = st.columns(4)
                b1.markdown(make_card(f"{operating_mode} Mobility (@ Peak)", format_stat('mu_bwd', 'cm²/V·s'), "#2E60AB"), unsafe_allow_html=True)
                b2.markdown(make_card("Threshold Voltage (Vₜₕ)", format_stat('vth_bwd', 'V'), "#A23B72"), unsafe_allow_html=True)
                b3.markdown(make_card("Peak Point (V<sub>g</sub>)", format_stat('gm_max_bwd', 'V'), "#F18F01"), unsafe_allow_html=True)
                b4.markdown(make_card("SS (Subthreshold Swing)", format_stat('ss_bwd', 'mV/dec'), "#18A558"), unsafe_allow_html=True)
                
                st.markdown("<h4 style='margin-top: 20px;'>Overall Device Parameters (Avg)</h4>", unsafe_allow_html=True)
                o1, o2, o3, o4 = st.columns(4) 
                o1.markdown(make_card("On/Off Ratio (Mean)", format_stat('onoff', '', is_log=True), "#5B5F97"), unsafe_allow_html=True)
                o2.markdown(make_card("Hysteresis", format_stat('hysteresis', 'V'), "#5B5F97"), unsafe_allow_html=True)
                st.markdown("---")

        # =====================================================================
        # [모드 2] 특정 단일 시트 선택 시 로직
        # =====================================================================
        else:
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            if 'GateV' not in df.columns or 'DrainI' not in df.columns:
                st.warning(f"'{selected_sheet}' 시트에 'GateV' 또는 'DrainI' 컬럼이 없어 분석할 수 없습니다.")
            else:
                # ✅ 함수 호출 결과 받기
                res = extract_parameters_from_sheet(df, file_id, selected_sheet, W, L, Cox, operating_mode)
                
                vg_fwd, id_fwd = res['vg_fwd'], res['id_fwd']
                vg_bwd, id_bwd = res['vg_bwd'], res['id_bwd']
                gm_fwd_raw, mobility_fwd_raw = res['gm_fwd_raw'], res['mobility_fwd_raw']
                gm_bwd_raw, mobility_bwd_raw = res['gm_bwd_raw'], res['mobility_bwd_raw']
                vg = res['vg_full']
                vd_val = res['vd'] 
                
                has_ig = 'GateI' in df.columns
                if has_ig:
                    ig_raw = df['GateI']
                    peak_idx = len(vg_fwd) - 1
                    ig_fwd, ig_bwd = ig_raw[:peak_idx+1].reset_index(drop=True), ig_raw[peak_idx:].reset_index(drop=True)

                st.sidebar.markdown("---")
                st.sidebar.markdown(f"**Peak Point Adjustment ({selected_sheet})**")
                vg_step = float(abs(vg_fwd.iloc[1] - vg_fwd.iloc[0])) if len(vg_fwd) > 1 else 0.5
                
                # 마스터 세션 키
                key_f_current = f"val_fwd_{file_id}_{selected_sheet}_{operating_mode}"
                key_b_current = f"val_bwd_{file_id}_{selected_sheet}_{operating_mode}"
                
                # 위젯 고유 키
                fwd_slider_key = f"fs_{file_id}_{selected_sheet}_{operating_mode}"
                fwd_number_key = f"fn_{file_id}_{selected_sheet}_{operating_mode}"
                bwd_slider_key = f"bs_{file_id}_{selected_sheet}_{operating_mode}"
                bwd_number_key = f"bn_{file_id}_{selected_sheet}_{operating_mode}"

                # 가장 확실한 연동 방식: 위젯이 그려지기 전에 세션 키를 서로 동기화
                if fwd_slider_key not in st.session_state:
                    st.session_state[fwd_slider_key] = st.session_state[key_f_current]
                if fwd_number_key not in st.session_state:
                    st.session_state[fwd_number_key] = st.session_state[key_f_current]
                if bwd_slider_key not in st.session_state:
                    st.session_state[bwd_slider_key] = st.session_state[key_b_current]
                if bwd_number_key not in st.session_state:
                    st.session_state[bwd_number_key] = st.session_state[key_b_current]

                # 콜백 함수: 하나가 바뀌면 다른 위젯 키와 마스터 키를 모두 업데이트
                def sync_fwd_from_slider():
                    val = st.session_state[fwd_slider_key]
                    st.session_state[fwd_number_key] = val
                    st.session_state[key_f_current] = val

                def sync_fwd_from_number():
                    val = st.session_state[fwd_number_key]
                    st.session_state[fwd_slider_key] = val
                    st.session_state[key_f_current] = val

                def sync_bwd_from_slider():
                    val = st.session_state[bwd_slider_key]
                    st.session_state[bwd_number_key] = val
                    st.session_state[key_b_current] = val

                def sync_bwd_from_number():
                    val = st.session_state[bwd_number_key]
                    st.session_state[bwd_slider_key] = val
                    st.session_state[key_b_current] = val

                # 🌟 Forward UI
                st.sidebar.markdown("<span style=' font-weight: bold;'>Forward V<sub>g</sub> Point</span>", unsafe_allow_html=True)
                fwd_min, fwd_max = float(vg_fwd.min()), float(vg_fwd.max())
                
                # 주의: value 인자를 제거하고 오직 key로만 제어
                st.sidebar.slider(
                    "Fwd $V_g$ Drag", 
                    min_value=fwd_min, max_value=fwd_max, 
                    step=vg_step, 
                    key=fwd_slider_key,
                    on_change=sync_fwd_from_slider,
                    label_visibility="collapsed"
                )
                
                st.sidebar.number_input(
                    "Fwd $V_g$ Button", 
                    min_value=fwd_min, max_value=fwd_max, 
                    step=vg_step, format="%.2f", 
                    key=fwd_number_key,
                    on_change=sync_fwd_from_number,
                    label_visibility="collapsed"
                )
                
                # 🌟 Backward UI
                st.sidebar.markdown("<br><span style=' font-weight: bold;'>Backward V<sub>g</sub> Point</span>", unsafe_allow_html=True)
                bwd_min, bwd_max = float(vg_bwd.min()), float(vg_bwd.max())
                
                st.sidebar.slider(
                    "Bwd $V_g$ Drag", 
                    min_value=bwd_min, max_value=bwd_max, 
                    step=vg_step, 
                    key=bwd_slider_key,
                    on_change=sync_bwd_from_slider,
                    label_visibility="collapsed"
                )
                
                st.sidebar.number_input(
                    "Bwd $V_g$ Button", 
                    min_value=bwd_min, max_value=bwd_max, 
                    step=vg_step, format="%.2f", 
                    key=bwd_number_key,
                    on_change=sync_bwd_from_number,
                    label_visibility="collapsed"
                )

                # UI 출력값 구성
                vg_max_gm_fwd = res['gm_max_fwd']
                vg_max_gm_bwd = res['gm_max_bwd']
                
                ss_fwd_display = f"{res['ss_fwd']:.1f} mV/dec" if np.isfinite(res['ss_fwd']) else "N/A"
                ss_bwd_display = f"{res['ss_bwd']:.1f} mV/dec" if np.isfinite(res['ss_bwd']) else "N/A"
                
                exponent = int(np.floor(np.log10(res['onoff'])))
                coefficient = res['onoff'] / (10 ** exponent)
                onoff_str = f"{coefficient:.2f}E{exponent}"
                
                st.markdown(f"<h3 style='color: #333;'>📊 Data Sheet: {selected_sheet} ({operating_mode} Mode)</h3>", unsafe_allow_html=True)
                
                st.markdown("<h4 style='color: #6FADCF;'>Forward Sweep Parameters</h4>", unsafe_allow_html=True)
                f1, f2, f3, f4 = st.columns(4)
                f1.markdown(make_card("Peak Mobility", f"{res['mu_fwd']:.2f} cm²/V·s", "#2E60AB"), unsafe_allow_html=True)
                f2.markdown(make_card("Threshold Voltage (Vₜₕ)", f"{res['vth_fwd']:.2f} V", "#A23B72"), unsafe_allow_html=True)
                f3.markdown(make_card("Peak Point (V<sub>g</sub>)", f"{vg_max_gm_fwd:.1f} V", "#F18F01"), unsafe_allow_html=True)
                f4.markdown(make_card("SS (Subthreshold Swing)", ss_fwd_display, "#18A558"), unsafe_allow_html=True)

                st.markdown("<h4 style='color: #F05650; margin-top: 20px;'>Backward Sweep Parameters</h4>", unsafe_allow_html=True)
                b1, b2, b3, b4 = st.columns(4)
                b1.markdown(make_card("Peak Mobility", f"{res['mu_bwd']:.2f} cm²/V·s", "#2E60AB"), unsafe_allow_html=True)
                b2.markdown(make_card("Threshold Voltage (Vₜₕ)", f"{res['vth_bwd']:.2f} V", "#A23B72"), unsafe_allow_html=True)
                b3.markdown(make_card("Peak Point (V<sub>g</sub>)", f"{vg_max_gm_bwd:.1f} V", "#F18F01"), unsafe_allow_html=True)
                b4.markdown(make_card("SS (Subthreshold Swing)", ss_bwd_display, "#18A558"), unsafe_allow_html=True)
                
                st.markdown("<h4 style='margin-top: 20px;'>Overall Device Parameters</h4>", unsafe_allow_html=True)
                o1, o2, o3, o4 = st.columns(4) 
                o1.markdown(make_card("On/Off Ratio", onoff_str, "#5B5F97"), unsafe_allow_html=True)
                o2.markdown(make_card("Hysteresis (Based on the Vₜₕ)", f"{res['hysteresis']:.2f} V", "#5B5F97"), unsafe_allow_html=True)
                st.markdown("---")

                # 그래프 생성 (모드에 따라 타이틀 분기)
                graph3_title = "3. Transconductance (Gₘ)" if operating_mode == "Linear" else "3. d(√I<sub>D</sub>)/dV<sub>G</sub>"
                # ✅ 4번 그래프 Y축 이름 분기
                graph4_title = "4. Linear Mobility" if operating_mode == "Linear" else "4. Saturation Mobility"

                fig = make_subplots(rows=2, cols=2, 
                                    subplot_titles=("1. Transfer (Log Scale)", "2. Transfer (Linear Scale)", 
                                                    graph3_title, graph4_title),
                                    horizontal_spacing=0.25, vertical_spacing=0.25)

                color_fwd, color_bwd = 'blue', 'red'
                color_fwd_smooth, color_bwd_smooth = '#6FADCF', '#F05650'
                dense_dash = '5px, 4px'

                fig.add_trace(go.Scatter(x=vg_fwd, y=id_fwd.abs(), name="Forward", line=dict(color=color_fwd), legend="legend"), row=1, col=1)
                fig.add_trace(go.Scatter(x=vg_bwd, y=id_bwd.abs(), name="Backward", line=dict(color=color_bwd), legend="legend"), row=1, col=1)
                if has_ig:
                    fig.add_trace(go.Scatter(x=vg_fwd, y=ig_fwd.abs(), name="I<sub>g</sub> (Fwd)", line=dict(color='dimgray', dash='dot'), showlegend=False), row=1, col=1)
                    fig.add_trace(go.Scatter(x=vg_bwd, y=ig_bwd.abs(), name="I<sub>g</sub> (Bwd)", line=dict(color='dimgray', dash='dot'), showlegend=False), row=1, col=1)
                    
                fig.add_trace(go.Scatter(x=vg_fwd, y=id_fwd.abs(), name="Forward", line=dict(color=color_fwd), legend="legend2"), row=1, col=2)
                fig.add_trace(go.Scatter(x=vg_bwd, y=id_bwd.abs(), name="Backward", line=dict(color=color_bwd), legend="legend2"), row=1, col=2)
                        
                fig.add_trace(go.Scatter(x=vg_fwd, y=abs(gm_fwd_raw), name="Forward", line=dict(color=color_fwd), legend="legend3"), row=2, col=1)
                fig.add_trace(go.Scatter(x=vg_bwd, y=abs(gm_bwd_raw), name="Backward", line=dict(color=color_bwd), legend="legend3"), row=2, col=1)
                
                # 시각화 
                fig.add_vline(x=vg_max_gm_fwd, line_width=1.5, line_dash=dense_dash, line_color=color_fwd_smooth, opacity=0.8, row=2, col=1)
                fig.add_vline(x=vg_max_gm_bwd, line_width=1.5, line_dash=dense_dash, line_color=color_bwd_smooth, opacity=0.8, row=2, col=1)
                        
                fig.add_trace(go.Scatter(x=vg_fwd, y=mobility_fwd_raw, name="Forward", line=dict(color=color_fwd), legend="legend4"), row=2, col=2)
                fig.add_trace(go.Scatter(x=vg_bwd, y=mobility_bwd_raw, name="Backward", line=dict(color=color_bwd), legend="legend4"), row=2, col=2)
                
                fig.add_vline(x=vg_max_gm_fwd, line_width=1.5, line_dash=dense_dash, line_color=color_fwd_smooth, opacity=0.8, row=2, col=2)
                fig.add_vline(x=vg_max_gm_bwd, line_width=1.5, line_dash=dense_dash, line_color=color_bwd_smooth, opacity=0.8, row=2, col=2)
                
                # ✅ 1번 그래프 좌하단에 DrainV 표시 추가 및 글씨 줄임 (유효숫자 처리)
                vd_formatted = f"{vd_val:.2f}".rstrip('0').rstrip('.') # 불필요한 0과 소수점 제거 (예: -0.1000 -> -0.1)
                fig.add_annotation(
                    x=0.001, y=0.001, xref="x domain", yref="y domain",
                    text=f"<b>V<sub>D</sub> = {vd_formatted} V</b>",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    row=1, col=1
                )

                # ✅ Legend 폰트 크기 증가
                leg_style = dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1, xanchor="right", yanchor="top", font=dict(color="black", size=14))

                # ✅ Subplot 타이틀 폰트 크기 증가 (Drain V 크기 인듯 ?)
                fig.update_annotations(font_size=16)

                fig.update_layout(width=1000, height=1000, autosize=False, template="plotly_white", margin=dict(t=120, b=80, l=100, r=100),
                                  legend=dict(x=0.375, y=1.0, **leg_style), legend2=dict(x=1.0, y=1.0, **leg_style),
                                  legend3=dict(x=0.375, y=0.375, **leg_style), legend4=dict(x=1.0, y=0.375, **leg_style))
                
                # ✅ 에러 방지 처리 (AttributeError: 'Annotation' object has no attribute 'get')
                # getattr 또는 hasattr을 사용하여 안전하게 접근
                for annotation in fig['layout']['annotations']:
                    ann_text = getattr(annotation, 'text', '')
                    if ann_text is not None and 'V<sub>D</sub>' not in str(ann_text):
                        annotation.font.color = 'black'
                        annotation.font.size = 22
                        annotation.yshift = 25
                
                # ✅ X축, Y축 라벨 폰트 크기 및 눈금 폰트 크기 증가
                common_axis_params = dict(
                    ticks="outside", tickwidth=1.5, tickcolor='black', ticklen=8, 
                    showline=True, linewidth=1.5, linecolor='black', mirror=True, 
                    showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='dot', 
                    zeroline=False, layer='below traces',
                    title_font=dict(size=22),
                    tickfont=dict(size=15)
                )

                # ✅ NameError 해결 (원본 데이터 추출 함수에서 넘겨받은 vg_full 활용)
                vg_range = abs(vg.max() - vg.min())
                dynamic_dtick = 2.5 if vg_range <= 10 else 10

                y_title_3 = "Gₘ (S)" if operating_mode == "Linear" else "d(√I<sub>D</sub>)/dV<sub>G</sub> (A<sup>0.5</sup>/V)"
                # ✅ Y축 타이틀 분기
                y_title_4 = "Linear Mobility (cm²/V·s)" if operating_mode == "Linear" else "Saturation Mobility (cm²/V·s)"

                fig.update_xaxes(title_text="Gate Voltage (V)", dtick=dynamic_dtick, **common_axis_params)
                fig.update_yaxes(**common_axis_params)
                fig.update_yaxes(title_text="Drain Current (A)", type="log", dtick=1, exponentformat="power", row=1, col=1)
                fig.update_yaxes(title_text="Drain Current (A)", exponentformat="power", row=1, col=2)
                fig.update_yaxes(title_text=y_title_3, exponentformat="power", row=2, col=1)
                fig.update_yaxes(title_text=y_title_4, row=2, col=2)

                st.plotly_chart(fig, use_container_width=False)