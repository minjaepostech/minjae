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
</style>
""", unsafe_allow_html=True)

# 2. 소자 파라미터 
st.sidebar.header("Device Information")

# 🌟 Operating Mode 선택 기능 추가 (사이드바 최상단)
operating_mode = st.sidebar.radio("Operating Mode", ["Linear", "Saturation"])
st.sidebar.markdown("---")

W = st.sidebar.number_input("Width (μm)", value=1000, step=50) 
L = st.sidebar.number_input("Length (μm)", value=100, step=50)
Cox_nf = st.sidebar.number_input("Capacitance (nF/cm⁻²)", value=34.5) 
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
uploaded_file = st.file_uploader("측정된 엑셀 파일을 업로드하세요", type=["xlsx", "xls"])

if uploaded_file:
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
            st.info("해당 값은 각 시트에서 추출된(수정된 Vg 포인트가 반영된) 파라미터의 평균(± 표준편차)입니다.")
            
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
                f3.markdown(make_card("Peak Point (Vg)", format_stat('gm_max_fwd', 'V'), "#F18F01"), unsafe_allow_html=True)
                f4.markdown(make_card("SS (Subthreshold Swing)", format_stat('ss_fwd', 'mV/dec'), "#18A558"), unsafe_allow_html=True)

                st.markdown("<h4 style='color: #F05650; margin-top: 20px;'>Backward Sweep Parameters (Avg)</h4>", unsafe_allow_html=True)
                b1, b2, b3, b4 = st.columns(4)
                b1.markdown(make_card(f"{operating_mode} Mobility (@ Peak)", format_stat('mu_bwd', 'cm²/V·s'), "#2E60AB"), unsafe_allow_html=True)
                b2.markdown(make_card("Threshold Voltage (Vₜₕ)", format_stat('vth_bwd', 'V'), "#A23B72"), unsafe_allow_html=True)
                b3.markdown(make_card("Peak Point (Vg)", format_stat('gm_max_bwd', 'V'), "#F18F01"), unsafe_allow_html=True)
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
                st.sidebar.markdown("<span style=' font-weight: bold;'>Forward $V_g$ Point</span>", unsafe_allow_html=True)
                fwd_min, fwd_max = float(vg_fwd.min()), float(vg_fwd.max())
                
                # 주의: value 인자를 제거하고 오직 key로만 제어
                st.sidebar.slider(
                    "Fwd Vg Drag", 
                    min_value=fwd_min, max_value=fwd_max, 
                    step=vg_step, 
                    key=fwd_slider_key,
                    on_change=sync_fwd_from_slider,
                    label_visibility="collapsed"
                )
                
                st.sidebar.number_input(
                    "Fwd Vg Button", 
                    min_value=fwd_min, max_value=fwd_max, 
                    step=vg_step, format="%.2f", 
                    key=fwd_number_key,
                    on_change=sync_fwd_from_number,
                    label_visibility="collapsed"
                )
                
                # 🌟 Backward UI
                st.sidebar.markdown("<br><span style=' font-weight: bold;'>Backward $V_g$ Point</span>", unsafe_allow_html=True)
                bwd_min, bwd_max = float(vg_bwd.min()), float(vg_bwd.max())
                
                st.sidebar.slider(
                    "Bwd Vg Drag", 
                    min_value=bwd_min, max_value=bwd_max, 
                    step=vg_step, 
                    key=bwd_slider_key,
                    on_change=sync_bwd_from_slider,
                    label_visibility="collapsed"
                )
                
                st.sidebar.number_input(
                    "Bwd Vg Button", 
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
                f3.markdown(make_card("Peak Point (Vg)", f"{vg_max_gm_fwd:.1f} V", "#F18F01"), unsafe_allow_html=True)
                f4.markdown(make_card("SS (Subthreshold Swing)", ss_fwd_display, "#18A558"), unsafe_allow_html=True)

                st.markdown("<h4 style='color: #F05650; margin-top: 20px;'>Backward Sweep Parameters</h4>", unsafe_allow_html=True)
                b1, b2, b3, b4 = st.columns(4)
                b1.markdown(make_card("Peak Mobility", f"{res['mu_bwd']:.2f} cm²/V·s", "#2E60AB"), unsafe_allow_html=True)
                b2.markdown(make_card("Threshold Voltage (Vₜₕ)", f"{res['vth_bwd']:.2f} V", "#A23B72"), unsafe_allow_html=True)
                b3.markdown(make_card("Peak Point (Vg)", f"{vg_max_gm_bwd:.1f} V", "#F18F01"), unsafe_allow_html=True)
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
                    fig.add_trace(go.Scatter(x=vg_fwd, y=ig_fwd.abs(), name="Ig (Fwd)", line=dict(color='dimgray', dash='dot'), showlegend=False), row=1, col=1)
                    fig.add_trace(go.Scatter(x=vg_bwd, y=ig_bwd.abs(), name="Ig (Bwd)", line=dict(color='dimgray', dash='dot'), showlegend=False), row=1, col=1)
                    
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