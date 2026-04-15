import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.signal import savgol_filter, medfilt  

# 1. 페이지 설정
st.set_page_config(page_title="FET-Analysis_Minjae", layout="wide")
st.title("FET-Analysis_Minjae")

# 2. 소자 파라미터 
st.sidebar.header("Device Information")
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
        <p style='font-size: 18px; margin-bottom: 5px; color: #555;'>{title}</p>
        <p style='font-size: 32px; font-weight: bold; color: {color}; margin: 0; line-height: 1.2;'>{value}</p>
    </div>
    """

# ✅ 파라미터 계산을 묶어둔 헬퍼 함수 (Average 계산용)
def extract_parameters_from_sheet(df, w, l, cox):
    vg = df['GateV']
    id_raw = df['DrainI']
    vd = df['DrainV'].iloc[0]
    
    if abs(vg.max() - vg.iloc[0]) > abs(vg.min() - vg.iloc[0]):
        peak_idx = vg.idxmax()
    else:
        peak_idx = vg.idxmin()
        
    vg_fwd, id_fwd = vg[:peak_idx+1].reset_index(drop=True), id_raw[:peak_idx+1].reset_index(drop=True)
    vg_bwd, id_bwd = vg[peak_idx:].reset_index(drop=True), id_raw[peak_idx:].reset_index(drop=True)
    
    id_fwd_med = medfilt(id_fwd, kernel_size=3) 
    id_bwd_med = medfilt(id_bwd, kernel_size=3)

    win_len = min(len(id_fwd), 15)
    if win_len % 2 == 0: win_len -= 1
    if win_len >= 3:
        id_fwd_smooth = savgol_filter(id_fwd_med, win_len, 2)
        id_bwd_smooth = savgol_filter(id_bwd_med, win_len, 2)
    else:
        id_fwd_smooth, id_bwd_smooth = id_fwd_med, id_bwd_med

    gm_fwd_raw = fix_inf(np.gradient(id_fwd.values, vg_fwd.values))
    mobility_fwd_raw = (abs(gm_fwd_raw) * l) / (w * cox * abs(vd))
    
    gm_bwd_raw = fix_inf(np.gradient(id_bwd.values, vg_bwd.values))
    mobility_bwd_raw = (abs(gm_bwd_raw) * l) / (w * cox * abs(vd))

    gm_fwd_smooth = fix_inf(np.gradient(id_fwd_smooth, vg_fwd.values))
    gm_bwd_smooth = fix_inf(np.gradient(id_bwd_smooth, vg_bwd.values))

    idx_max_gm_fwd = np.argmax(np.abs(gm_fwd_smooth))
    idx_max_gm_bwd = np.argmax(np.abs(gm_bwd_smooth))
    
    vg_max_gm_fwd = float(vg_fwd.iloc[idx_max_gm_fwd])
    vg_max_gm_bwd = float(vg_bwd.iloc[idx_max_gm_bwd])

    vth_fwd = -id_fwd.iloc[idx_max_gm_fwd] / gm_fwd_raw[idx_max_gm_fwd] + vg_max_gm_fwd
    peak_mu_fwd = mobility_fwd_raw[idx_max_gm_fwd]
    
    vth_bwd = -id_bwd.iloc[idx_max_gm_bwd] / gm_bwd_raw[idx_max_gm_bwd] + vg_max_gm_bwd
    peak_mu_bwd = mobility_bwd_raw[idx_max_gm_bwd]
    
    hysteresis = abs(vth_fwd - vth_bwd)
    
    onoff_ratio = id_raw.abs().max() / id_raw.abs().min()
    ss_fwd = calculate_ss(id_fwd.values, vg_fwd.values)
    ss_bwd = calculate_ss(id_bwd.values, vg_bwd.values)
    
    return {
        'mu_fwd': peak_mu_fwd, 'vth_fwd': vth_fwd, 'gm_max_fwd': vg_max_gm_fwd, 'ss_fwd': ss_fwd,
        'mu_bwd': peak_mu_bwd, 'vth_bwd': vth_bwd, 'gm_max_bwd': vg_max_gm_bwd, 'ss_bwd': ss_bwd,
        'onoff': onoff_ratio, 'hysteresis': hysteresis
    }

# 3. 파일 업로드
uploaded_file = st.file_uploader("측정된 엑셀 파일을 업로드하세요", type=["xlsx", "xls"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    target_sheets = [s for s in sheet_names if s == 'Data' or s.lower().startswith('append')]
    
    if not target_sheets:
        st.error("분석할 수 있는 시트('Data' 또는 'Append...')가 없습니다.")
    else:
        st.sidebar.markdown("---")
        # ✅ Average (All Sheets) 옵션 추가
        options = ["Average (All Sheets)"] + target_sheets
        selected_sheet = st.sidebar.selectbox("📂 Select Data Sheet", options)
        
        # =====================================================================
        # [모드 1] Average (All Sheets) 선택 시 로직
        # =====================================================================
        if selected_sheet == "Average (All Sheets)":
            st.markdown(f"<h3 style='color: #333;'>📊 Statistics (Average of {len(target_sheets)} sheets)</h3>", unsafe_allow_html=True)
            st.info("해당 값은 각 시트에서 추출된 파라미터의 평균(± 표준편차)입니다. (그래프는 표시되지 않습니다.)")
            
            results = []
            for sheet in target_sheets:
                df = pd.read_excel(uploaded_file, sheet_name=sheet)
                if 'GateV' in df.columns and 'DrainI' in df.columns:
                    res = extract_parameters_from_sheet(df, W, L, Cox)
                    results.append(res)
                    
            if not results:
                st.error("유효한 데이터가 있는 시트가 없습니다.")
            else:
                df_res = pd.DataFrame(results)
                
                # 통계량 포맷팅 도우미
                def format_stat(col, unit, is_log=False):
                    mean_val = df_res[col].mean()
                    std_val = df_res[col].std()
                    if is_log:
                        exp = int(np.floor(np.log10(mean_val)))
                        coef = mean_val / (10 ** exp)
                        return f"{coef:.2f}E{exp}" # On/Off는 평균만 E표기법으로
                    
                    if not np.isfinite(mean_val): return "N/A"
                    return f"{mean_val:.2f} ± {std_val:.2f} {unit}"
                
                st.markdown("<h4 style='color: #6FADCF;'>Forward Sweep Parameters (Avg)</h4>", unsafe_allow_html=True)
                f1, f2, f3, f4 = st.columns(4)
                f1.markdown(make_card("Peak Mobility", format_stat('mu_fwd', 'cm²/V·s'), "#2E60AB"), unsafe_allow_html=True)
                f2.markdown(make_card("Threshold Voltage (Vₜₕ)", format_stat('vth_fwd', 'V'), "#A23B72"), unsafe_allow_html=True)
                f3.markdown(make_card("Gₘ Max Point", format_stat('gm_max_fwd', 'V'), "#F18F01"), unsafe_allow_html=True)
                f4.markdown(make_card("SS (Subthreshold Swing)", format_stat('ss_fwd', 'mV/dec'), "#18A558"), unsafe_allow_html=True)

                st.markdown("<h4 style='color: #F05650; margin-top: 20px;'>Backward Sweep Parameters (Avg)</h4>", unsafe_allow_html=True)
                b1, b2, b3, b4 = st.columns(4)
                b1.markdown(make_card("Peak Mobility", format_stat('mu_bwd', 'cm²/V·s'), "#2E60AB"), unsafe_allow_html=True)
                b2.markdown(make_card("Threshold Voltage (Vₜₕ)", format_stat('vth_bwd', 'V'), "#A23B72"), unsafe_allow_html=True)
                b3.markdown(make_card("Gₘ Max Point", format_stat('gm_max_bwd', 'V'), "#F18F01"), unsafe_allow_html=True)
                b4.markdown(make_card("SS (Subthreshold Swing)", format_stat('ss_bwd', 'mV/dec'), "#18A558"), unsafe_allow_html=True)
                
                st.markdown("<h4 style='margin-top: 20px;'>Overall Device Parameters (Avg)</h4>", unsafe_allow_html=True)
                o1, o2, o3, o4 = st.columns(4) 
                o1.markdown(make_card("On/Off Ratio (Mean)", format_stat('onoff', '', is_log=True), "#5B5F97"), unsafe_allow_html=True)
                o2.markdown(make_card("Hysteresis", format_stat('hysteresis', 'V'), "#5B5F97"), unsafe_allow_html=True)
                st.markdown("---")

        # =====================================================================
        # [모드 2] 특정 단일 시트 선택 시 로직 (기존 로직 완벽 유지)
        # =====================================================================
        else:
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            if 'GateV' not in df.columns or 'DrainI' not in df.columns:
                st.warning(f"'{selected_sheet}' 시트에 'GateV' 또는 'DrainI' 컬럼이 없어 분석할 수 없습니다.")
            else:
                vg = df['GateV']
                id_raw = df['DrainI']
                vd = df['DrainV'].iloc[0]
                
                has_ig = 'GateI' in df.columns
                if has_ig:
                    ig_raw = df['GateI']

                if abs(vg.max() - vg.iloc[0]) > abs(vg.min() - vg.iloc[0]):
                    peak_idx = vg.idxmax()
                else:
                    peak_idx = vg.idxmin()
                    
                vg_fwd, id_fwd = vg[:peak_idx+1].reset_index(drop=True), id_raw[:peak_idx+1].reset_index(drop=True)
                vg_bwd, id_bwd = vg[peak_idx:].reset_index(drop=True), id_raw[peak_idx:].reset_index(drop=True)
                
                if has_ig:
                    ig_fwd, ig_bwd = ig_raw[:peak_idx+1].reset_index(drop=True), ig_raw[peak_idx:].reset_index(drop=True)

                id_fwd_med = medfilt(id_fwd, kernel_size=3) 
                id_bwd_med = medfilt(id_bwd, kernel_size=3)

                win_len = min(len(id_fwd), 15)
                if win_len % 2 == 0: win_len -= 1
                
                if win_len >= 3:
                    id_fwd_smooth = savgol_filter(id_fwd_med, win_len, 2)
                    id_bwd_smooth = savgol_filter(id_bwd_med, win_len, 2)
                else:
                    id_fwd_smooth, id_bwd_smooth = id_fwd_med, id_bwd_med

                gm_fwd_raw = fix_inf(np.gradient(id_fwd.values, vg_fwd.values))
                mobility_fwd_raw = (abs(gm_fwd_raw) * L) / (W * Cox * abs(vd))
                
                gm_bwd_raw = fix_inf(np.gradient(id_bwd.values, vg_bwd.values))
                mobility_bwd_raw = (abs(gm_bwd_raw) * L) / (W * Cox * abs(vd))

                gm_fwd_smooth = fix_inf(np.gradient(id_fwd_smooth, vg_fwd.values))
                mobility_fwd_smooth = (abs(gm_fwd_smooth) * L) / (W * Cox * abs(vd))
                
                gm_bwd_smooth = fix_inf(np.gradient(id_bwd_smooth, vg_bwd.values))
                mobility_bwd_smooth = (abs(gm_bwd_smooth) * L) / (W * Cox * abs(vd))

                idx_max_gm_fwd_auto = np.argmax(np.abs(gm_fwd_smooth))
                vg_max_gm_fwd_auto = float(vg_fwd.iloc[idx_max_gm_fwd_auto])
                
                idx_max_gm_bwd_auto = np.argmax(np.abs(gm_bwd_smooth))
                vg_max_gm_bwd_auto = float(vg_bwd.iloc[idx_max_gm_bwd_auto])

                st.sidebar.markdown("---")
                st.sidebar.markdown(f"**Gₘ Max Point Adjustment ({selected_sheet})**")
                vg_step = float(abs(vg_fwd.iloc[1] - vg_fwd.iloc[0])) if len(vg_fwd) > 1 else 0.5
                
                st.sidebar.markdown("Adjust <span style='color: #6FADCF; font-weight: bold;'>Forward</span> $V_g$ Point", unsafe_allow_html=True)
                target_vg_fwd = st.sidebar.number_input("Adjust Fwd Vg", value=vg_max_gm_fwd_auto, step=vg_step, format="%.2f", label_visibility="collapsed")
                
                st.sidebar.markdown("Adjust <span style='color: #F05650; font-weight: bold;'>Backward</span> $V_g$ Point", unsafe_allow_html=True)
                target_vg_bwd = st.sidebar.number_input("Adjust Bwd Vg", value=vg_max_gm_bwd_auto, step=vg_step, format="%.2f", label_visibility="collapsed")

                selected_idx_fwd = (vg_fwd - target_vg_fwd).abs().argmin()
                vg_max_gm_fwd = vg_fwd.iloc[selected_idx_fwd]
                
                selected_idx_bwd = (vg_bwd - target_vg_bwd).abs().argmin()
                vg_max_gm_bwd = vg_bwd.iloc[selected_idx_bwd]

                vth_fwd = -id_fwd.iloc[selected_idx_fwd] / gm_fwd_raw[selected_idx_fwd] + vg_max_gm_fwd
                peak_mu_fwd = mobility_fwd_raw[selected_idx_fwd]
                
                vth_bwd = -id_bwd.iloc[selected_idx_bwd] / gm_bwd_raw[selected_idx_bwd] + vg_max_gm_bwd
                peak_mu_bwd = mobility_bwd_raw[selected_idx_bwd]
                
                hysteresis = abs(vth_fwd - vth_bwd)
                
                onoff_ratio = id_raw.abs().max() / id_raw.abs().min()
                exponent = int(np.floor(np.log10(onoff_ratio)))
                coefficient = onoff_ratio / (10 ** exponent)
                onoff_str = f"{coefficient:.2f}E{exponent}"
                
                ss_fwd = calculate_ss(id_fwd.values, vg_fwd.values)
                ss_bwd = calculate_ss(id_bwd.values, vg_bwd.values)
                
                ss_fwd_display = f"{ss_fwd:.1f} mV/dec" if np.isfinite(ss_fwd) else "N/A"
                ss_bwd_display = f"{ss_bwd:.1f} mV/dec" if np.isfinite(ss_bwd) else "N/A"
                
                st.markdown(f"<h3 style='color: #333;'>📊 Data Sheet: {selected_sheet}</h3>", unsafe_allow_html=True)
                
                st.markdown("<h4 style='color: #6FADCF;'>Forward Sweep Parameters</h4>", unsafe_allow_html=True)
                f1, f2, f3, f4 = st.columns(4)
                f1.markdown(make_card("Peak Mobility", f"{peak_mu_fwd:.2f} cm²/V·s", "#2E60AB"), unsafe_allow_html=True)
                f2.markdown(make_card("Threshold Voltage (Vₜₕ)", f"{vth_fwd:.2f} V", "#A23B72"), unsafe_allow_html=True)
                f3.markdown(make_card("Gₘ Max Point", f"{vg_fwd.iloc[selected_idx_fwd]:.1f} V", "#F18F01"), unsafe_allow_html=True)
                f4.markdown(make_card("SS (Subthreshold Swing)", ss_fwd_display, "#18A558"), unsafe_allow_html=True)

                st.markdown("<h4 style='color: #F05650; margin-top: 20px;'>Backward Sweep Parameters</h4>", unsafe_allow_html=True)
                b1, b2, b3, b4 = st.columns(4)
                b1.markdown(make_card("Peak Mobility", f"{peak_mu_bwd:.2f} cm²/V·s", "#2E60AB"), unsafe_allow_html=True)
                b2.markdown(make_card("Threshold Voltage (Vₜₕ)", f"{vth_bwd:.2f} V", "#A23B72"), unsafe_allow_html=True)
                b3.markdown(make_card("Gₘ Max Point", f"{vg_bwd.iloc[selected_idx_bwd]:.1f} V", "#F18F01"), unsafe_allow_html=True)
                b4.markdown(make_card("SS (Subthreshold Swing)", ss_bwd_display, "#18A558"), unsafe_allow_html=True)
                
                st.markdown("<h4 style='margin-top: 20px;'>Overall Device Parameters</h4>", unsafe_allow_html=True)
                o1, o2, o3, o4 = st.columns(4) 
                o1.markdown(make_card("On/Off Ratio", onoff_str, "#5B5F97"), unsafe_allow_html=True)
                o2.markdown(make_card("Hysteresis (Based on the Vₜₕ)", f"{hysteresis:.2f} V", "#5B5F97"), unsafe_allow_html=True)
                st.markdown("---")

                # 그래프 생성 
                fig = make_subplots(rows=2, cols=2, 
                                    subplot_titles=("1. Transfer (Log Scale)", "2. Transfer (Linear Scale)", 
                                                    "3. Transconductance (Gₘ)", "4. Field-Effect Mobility"),
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
                fig.add_trace(go.Scatter(x=vg_fwd, y=abs(gm_fwd_smooth), name="Fwd (Smoothed)", line=dict(color=color_fwd_smooth, dash=dense_dash, width=2), legend="legend3"), row=2, col=1)
                fig.add_trace(go.Scatter(x=vg_bwd, y=abs(gm_bwd_smooth), name="Bwd (Smoothed)", line=dict(color=color_bwd_smooth, dash=dense_dash, width=2), legend="legend3"), row=2, col=1)
                fig.add_vline(x=vg_max_gm_fwd, line_width=1.5, line_dash=dense_dash, line_color=color_fwd_smooth, opacity=0.8, row=2, col=1)
                fig.add_vline(x=vg_max_gm_bwd, line_width=1.5, line_dash=dense_dash, line_color=color_bwd_smooth, opacity=0.8, row=2, col=1)
                        
                fig.add_trace(go.Scatter(x=vg_fwd, y=mobility_fwd_raw, name="Forward", line=dict(color=color_fwd), legend="legend4"), row=2, col=2)
                fig.add_trace(go.Scatter(x=vg_bwd, y=mobility_bwd_raw, name="Backward", line=dict(color=color_bwd), legend="legend4"), row=2, col=2)
                fig.add_trace(go.Scatter(x=vg_fwd, y=mobility_fwd_smooth, name="Fwd (Smoothed)", line=dict(color=color_fwd_smooth, dash=dense_dash, width=2), legend="legend4"), row=2, col=2)
                fig.add_trace(go.Scatter(x=vg_bwd, y=mobility_bwd_smooth, name="Bwd (Smoothed)", line=dict(color=color_bwd_smooth, dash=dense_dash, width=2), legend="legend4"), row=2, col=2)
                fig.add_vline(x=vg_max_gm_fwd, line_width=1.5, line_dash=dense_dash, line_color=color_fwd_smooth, opacity=0.8, row=2, col=2)
                fig.add_vline(x=vg_max_gm_bwd, line_width=1.5, line_dash=dense_dash, line_color=color_bwd_smooth, opacity=0.8, row=2, col=2)

                leg_style = dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1, xanchor="right", yanchor="top", font=dict(color="black"))

                fig.update_layout(width=1000, height=1000, autosize=False, template="plotly_white", margin=dict(t=120, b=80, l=100, r=100),
                                  legend=dict(x=0.375, y=1.0, **leg_style), legend2=dict(x=1.0, y=1.0, **leg_style),
                                  legend3=dict(x=0.375, y=0.375, **leg_style), legend4=dict(x=1.0, y=0.375, **leg_style))
                
                for annotation in fig['layout']['annotations']:
                    annotation['font'] = dict(color='black', size=16)
                    annotation['yshift'] = 25
                
                common_axis_params = dict(ticks="outside", tickwidth=1.5, tickcolor='black', ticklen=8, showline=True, linewidth=1.5, linecolor='black', mirror=True, showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='dot', zeroline=False, layer='below traces')

                vg_range = abs(vg.max() - vg.min())
                dynamic_dtick = 2.5 if vg_range <= 10 else 10

                fig.update_xaxes(title_text="Gate Voltage (V)", dtick=dynamic_dtick, **common_axis_params)
                fig.update_yaxes(**common_axis_params)
                fig.update_yaxes(title_text="Drain Current (A)", type="log", dtick=1, exponentformat="power", row=1, col=1)
                fig.update_yaxes(title_text="Drain Current (A)", exponentformat="power", row=1, col=2)
                fig.update_yaxes(title_text="Gₘ (S)", exponentformat="power", row=2, col=1)
                fig.update_yaxes(title_text="Mobility (cm²/V·s)", row=2, col=2)

                st.plotly_chart(fig, use_container_width=False)