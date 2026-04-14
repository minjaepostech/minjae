import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 1. 페이지 설정 (타이틀 FAMJ 유지)
st.set_page_config(page_title="FET-Analysis_Minjae", layout="wide")
st.title("FET-Analysis_Minjae")

# 2. 소자 파라미터 
st.sidebar.header("Device Information")
W = st.sidebar.number_input("Width (μm)", value=1000.0) 
L = st.sidebar.number_input("Length (μm)", value=100.0)
Cox_nf = st.sidebar.number_input("Capacitance (nF/cm⁻²)", value=34.5) 
Cox = Cox_nf * 1e-9

# 3. 파일 업로드
uploaded_file = st.file_uploader("측정된 엑셀 파일을 업로드하세요", type=["xlsx", "xls"])

if uploaded_file:
    # 엑셀의 'Data' 시트 읽기
    df = pd.read_excel(uploaded_file, sheet_name='Data')
    vg = df['GateV']
    id_raw = df['DrainI']
    vd = df['DrainV'].iloc[0]
    
    has_ig = 'GateI' in df.columns
    if has_ig:
        ig_raw = df['GateI']

    # ✅ [핵심 수정 부분] Forward / Backward 스윕 분리
    # 시작점을 기준으로 전압이 양의 방향으로 크게 스윕했는지, 음의 방향으로 크게 스윕했는지 판단
    if abs(vg.max() - vg.iloc[0]) > abs(vg.min() - vg.iloc[0]):
        peak_idx = vg.idxmax()
    else:
        peak_idx = vg.idxmin()
        
    # ✅ 수정: .values를 추가하여 NumPy 배열로 변환 + reset_index()로 인덱스 리셋
    vg_fwd = vg[:peak_idx+1].reset_index(drop=True)
    id_fwd = id_raw[:peak_idx+1].reset_index(drop=True)
    vg_bwd = vg[peak_idx:].reset_index(drop=True)
    id_bwd = id_raw[peak_idx:].reset_index(drop=True)
    
    if has_ig:
        ig_fwd = ig_raw[:peak_idx+1].reset_index(drop=True)
        ig_bwd = ig_raw[peak_idx:].reset_index(drop=True)

    # 4. 물리량 계산 
    gm_fwd = np.gradient(id_fwd, vg_fwd)
    mobility_fwd = (abs(gm_fwd) * L) / (W * Cox * abs(vd))
    gm_bwd = np.gradient(id_bwd, vg_bwd)
    mobility_bwd = (abs(gm_bwd) * L) / (W * Cox * abs(vd))
    
    # 5. 파라미터 추출
    idx_max_gm = np.argmax(np.abs(gm_fwd))
    vg_max_gm = vg_fwd.iloc[idx_max_gm]  
    vth = -id_fwd.iloc[idx_max_gm] / gm_fwd[idx_max_gm] + vg_max_gm
    peak_mu = mobility_fwd[idx_max_gm]
    
    # On/Off Ratio 계산 및 E 표기법으로 변환
    onoff_ratio = id_raw.abs().max() / id_raw.abs().min()
    exponent = int(np.floor(np.log10(onoff_ratio)))
    coefficient = onoff_ratio / (10 ** exponent)
    onoff_str = f"{coefficient:.2f}E{exponent}"
    
    # SS (Subthreshold Swing) 계산 - (이동평균 적용하여 노이즈에 강하게 수정)
    vg_fwd_arr = vg_fwd.values
    id_fwd_arr = id_fwd.values
    
    # 로그 스케일 변환
    log_id_fwd = np.log10(np.abs(id_fwd_arr) + 1e-15)
    
    # Vg에 따른 log(Id)의 기울기 계산 (단위: decade/V)
    d_log_id = np.abs(np.gradient(log_id_fwd, vg_fwd_arr))
    
    # 실제 측정 데이터는 튀는 구간(노이즈)이 있으므로, 주변 3개 데이터 이동평균 적용
    d_log_id_smooth = np.convolve(d_log_id, np.ones(3)/3, mode='same')
    
    # 유효한 기울기 중 가장 가파른(최댓값) 지점 추출
    valid_slopes = d_log_id_smooth[np.isfinite(d_log_id_smooth) & (d_log_id_smooth > 0)]
    
    if len(valid_slopes) > 0:
        max_slope = np.max(valid_slopes)  # 가장 가파른 기울기 (decade / V)
        ss_value = 1.0 / max_slope        # 역수를 취함 (V / decade)
        ss_mv = ss_value * 1000           # mV / decade 단위 변환
    else:
        ss_mv = np.inf
    
    # 파라미터 표시 - 5개 컬럼
    c1, c2, c3, c4, c5 = st.columns(5)
    
    ss_display = f"{ss_mv:.1f} mV/dec" if np.isfinite(ss_mv) else "N/A"
    
    c1.markdown(f"""
    <div style='text-align: left;'>
        <p style='font-size: 18px; margin-bottom: 8px; height: 24px; line-height: 24px;'>Peak Mobility (Forward)</p>
        <p style='font-size: 28px; font-weight: bold; color: #2E86AB; margin: 0; height: 36px; line-height: 36px;'>{peak_mu:.2f} cm²/V·s</p>
    </div>
    """, unsafe_allow_html=True)
    
    c2.markdown(f"""
    <div style='text-align: left;'>
        <p style='font-size: 18px; margin-bottom: 8px; height: 24px; line-height: 24px;'>Vₜₕ (Linear)</p>
        <p style='font-size: 28px; font-weight: bold; color: #A23B72; margin: 0; height: 36px; line-height: 36px;'>{vth:.2f} V</p>
    </div>
    """, unsafe_allow_html=True)
    
    c3.markdown(f"""
    <div style='text-align: left;'>
        <p style='font-size: 18px; margin-bottom: 8px; height: 24px; line-height: 24px;'>Gₘ Max Point</p>
        <p style='font-size: 28px; font-weight: bold; color: #F18F01; margin: 0; height: 36px; line-height: 36px;'>{vg_max_gm:.1f} V</p>
    </div>
    """, unsafe_allow_html=True)
    
    c4.markdown(f"""
    <div style='text-align: left;'>
        <p style='font-size: 18px; margin-bottom: 8px; height: 24px; line-height: 24px;'>On/Off Ratio</p>
        <p style='font-size: 28px; font-weight: bold; color: #C73E1D; margin: 0; height: 36px; line-height: 36px;'>{onoff_str}</p>
    </div>
    """, unsafe_allow_html=True)
    
    c5.markdown(f"""
    <div style='text-align: left;'>
        <p style='font-size: 18px; margin-bottom: 8px; height: 24px; line-height: 24px;'>SS (Subthreshold Swing)</p>
        <p style='font-size: 28px; font-weight: bold; color: #18A558; margin: 0; height: 36px; line-height: 36px;'>{ss_display}</p>
    </div>
    """, unsafe_allow_html=True)

    # 6. 그래프 생성
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=("1. Transfer (Log Scale)", "2. Transfer (Linear Scale)", 
                                        "3. Transconductance (Gₘ)", "4. Field-Effect Mobility"),
                        horizontal_spacing=0.25, vertical_spacing=0.25)

    color_fwd = 'blue'
    color_bwd = 'red'

    # [1-1] Log Scale
    fig.add_trace(go.Scatter(x=vg_fwd, y=id_fwd.abs(), name="Forward", line=dict(color=color_fwd), legend="legend"), row=1, col=1)
    fig.add_trace(go.Scatter(x=vg_bwd, y=id_bwd.abs(), name="Backward", line=dict(color=color_bwd), legend="legend"), row=1, col=1)
    if has_ig:
        fig.add_trace(go.Scatter(x=vg_fwd, y=ig_fwd.abs(), name="Ig", line=dict(color='gray', dash='dot'), showlegend=False), row=1, col=1)
        
    # [1-2] Linear Scale - 절댓값 적용
    fig.add_trace(go.Scatter(x=vg_fwd, y=id_fwd.abs(), name="Forward", line=dict(color=color_fwd), legend="legend2"), row=1, col=2)
    fig.add_trace(go.Scatter(x=vg_bwd, y=id_bwd.abs(), name="Backward", line=dict(color=color_bwd), legend="legend2"), row=1, col=2)
            
    # [2-1] Gm
    fig.add_trace(go.Scatter(x=vg_fwd, y=abs(gm_fwd), name="Forward", line=dict(color=color_fwd), legend="legend3"), row=2, col=1)
    fig.add_trace(go.Scatter(x=vg_bwd, y=abs(gm_bwd), name="Backward", line=dict(color=color_bwd), legend="legend3"), row=2, col=1)
    fig.add_vline(x=vg_max_gm, line_width=1.5, line_dash="dot", line_color="green", opacity=0.8, row=2, col=1)
            
    # [2-2] Mobility
    fig.add_trace(go.Scatter(x=vg_fwd, y=mobility_fwd, name="Forward", line=dict(color=color_fwd), legend="legend4"), row=2, col=2)
    fig.add_trace(go.Scatter(x=vg_bwd, y=mobility_bwd, name="Backward", line=dict(color=color_bwd), legend="legend4"), row=2, col=2)
    fig.add_vline(x=vg_max_gm, line_width=1.5, line_dash="dot", line_color="green", opacity=0.8, row=2, col=2)

    # 범례 스타일
    leg_style = dict(
        bgcolor="rgba(255,255,255,0.8)", 
        bordercolor="black", 
        borderwidth=1, 
        xanchor="right", 
        yanchor="top",
        font=dict(color="black")
    )

    # 7. 레이아웃 설정
    fig.update_layout(
        width=1000, height=1000, 
        autosize=False,
        template="plotly_white",
        margin=dict(t=120, b=80, l=100, r=100),
        
        legend=dict(x=0.375, y=1.0, **leg_style),
        legend2=dict(x=1.0, y=1.0, **leg_style),
        legend3=dict(x=0.375, y=0.375, **leg_style),
        legend4=dict(x=1.0, y=0.375, **leg_style)
    )
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='black', size=16)
        annotation['yshift'] = 25
    
    # 격자선이 테두리까지 완전히 닿도록 설정
    common_axis_params = dict(
        ticks="outside", 
        tickwidth=1.5, tickcolor='black', ticklen=8,
        showline=True, linewidth=1.5, linecolor='black', mirror=True,
        showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='dot',
        zeroline=False,
        layer='below traces'
    )

    fig.update_xaxes(title_text="Gate Voltage (V)", dtick=10, **common_axis_params)
    fig.update_yaxes(**common_axis_params)
    
    # 개별 Y축 세부 설정
    fig.update_yaxes(title_text="Drain Current (A)", type="log", dtick=1, exponentformat="power", row=1, col=1)
    fig.update_yaxes(title_text="Drain Current (A)", exponentformat="power", row=1, col=2)
    fig.update_yaxes(title_text="Gₘ (S)", exponentformat="power", row=2, col=1)
    fig.update_yaxes(title_text="Mobility (cm²/V·s)", row=2, col=2)

    st.plotly_chart(fig, use_container_width=False)
