"""
物理金属丝线膨胀系数仿真实验
运行：streamlit run 物理金属丝线膨胀系数仿真实验.py
"""
import streamlit as st
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="金属丝线膨胀系数仿真实验", layout="wide")

def init_global_state():
    if "theme" not in st.session_state:
        st.session_state.theme = "亮色"
    if "assembly_done" not in st.session_state:
        st.session_state.assembly_done = False
    if "zero_offset" not in st.session_state:
        st.session_state.zero_offset = 0.0
    if "data" not in st.session_state:
        st.session_state.data = {}
    if "groups" not in st.session_state:
        st.session_state.groups = []
    if "current_T" not in st.session_state:
        st.session_state.current_T = 25.0
    if "log" not in st.session_state:
        st.session_state.log = []
    if "custom_materials" not in st.session_state:
        st.session_state.custom_materials = {}

init_global_state()

theme = st.session_state.theme
if theme == "暗色":
    bg, fg, card, accent = "#0A0E17", "#CDD6F4", "#1A1F2E", "#38BDF8"
else:
    bg, fg, card, accent = "#F5F7FA", "#1E1E1E", "#FFFFFF", "#0066CC"

st.markdown(f"""
<style>
    html, body, .stApp {{ background: {bg}; color: {fg}; }}
    .stButton>button {{ background: {card}; color: {accent}; border:1px solid {accent}40; border-radius:8px; }}
    .stButton>button:hover {{ background: {accent}22; color: {accent}; }}
    .stMetric {{ background:{card}; border-radius:12px; padding:10px; }}
    .stDataFrame {{ background:{card}; }}
    hr {{ border-color:{accent}40; margin:20px 0; }}
    .gauge-box {{ display:flex; align-items:center; gap:15px; margin:10px 0; }}
    .gauge {{
        position:relative; width:120px; height:120px; border-radius:50%;
        background: #111122; border:3px solid {accent}; box-shadow:0 0 12px {accent}40;
    }}
    .gauge .tick {{ position:absolute; width:2px; height:10px; background:#aaa; transform-origin:bottom center; }}
    .gauge .tick.major {{ width:2px; height:16px; background:#fff; }}
    .gauge .number {{ position:absolute; color:#aaa; font-size:10px; transform:translate(-50%,-50%); }}
    .gauge .pointer {{ position:absolute; bottom:50%; left:50%; width:3px; height:45%; background:red; transform-origin:bottom center; }}
    .gauge .center {{ position:absolute; top:50%; left:50%; width:12px; height:12px; background:white; border-radius:50%; transform:translate(-50%,-50%); }}
    .gauge-value {{ color:{accent}; font-size:14px; text-align:center; margin-top:5px; font-weight:bold; }}
    .rod-bar {{ height:25px; border-radius:4px; transition:width 0.3s; margin:5px 0; }}
    .scale-line {{ display:flex; justify-content:space-between; color:#888; font-size:11px; }}
</style>
""", unsafe_allow_html=True)

DEFAULT_MATERIALS = {
    "铜":   {"alpha": 16.5e-6, "gradient": "linear-gradient(90deg, #CD7F32, #FFA500)"},
    "铝":   {"alpha": 23.1e-6, "gradient": "linear-gradient(90deg, #A8A8A8, #E0E0E0)"},
    "铁":   {"alpha": 12.0e-6, "gradient": "linear-gradient(90deg, #2F3537, #5C6366)"},
    "钢":   {"alpha": 11.5e-6, "gradient": "linear-gradient(90deg, #4A5054, #90989C)"},
    "钨":   {"alpha": 4.5e-6,  "gradient": "linear-gradient(90deg, #3A3A3A, #8B8B8B)"},
    "黄铜": {"alpha": 18.7e-6, "gradient": "linear-gradient(90deg, #B5A642, #D4C964)"},
    "锌":   {"alpha": 30.2e-6, "gradient": "linear-gradient(90deg, #5D6D6E, #95A5A6)"},
    "铅":   {"alpha": 28.9e-6, "gradient": "linear-gradient(90deg, #4A4A4A, #8C8C8C)"},
    "不锈钢": {"alpha": 17.3e-6, "gradient": "linear-gradient(90deg, #808080, #C0C0C0)"},
    "因瓦合金": {"alpha": 1.2e-6,  "gradient": "linear-gradient(90deg, #707070, #B0B0B0)"},
}

def get_full_materials():
    full = DEFAULT_MATERIALS.copy()
    full.update(st.session_state.custom_materials)
    return full

def compute_elongation(L0, T0, T, alpha, noise_std=0.0):
    np.random.seed(int(T * 1000) + int(alpha * 1e10) % 10000)
    dT = T - T0
    dL = alpha * L0 * dT
    if noise_std > 0:
        dL += np.random.normal(0, noise_std)
    return dL

def linear_fit(T_list, dL_list, L0, T0):
    dT = np.array(T_list) - T0
    slope, intercept, r, _, _ = stats.linregress(dT, dL_list)
    return slope / L0, r**2

def a_uncertainty(values):
    if len(values) < 2:
        return 0.0
    avg = np.mean(values)
    std = np.std(values, ddof=1)
    return std / np.sqrt(len(values))

def build_gauge_html(dL, max_micron=200):
    micron = np.clip(dL * 1e6, 0, max_micron)
    angle = (micron / max_micron) * 270.0
    ticks = ""
    for val in range(0, max_micron+1, 10):
        is_major = (val % 50 == 0)
        tick_angle = (val / max_micron) * 270.0 - 135.0
        rad = tick_angle * np.pi / 180
        x = 50 + 45 * np.cos(rad)
        y = 50 + 45 * np.sin(rad)
        cls = "tick major" if is_major else "tick"
        ticks += (
            '<div class="' + cls + '" style="left:' + str(x) + '%;top:' + str(y)
            + '%;transform:translate(-50%,-50%) rotate(' + str(tick_angle+90) + 'deg);"></div>'
        )
        if is_major:
            nx = 50 + 35 * np.cos(rad)
            ny = 50 + 35 * np.sin(rad)
            ticks += (
                '<div class="number" style="left:' + str(nx) + '%;top:' + str(ny) + '%;">'
                + str(val) + '</div>'
            )
    html = (
        '<div class="gauge-box">'
        '<div class="gauge">'
        + ticks +
        '<div class="pointer" style="transform:rotate(' + str(angle) + 'deg);"></div>'
        '<div class="center"></div>'
        '</div>'
        '<div class="gauge-value">' + f"{micron:.1f}" + ' μm</div>'
        '</div>'
    )
    return html

def rod_with_gauge(material_name, L0, dL):
    MATERIAL_ALPHA = get_full_materials()
    info = MATERIAL_ALPHA[material_name]
    max_dL_visual = 5e-4
    percent = min(dL / max_dL_visual * 100, 100)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"<div style='font-size:13px;'>{material_name}</div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="rod-bar" style="width:{50+percent/2}%; background:{info['gradient']};"></div>
            <div class="scale-line">
                <span>0</span><span>L₀={L0*1000:.1f}mm</span><span>{max_dL_visual*1e6:.0f}μm</span>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(build_gauge_html(dL, max_micron=200), unsafe_allow_html=True)

def init_material_data(materials):
    for m in materials:
        if m not in st.session_state.data:
            st.session_state.data[m] = []

def generate_report(materials, L0, T0, T_max, groups):
    MATERIAL_ALPHA = get_full_materials()
    report = "# 金属线膨胀系数实验报告\n\n"
    report += f"**实验日期**：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    report += f"**实验条件**：L₀={L0} m, T₀={T0}℃, 最高温度{T_max}℃\n\n"
    report += "## 实验数据\n"
    for i, group in enumerate(groups):
        report += f"### 第{i+1}组平行实验\n"
        for m in materials:
            pts = group.get(m, [])
            report += f"**{m}**：\n"
            for p in pts:
                report += f"- T={p['T']}℃, ΔL={p['dL']*1e6:.2f}μm\n"
            if len(pts) >= 2:
                T_list = [p["T"] for p in pts]
                dL_list = [p["dL"] for p in pts]
                alpha_fit, r2 = linear_fit(T_list, dL_list, L0, T0)
                report += f"拟合 α={alpha_fit:.3e}/℃, R²={r2:.4f}\n"
    report += "\n## A类不确定度\n"
    all_alphas = {m: [] for m in materials}
    for group in groups:
        for m in materials:
            pts = group.get(m, [])
            if len(pts) >= 2:
                T_list = [p["T"] for p in pts]
                dL_list = [p["dL"] for p in pts]
                alpha_fit, _ = linear_fit(T_list, dL_list, L0, T0)
                all_alphas[m].append(alpha_fit)
    for m, alist in all_alphas.items():
        if alist:
            avg = np.mean(alist)
            ua = a_uncertainty(alist)
            report += f"**{m}**：平均 α={avg:.3e}/℃, u_A={ua:.3e}/℃\n"
    report += "\n## 结论\n"
    report += "本次实验通过加热测量金属丝伸长量，利用最小二乘法拟合得到线膨胀系数，平行实验提高了数据可靠性。\n"
    return report

def main():
    MATERIAL_ALPHA = get_full_materials()
    with st.sidebar:
        st.title("🧪 实验控制")
        theme = st.radio("主题", ["暗色", "亮色"], index=1)
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()

    tabs = st.tabs(["📐 实验准备", "⚙️ 参数设置", "🔥 实验操作", "📊 数据分析"])

    with tabs[0]:
        st.header("实验前准备")
        st.markdown("### 1. 仪器组装")
        st.info("请将金属杆安装在加热管中，连接千分表与顶针，确保千分表测杆与杆端对齐。")
        if st.button("✅ 确认组装完成"):
            st.session_state.assembly_done = True
            st.success("组装完毕，可进行千分表调零。")

        st.markdown("### 2. 千分表精密调零")
        zero = st.slider("指针偏移 (μm)", -10.0, 10.0, st.session_state.zero_offset, 0.1, key="zero_slider")
        st.session_state.zero_offset = zero
        st.markdown(build_gauge_html(zero*1e-6, max_micron=20), unsafe_allow_html=True)
        if abs(zero) < 0.5:
            st.success("调零完成！")
        else:
            st.warning("请调节至指针归零")

        st.markdown("### 3. 预习自测")
        with st.expander("点击展开知识点测验"):
            q1 = st.radio("线膨胀系数 α 的单位是？", ["℃", "1/℃", "m/℃"], index=None)
            if q1 == "1/℃":
                st.success("正确！")
            elif q1:
                st.error("再想想…")
            q2 = st.radio("本实验使用的测量仪器是？", ["游标卡尺", "千分表", "米尺"], index=None)
            if q2 == "千分表":
                st.success("正确！")
            elif q2:
                st.error("提示：微小位移测量。")

        st.markdown("### ⚠️ 常见错误提醒")
        st.warning("- 千分表未调零会导致系统误差\n- 升温过快可能使读数滞后\n- 务必记录室温 T₀")

    with tabs[1]:
        st.subheader("实验条件")
        col1, col2 = st.columns(2)
        with col1:
            L0 = st.number_input("初始长度 L₀ (m)", 0.1, 2.0, 0.5, 0.01)
            T0 = st.number_input("参考温度 T₀ (℃)", 20.0, 30.0, 25.0, 0.5)
        with col2:
            T_max = st.slider("最高加热温度 (℃)", 30, 500, 100, 1)
            step = st.selectbox("升温/降温步长 (℃)", [1.0, 2.0, 5.0])

        noise_enable = st.checkbox("模拟读数噪声")
        noise_std = 1e-6 if noise_enable else 0.0

        materials = st.multiselect("选择对比材料（可多选）", list(MATERIAL_ALPHA.keys()), default=["铜", "铝"])
        
        with st.expander("➕ 新增自定义金属"):
            col_a, col_b = st.columns(2)
            with col_a:
                custom_name = st.text_input("金属名称", placeholder="例如：镁合金")
                custom_alpha = st.number_input("线膨胀系数 (×10⁻⁶ /℃)", 0.0, 100.0, 20.0, 0.1)
            with col_b:
                color1 = st.color_picker("杆体起始色", "#808080")
                color2 = st.color_picker("杆体结束色", "#C0C0C0")
            if st.button("添加到材料库"):
                if not custom_name.strip():
                    st.warning("请输入金属名称")
                elif custom_name in MATERIAL_ALPHA:
                    st.warning("该材料名称已存在")
                else:
                    gradient = f"linear-gradient(90deg, {color1}, {color2})"
                    st.session_state.custom_materials[custom_name] = {
                        "alpha": custom_alpha * 1e-6,
                        "gradient": gradient
                    }
                    st.success(f"已添加「{custom_name}」到材料库")
                    st.rerun()

        if not materials:
            st.warning("至少选一种材料")
            st.stop()

        init_material_data(materials)

        st.markdown("**理论线膨胀系数 (×10⁻⁶/℃)**")
        cols = st.columns(len(materials))
        for i, m in enumerate(materials):
            cols[i].metric(m, f"{MATERIAL_ALPHA[m]['alpha']*1e6:.1f}")

    with tabs[2]:
        if not st.session_state.assembly_done:
            st.warning("请先完成实验准备页面中的组装与调零！")
        else:
            st.subheader("温度控制")
            st.metric("当前温度", f"{st.session_state.current_T:.1f} ℃")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔥 升温 +{}℃".format(step)):
                    new_T = st.session_state.current_T + step
                    if new_T > T_max:
                        st.error("已达上限")
                    else:
                        temp_exists = False
                        for m in materials:
                            for item in st.session_state.data[m]:
                                if item["T"] == new_T:
                                    temp_exists = True
                                    break
                        if not temp_exists:
                            st.session_state.current_T = new_T
                            for m in materials:
                                alpha = MATERIAL_ALPHA[m]["alpha"]
                                dL = compute_elongation(L0, T0, new_T, alpha, noise_std)
                                dL += st.session_state.zero_offset * 1e-6
                                st.session_state.data[m].append({"T": new_T, "dL": dL})
                            st.session_state.log.append(f"升至 {new_T:.1f}℃")
                        st.rerun()
            with col2:
                if st.button("❄️ 降温 -{}℃".format(step)):
                    new_T = st.session_state.current_T - step
                    if new_T < T0:
                        st.error("不能低于室温")
                    else:
                        temp_exists = False
                        for m in materials:
                            for item in st.session_state.data[m]:
                                if item["T"] == new_T:
                                    temp_exists = True
                                    break
                        if not temp_exists:
                            st.session_state.current_T = new_T
                            for m in materials:
                                alpha = MATERIAL_ALPHA[m]["alpha"]
                                dL = compute_elongation(L0, T0, new_T, alpha, noise_std)
                                dL += st.session_state.zero_offset * 1e-6
                                st.session_state.data[m].append({"T": new_T, "dL": dL})
                            st.session_state.log.append(f"降至 {new_T:.1f}℃")
                        st.rerun()
            with col3:
                if st.button("🔄 重置数据"):
                    for m in materials:
                        st.session_state.data[m] = []
                    st.session_state.current_T = T0
                    st.session_state.log = []
                    st.rerun()

            if st.session_state.log:
                st.info(st.session_state.log[-1])

            st.subheader("金属杆 & 千分表")
            for m in materials:
                if st.session_state.data[m]:
                    current_dL = st.session_state.data[m][-1]["dL"]
                else:
                    current_dL = 0.0
                rod_with_gauge(m, L0, current_dL)

            if st.button("📝 保存本组平行实验数据"):
                group = {}
                for m in materials:
                    group[m] = st.session_state.data[m].copy()
                st.session_state.groups.append(group)
                st.success(f"已保存第{len(st.session_state.groups)}组平行实验")

    with tabs[3]:
        st.subheader("数据与拟合")
        if not any(len(st.session_state.data[m]) > 0 for m in materials):
            st.info("请先进行实验操作")
        else:
            sel = st.selectbox("查看材料", materials)
            if st.session_state.data[sel]:
                df = [{"T(℃)": d["T"], "ΔL(m)": f"{d['dL']:.6e}"} for d in st.session_state.data[sel]]
                st.dataframe(df, use_container_width=True)
                if len(st.session_state.data[sel]) >= 2:
                    T_list = [d["T"] for d in st.session_state.data[sel]]
                    dL_list = [d["dL"] for d in st.session_state.data[sel]]
                    alpha_fit, r2 = linear_fit(T_list, dL_list, L0, T0)
                    alpha_true = MATERIAL_ALPHA[sel]["alpha"]
                    err = abs(alpha_fit - alpha_true) / alpha_true * 100
                    col_a, col_e = st.columns(2)
                    with col_a:
                        st.metric("拟合 α", f"{alpha_fit:.3e}/℃")
                    with col_e:
                        st.metric("相对误差", f"{err:.2f}%")

            if any(len(st.session_state.data[m]) >= 2 for m in materials):
                fig, ax = plt.subplots(figsize=(7,4))
                ax.set_facecolor(bg)
                fig.patch.set_facecolor(bg)
                for m in materials:
                    pts = st.session_state.data[m]
                    if len(pts) < 2:
                        continue
                    T_pts = [p["T"] for p in pts]
                    dL_pts = [p["dL"] for p in pts]
                    ax.scatter(T_pts, dL_pts, label=m, s=30)
                    a_fit, _ = linear_fit(T_pts, dL_pts, L0, T0)
                    T_line = np.linspace(min(T_pts), max(T_pts), 50)
                    ax.plot(T_line, a_fit * L0 * (T_line - T0), linestyle='--', linewidth=1.5)
                ax.set_xlabel('温度(℃)', color=fg)
                ax.set_ylabel('伸长量 ΔL (m)', color=fg)
                ax.tick_params(colors=fg, labelsize=10)
                ax.legend(facecolor=card, labelcolor=fg)
                st.pyplot(fig)

            if st.session_state.groups:
                st.subheader("平行实验A类不确定度")
                all_alphas = {m: [] for m in materials}
                for group in st.session_state.groups:
                    for m in materials:
                        pts = group.get(m, [])
                        if len(pts) >= 2:
                            T_pts = [p["T"] for p in pts]
                            dL_pts = [p["dL"] for p in pts]
                            a_fit, _ = linear_fit(T_pts, dL_pts, L0, T0)
                            all_alphas[m].append(a_fit)
                rows = []
                for m, alist in all_alphas.items():
                    if alist:
                        avg = np.mean(alist)
                        ua = a_uncertainty(alist)
                        rows.append({"材料": m, "平均α": f"{avg:.3e}", "u_A": f"{ua:.3e}"})
                if rows:
                    st.table(rows)

                if st.button("📄 生成实验报告"):
                    report = generate_report(materials, L0, T0, T_max, st.session_state.groups)
                    st.download_button("下载报告 (TXT)", report, "exp_report.txt", "text/plain")
                    st.text_area("报告预览", report, height=300)

if __name__ == "__main__":
    main()
