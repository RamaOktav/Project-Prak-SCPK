import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Sistem Kelayakan Beasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1a2744 100%);
    border-right: 1px solid #1e3a5f;
}

/* Cards */
.card {
    background: linear-gradient(135deg, rgba(13,27,42,0.9) 0%, rgba(26,39,68,0.9) 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

.metric-card {
    background: linear-gradient(135deg,
        rgba(0,100,200,0.15) 0%,
        rgba(0,60,120,0.1) 100%);
    border: 1px solid #1e4d8c;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    min-height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

/* Result badges */
.badge-diterima {
    background: linear-gradient(135deg, #00b09b, #96c93d);
    color: white;
    padding: 12px 30px;
    border-radius: 50px;
    font-size: 1.4rem;
    font-weight: 700;
    display: inline-block;
    letter-spacing: 1px;
}
.badge-dipertimbangkan {
    background: linear-gradient(135deg, #f7971e, #ffd200);
    color: #1a1a1a;
    padding: 12px 30px;
    border-radius: 50px;
    font-size: 1.4rem;
    font-weight: 700;
    display: inline-block;
    letter-spacing: 1px;
}
.badge-ditolak {
    background: linear-gradient(135deg, #cb2d3e, #ef473a);
    color: white;
    padding: 12px 30px;
    border-radius: 50px;
    font-size: 1.4rem;
    font-weight: 700;
    display: inline-block;
    letter-spacing: 1px;
}

/* Score display */
.score-big {
    font-family: 'JetBrains Mono', monospace;
    font-size: 4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4facfe, #00f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}

/* Headers */
h1 {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
}
h2, h3 {
    color: #a8d4f5 !important;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e4d8c, transparent);
    margin: 20px 0;
}

/* Grade pill */
.grade-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-left: 8px;
}
.grade-a { background: #00b09b33; color: #00b09b; border: 1px solid #00b09b; }
.grade-b { background: #4facfe33; color: #4facfe; border: 1px solid #4facfe; }
.grade-c { background: #f7971e33; color: #f7971e; border: 1px solid #f7971e; }
.grade-d { background: #ef473a33; color: #ef473a; border: 1px solid #ef473a; }
.grade-f { background: #cb2d3e33; color: #cb2d3e; border: 1px solid #cb2d3e; }

/* Slider labels */
.stSlider label { color: #a8d4f5 !important; font-size: 0.95rem !important; }

/* Info boxes */
.info-box {
    background: rgba(79,172,254,0.08);
    border-left: 3px solid #4facfe;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #a8d4f5;
}

/* Footer */
.footer {
    text-align: center;
    color: #3d5a80;
    font-size: 0.8rem;
    padding: 20px 0;
}
</style>
""", unsafe_allow_html=True)


# Proses Fuzzy
@st.cache_resource
def build_fuzzy_system():
    # Input variabel
    gpa = ctrl.Antecedent(np.arange(0, 4.1, 0.1), 'gpa')
    absences = ctrl.Antecedent(np.arange(0, 31, 1), 'absences')
    study_time = ctrl.Antecedent(np.arange(0, 21, 1), 'study_time')

    # Output variabel
    kelayakan = ctrl.Consequent(np.arange(0, 101, 1), 'kelayakan')

    # Membership functions - GPA
    gpa['rendah'] = fuzz.trapmf(gpa.universe, [0, 0, 2, 2.5])
    gpa['cukup']  = fuzz.trimf(gpa.universe, [2, 2.75, 3.5])
    gpa['tinggi'] = fuzz.trapmf(gpa.universe, [3, 3.5, 4.0, 4.0])

    # Membership functions - Absences
    absences['sedikit'] = fuzz.trapmf(absences.universe, [0, 0, 5, 12])
    absences['sedang']  = fuzz.trimf(absences.universe, [8, 15, 22])
    absences['banyak']  = fuzz.trapmf(absences.universe, [18, 25, 30, 30])

    # Membership functions - Study Time
    study_time['kurang'] = fuzz.trapmf(study_time.universe, [0, 0, 4, 8])
    study_time['cukup']  = fuzz.trimf(study_time.universe, [6, 11, 15])
    study_time['banyak'] = fuzz.trapmf(study_time.universe, [13, 18, 20, 20])

    # Membership functions - Output
    kelayakan['ditolak']         = fuzz.trapmf(kelayakan.universe, [0, 0, 30, 50])
    kelayakan['dipertimbangkan'] = fuzz.trimf(kelayakan.universe, [40, 60, 80])
    kelayakan['diterima']        = fuzz.trapmf(kelayakan.universe, [70, 85, 100, 100])

    # Rules
    rules = [
        # GPA RENDAH
        ctrl.Rule(gpa['rendah'] & absences['banyak']  & study_time['kurang'], kelayakan['ditolak']),
        ctrl.Rule(gpa['rendah'] & absences['banyak']  & study_time['cukup'],  kelayakan['ditolak']),
        ctrl.Rule(gpa['rendah'] & absences['banyak']  & study_time['banyak'], kelayakan['ditolak']),
        ctrl.Rule(gpa['rendah'] & absences['sedang']  & study_time['kurang'], kelayakan['ditolak']),
        ctrl.Rule(gpa['rendah'] & absences['sedang']  & study_time['cukup'],  kelayakan['ditolak']),
        ctrl.Rule(gpa['rendah'] & absences['sedang']  & study_time['banyak'], kelayakan['ditolak']),
        ctrl.Rule(gpa['rendah'] & absences['sedikit'] & study_time['kurang'], kelayakan['ditolak']),
        ctrl.Rule(gpa['rendah'] & absences['sedikit'] & study_time['cukup'],  kelayakan['ditolak']),
        ctrl.Rule(gpa['rendah'] & absences['sedikit'] & study_time['banyak'], kelayakan['ditolak']),

        # GPA CUKUP
        ctrl.Rule(gpa['cukup'] & absences['banyak']  & study_time['kurang'], kelayakan['ditolak']),
        ctrl.Rule(gpa['cukup'] & absences['banyak']  & study_time['cukup'],  kelayakan['ditolak']),
        ctrl.Rule(gpa['cukup'] & absences['banyak']  & study_time['banyak'], kelayakan['ditolak']),
        ctrl.Rule(gpa['cukup'] & absences['sedang']  & study_time['kurang'], kelayakan['ditolak']),
        ctrl.Rule(gpa['cukup'] & absences['sedang']  & study_time['cukup'],  kelayakan['dipertimbangkan']),
        ctrl.Rule(gpa['cukup'] & absences['sedang']  & study_time['banyak'], kelayakan['dipertimbangkan']),
        ctrl.Rule(gpa['cukup'] & absences['sedikit'] & study_time['kurang'], kelayakan['ditolak']),
        ctrl.Rule(gpa['cukup'] & absences['sedikit'] & study_time['cukup'],  kelayakan['dipertimbangkan']),
        ctrl.Rule(gpa['cukup'] & absences['sedikit'] & study_time['banyak'], kelayakan['diterima']),

        # GPA TINGGI → dominan diterima
        ctrl.Rule(gpa['tinggi'] & absences['banyak']  & study_time['kurang'], kelayakan['ditolak']),
        ctrl.Rule(gpa['tinggi'] & absences['banyak']  & study_time['cukup'],  kelayakan['ditolak']),
        ctrl.Rule(gpa['tinggi'] & absences['banyak']  & study_time['banyak'], kelayakan['ditolak']),
        ctrl.Rule(gpa['tinggi'] & absences['sedang']  & study_time['kurang'], kelayakan['dipertimbangkan']),
        ctrl.Rule(gpa['tinggi'] & absences['sedang']  & study_time['cukup'],  kelayakan['dipertimbangkan']),
        ctrl.Rule(gpa['tinggi'] & absences['sedang']  & study_time['banyak'], kelayakan['diterima']),
        ctrl.Rule(gpa['tinggi'] & absences['sedikit'] & study_time['kurang'], kelayakan['dipertimbangkan']),
        ctrl.Rule(gpa['tinggi'] & absences['sedikit'] & study_time['cukup'],  kelayakan['diterima']),
        ctrl.Rule(gpa['tinggi'] & absences['sedikit'] & study_time['banyak'], kelayakan['diterima']),
    ]

    kelayakan_ctrl = ctrl.ControlSystem(rules)
    kelayakan_sim  = ctrl.ControlSystemSimulation(kelayakan_ctrl)

    return kelayakan_sim, gpa, absences, study_time, kelayakan


# ─── LOAD DATA ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Student_performance_data_.csv')
    except:
        df = pd.DataFrame()
    return df


def get_grade_label(gpa_val):
    if gpa_val >= 3.5:
        return "A", "grade-a"
    elif gpa_val >= 3.0:
        return "B", "grade-b"
    elif gpa_val >= 2.5:
        return "C", "grade-c"
    elif gpa_val >= 2.0:
        return "D", "grade-d"
    else:
        return "F", "grade-f"


def get_status(score):
    if score >= 70:
        return "✅ DITERIMA", "badge-diterima"
    elif score >= 45:
        return "⚠️ DIPERTIMBANGKAN", "badge-dipertimbangkan"
    else:
        return "❌ DITOLAK", "badge-ditolak"

def get_activity_bonus(extracurricular, sports, music, volunteering):
    bonus = 0

    if extracurricular:
        bonus += 2

    if sports:
        bonus += 2

    if music:
        bonus += 1

    if volunteering:
        bonus += 2

    return bonus

# ─── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Input Data Siswa")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    nama = st.text_input("Nama Siswa", placeholder="Masukkan nama...")

    st.markdown("### 📊 Parameter Akademik")

    gpa_input = st.slider(
        "GPA (Nilai Rata-rata)",
        min_value=0.0, max_value=4.0, value=2.8, step=0.1,
        help="Grade Point Average skala 0.0 - 4.0"
    )

    grade_label, grade_class = get_grade_label(gpa_input)
    st.markdown(
        f'<div class="info-box">Setara Grade: <span class="grade-pill {grade_class}">{grade_label}</span></div>',
        unsafe_allow_html=True
    )

    absences_input = st.slider(
        "Jumlah Absensi (hari)",
        min_value=0, max_value=30, value=10,
        help="Total hari tidak masuk dalam setahun"
    )

    study_input = st.slider(
        "Waktu Belajar Mingguan (jam)",
        min_value=0.0, max_value=20.0, value=10.0, step=0.5,
        help="Rata-rata jam belajar per minggu"
    )

    st.markdown("### Aktivitas Siswa")

    extracurricular = st.checkbox("Ekstrakurikuler")
    sports = st.checkbox("Olahraga")
    music = st.checkbox("Musik")
    volunteering = st.checkbox("Volunteer / Sosial")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    hitung = st.button("🔍 Hitung Kelayakan", use_container_width=True, type="primary")

    st.markdown("### ℹ️ Keterangan Nilai")
    st.markdown("""
    <div class="info-box">
    🟢 GPA ≥ 3.5 → Grade A<br>
    🔵 3.0 ≤ GPA < 3.5 → Grade B<br>
    🟡 2.5 ≤ GPA < 3.0 → Grade C<br>
    🔴 2.0 ≤ GPA < 2.5 → Grade D<br>
    ⛔ GPA < 2.0 → Grade F
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN CONTENT ──────────────────────────────────────────────
st.markdown("# 🎓 Sistem Kelayakan Beasiswa")
st.markdown("#### Berbasis Logika Fuzzy Mamdani")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Build fuzzy system
sim, fz_gpa, fz_abs, fz_study, fz_kel = build_fuzzy_system()

# ─── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Evaluasi Siswa", "📈 Visualisasi Fuzzy", "📊 Data Statistik"])

# ══════════════════════════════════════════════════════════════
# TAB 1: EVALUASI
# ══════════════════════════════════════════════════════════════
with tab1:
    if hitung or True:  # Always show initial state
        try:
            sim.input['gpa']        = gpa_input
            sim.input['absences']   = absences_input
            sim.input['study_time'] = study_input
            sim.compute()
            score_fuzzy = sim.output['kelayakan']
            bonus = get_activity_bonus(
                extracurricular,
                sports,
                music,
                volunteering
            )

            score = min(score_fuzzy + bonus, 100)
            
        except Exception as e:
            score = 0.0

        status_text, status_class = get_status(score)

        # ── Result Row ──
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            # st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 👤 Identitas")
            st.markdown(f"**Nama:** {nama if nama else 'Belum diisi'}")
            st.markdown(f"**GPA:** `{gpa_input:.1f}` / 4.0")
            st.markdown(f"**Absensi:** `{absences_input}` hari")
            st.markdown(f"**Waktu Belajar:** `{study_input:.1f}` jam/minggu")
            aktivitas = []

            if extracurricular:
                aktivitas.append("Ekskul")

            if sports:
                aktivitas.append("Olahraga")

            if music:
                aktivitas.append("Musik")

            if volunteering:
                aktivitas.append("Volunteer")

            st.markdown(
                f"**Aktivitas:** {', '.join(aktivitas) if aktivitas else 'Tidak ada'}"
            )
            

        with col2:
            # st.markdown('<div class="card" style="text-align:center">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Skor Kelayakan")
            st.markdown(f'<div class="score-big">{score:.1f}</div>', unsafe_allow_html=True)
            st.caption(f"Skor Fuzzy: {score_fuzzy:.1f} | Bonus Aktivitas: +{bonus}")
            st.markdown('<div style="color:#3d5a80; font-size:0.9rem;">dari 100</div>', unsafe_allow_html=True)

            # Progress bar using plotly gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#4facfe'},
                    'bar': {'color': '#4facfe'},
                    'bgcolor': '#0d1b2a',
                    'steps': [
                        {'range': [0, 45],  'color': 'rgba(203,45,62,0.2)'},
                        {'range': [45, 70], 'color': 'rgba(247,151,30,0.2)'},
                        {'range': [70, 100],'color': 'rgba(0,176,155,0.2)'},
                    ],
                    'threshold': {
                        'line': {'color': 'white', 'width': 2},
                        'thickness': 0.75,
                        'value': score
                    }
                },
                number={'font': {'color': '#4facfe', 'size': 40}}
            ))
            fig_gauge.update_layout(
                height=200,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#a8d4f5'}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            # st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            # st.markdown('<div class="card" style="text-align:center; padding-top:40px">', unsafe_allow_html=True)
            st.markdown("#### Status")
            st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Interpretation
            if score >= 70:
                st.markdown("""
                <div class="info-box">
                Siswa memenuhi kriteria dan <strong>layak menerima beasiswa</strong>.
                Pertahankan prestasi akademik!
                </div>
                """, unsafe_allow_html=True)
            elif score >= 45:
                st.markdown("""
                <div class="info-box">
                Siswa <strong>perlu ditinjau lebih lanjut</strong>.
                Tingkatkan GPA dan kurangi absensi.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                Siswa <strong>belum memenuhi kriteria</strong>.
                Fokus pada peningkatan GPA dan kehadiran.
                </div>
                """, unsafe_allow_html=True)
            # st.markdown('</div>', unsafe_allow_html=True)

        # ── Membership Degrees ──
        st.markdown("### 🔢 Derajat Keanggotaan Fuzzy")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            
            st.markdown("**📚 GPA**")
            gpa_u = fz_gpa.universe
            for label in ['rendah', 'cukup', 'tinggi']:
                mf = fuzz.interp_membership(gpa_u, fz_gpa[label].mf, gpa_input)
                color = "#4facfe" if mf > 0 else "#3d5a80"
                st.markdown(f'<div style="color:{color}">• {label.capitalize()}: <strong>{mf:.3f}</strong></div>', unsafe_allow_html=True)
            

        with col_b:
            
            st.markdown("**📅 Absensi**")
            abs_u = fz_abs.universe
            for label in ['sedikit', 'sedang', 'banyak']:
                mf = fuzz.interp_membership(abs_u, fz_abs[label].mf, absences_input)
                color = "#4facfe" if mf > 0 else "#3d5a80"
                st.markdown(f'<div style="color:{color}">• {label.capitalize()}: <strong>{mf:.3f}</strong></div>', unsafe_allow_html=True)
            

        with col_c:
            
            st.markdown("**⏱️ Waktu Belajar**")
            std_u = fz_study.universe
            for label in ['kurang', 'cukup', 'banyak']:
                mf = fuzz.interp_membership(std_u, fz_study[label].mf, study_input)
                color = "#4facfe" if mf > 0 else "#3d5a80"
                st.markdown(f'<div style="color:{color}">• {label.capitalize()}: <strong>{mf:.3f}</strong></div>', unsafe_allow_html=True)
            


# ══════════════════════════════════════════════════════════════
# TAB 2: VISUALISASI FUZZY
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📈 Fungsi Keanggotaan")

    def plot_mf(universe, variable, input_val, title, xlabel):
        fig = go.Figure()
        colors      = ['#4facfe', '#00f2fe', '#a8d4f5']
        fill_colors = ['rgba(79,172,254,0.15)', 'rgba(0,242,254,0.15)', 'rgba(168,212,245,0.15)']
        labels = list(variable.terms.keys())

        for i, label in enumerate(labels):
            mf_vals = variable[label].mf
            fig.add_trace(go.Scatter(
                x=universe, y=mf_vals,
                mode='lines',
                name=label.capitalize(),
                line=dict(color=colors[i % len(colors)], width=2.5),
                fill='tozeroy',
                fillcolor=fill_colors[i % len(fill_colors)]
            ))

        # Vertical line for current value
        fig.add_vline(
            x=input_val, line_dash="dash",
            line_color="#ffd200", line_width=2,
            annotation_text=f"  {input_val}",
            annotation_font_color="#ffd200"
        )

        fig.update_layout(
            title=dict(text=title, font=dict(color='#a8d4f5', size=14)),
            xaxis=dict(title=xlabel, color='#a8d4f5', gridcolor='#1e3a5f'),
            yaxis=dict(title='Derajat Keanggotaan', range=[0, 1.1], color='#a8d4f5', gridcolor='#1e3a5f'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,27,42,0.8)',
            legend=dict(font=dict(color='#a8d4f5'), bgcolor='rgba(0,0,0,0)'),
            height=280,
            margin=dict(l=10, r=10, t=40, b=30)
        )
        return fig

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            plot_mf(fz_gpa.universe, fz_gpa, gpa_input, "GPA", "GPA"),
            use_container_width=True
        )
        st.plotly_chart(
            plot_mf(fz_study.universe, fz_study, study_input, "Waktu Belajar Mingguan", "Jam"),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            plot_mf(fz_abs.universe, fz_abs, absences_input, "Absensi", "Hari"),
            use_container_width=True
        )
        st.plotly_chart(
            plot_mf(fz_kel.universe, fz_kel, score, "Output Kelayakan", "Skor"),
            use_container_width=True
        )


# ══════════════════════════════════════════════════════════════
# TAB 3: STATISTIK DATA
# ══════════════════════════════════════════════════════════════
with tab3:
    df = load_data()

    if df.empty:
        st.warning("⚠️ File `Student_performance_data_.csv` tidak ditemukan. Pastikan file ada di folder yang sama.")
    else:
        st.markdown("### 📊 Ringkasan Dataset")

        col1, col2, col3, col4 = st.columns(4)
        with col1:

            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin-bottom:10px; color:#a8d4f5;">Total Siswa</h4>
                <h2 style="color:#4facfe;">{len(df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
             st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin-bottom:10px; color:#a8d4f5;">Rata-rata GPA</h4>
                <h2 style="color:#4facfe;">{df['GPA'].mean():.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col3:
             st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin-bottom:10px; color:#a8d4f5;">Rata-rata Absensi</h4>
                <h2 style="color:#4facfe;">{df['Absences'].mean():.1f} hari</h2>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin-bottom:10px; color:#a8d4f5;"> lama Belajar(per minggu)</h4>
                <h2 style="color:#4facfe;">{df['StudyTimeWeekly'].mean():.1f} jam</h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Hitung skor fuzzy seluruh siswa
        df_rank = df.copy()

        scores = []

        for _, row in df_rank.iterrows():
            sim_tmp, _, _, _, _ = build_fuzzy_system()

            sim_tmp.input['gpa'] = row['GPA']
            sim_tmp.input['absences'] = row['Absences']
            sim_tmp.input['study_time'] = row['StudyTimeWeekly']

            sim_tmp.compute()

            scores.append(sim_tmp.output['kelayakan'])

        df_rank['Skor_Kelayakan'] = scores

        df_rank['Status'] = df_rank['Skor_Kelayakan'].apply(
            lambda x:
                "Diterima" if x >= 70
                else "Dipertimbangkan" if x >= 45
                else "Ditolak"
        )

        col_left, col_right = st.columns(2)

        with col_left:
            status_counts = df_rank['Status'].value_counts()

            fig_pie = go.Figure(go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                hole=0.4,
                marker=dict(
                    colors=[
                        '#cb2d3e',
                        '#f7971e',
                        '#00b09b'
                    ]
                ),
                textfont=dict(color='white')
            ))

            fig_pie.update_layout(
                title=dict(
                    text="Tingkat Kelolosan Siswa",
                    font=dict(color='#a8d4f5')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#a8d4f5'),
                height=320
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            # Distribusi GPA histogram
            fig_hist = px.histogram(
                df, x='GPA', nbins=30,
                color_discrete_sequence=['#4facfe'],
                title='Distribusi GPA Siswa'
            )
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,27,42,0.8)',
                font=dict(color='#a8d4f5'),
                xaxis=dict(gridcolor='#1e3a5f'),
                yaxis=dict(gridcolor='#1e3a5f'),
                height=320
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Ranking Siswa (Tabel List) ──
        st.markdown("### 🏆 Ranking Siswa Berdasarkan Skor Kelayakan")

        # Tentukan kolom kegiatan yang tersedia
        activity_cols = {
            'Extracurricular': 'Ekskul',
            'Sports':          'Olahraga',
            'Music':           'Musik',
            'Volunteering':    'Volunteer',
        }
        available_act = {k: v for k, v in activity_cols.items() if k in df_rank.columns}

        # Buat kolom Kegiatan gabungan
        def build_activities(row):
            acts = [label for col, label in available_act.items() if row.get(col, 0) == 1]
            return ', '.join(acts) if acts else '—'

        df_rank['Kegiatan'] = df_rank.apply(build_activities, axis=1)

        # Sort & ranking
        ranking_df = df_rank.sort_values('Skor_Kelayakan', ascending=False).reset_index(drop=True)
        ranking_df.insert(0, 'Ranking', ranking_df.index + 1)

        # Filter & pencarian
        col_search, col_filter = st.columns([2, 1])
        with col_search:
            search_id = st.text_input("🔍 Cari ID Siswa", placeholder="Ketik StudentID...")
        with col_filter:
            filter_status = st.selectbox("📌 Filter Status", ["Semua", "Diterima", "Dipertimbangkan", "Ditolak"])

        # Terapkan filter
        display_df = ranking_df.copy()
        if search_id:
            id_col = 'StudentID' if 'StudentID' in display_df.columns else display_df.columns[1]
            display_df = display_df[display_df[id_col].astype(str).str.contains(search_id, case=False)]
        if filter_status != "Semua":
            display_df = display_df[display_df['Status'] == filter_status]

        # Pilih kolom yang akan ditampilkan
        id_col = 'StudentID' if 'StudentID' in display_df.columns else display_df.columns[1]
        show_cols = ['Ranking', id_col, 'GPA', 'Absences', 'StudyTimeWeekly', 'Kegiatan', 'Skor_Kelayakan', 'Status']
        show_cols = [c for c in show_cols if c in display_df.columns]

        table_df = display_df[show_cols].rename(columns={
            id_col:            'ID Siswa',
            'GPA':             'GPA',
            'Absences':        'Absensi (hari)',
            'StudyTimeWeekly': 'Belajar (jam/minggu)',
            'Skor_Kelayakan':  'Skor Kelayakan',
        })

        # Warna baris berdasarkan status
        def highlight_status(row):
            s = row['Status']
            if s == 'Diterima':
                bg = 'background-color: rgba(0,176,155,0.15); color: #00b09b'
            elif s == 'Dipertimbangkan':
                bg = 'background-color: rgba(247,151,30,0.15); color: #f7971e'
            else:
                bg = 'background-color: rgba(203,45,62,0.15); color: #ef473a'
            return [bg] * len(row)

        st.dataframe(
            table_df.style
                .apply(highlight_status, axis=1)
                .format({'GPA': '{:.2f}', 'Skor Kelayakan': '{:.1f}'}),
            use_container_width=True,
            height=450,
        )

        st.caption(f"Menampilkan {len(display_df):,} dari {len(ranking_df):,} siswa")


# ─── FOOTER ────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="footer">Sistem Kelayakan Beasiswa • Logika Fuzzy Mamdani • 2025</div>',
    unsafe_allow_html=True
)