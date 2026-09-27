import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="CardioScan | Heart Risk Assessment",
                   page_icon="❤️", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1150px; }

.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5f8a 50%, #1e3a5f 100%);
    padding: 2.2rem 2.5rem; border-radius: 18px; margin-bottom: 1.8rem;
    box-shadow: 0 10px 30px rgba(30,58,95,0.25);
}
.hero h1 { color: #fff; font-size: 2.1rem; font-weight: 700; margin: 0 0 .4rem 0; letter-spacing: -0.5px; }
.hero p  { color: #c3d7ea; font-size: .98rem; margin: 0; font-weight: 400; }
.hero .pill {
    display:inline-block; background: rgba(255,255,255,.14); color:#e8f1f8;
    padding:.28rem .8rem; border-radius:20px; font-size:.75rem; font-weight:500;
    margin-bottom:.9rem; letter-spacing:.3px;
}

.sec-label {
    font-size:.78rem; font-weight:600; letter-spacing:1.1px; text-transform:uppercase;
    color:#5b7a99; margin:0 0 .7rem 2px;
}

.result-card { border-radius:16px; padding:1.8rem 2rem; margin-top:.5rem; }
.result-high { background: linear-gradient(135deg,#fff5f5,#ffe8e8); border-left:6px solid #d64545; }
.result-low  { background: linear-gradient(135deg,#f2fbf5,#e3f7ea); border-left:6px solid #2e9e5b; }
.result-card h2 { margin:0 0 .3rem 0; font-size:1.55rem; font-weight:700; }
.result-card p  { margin:0; font-size:.93rem; color:#4a5568; }

.score { font-size:3.4rem; font-weight:700; line-height:1; letter-spacing:-2px; }

.meter { height:12px; border-radius:8px; margin:1.1rem 0 .4rem 0;
         background: linear-gradient(90deg,#2e9e5b 0%,#e8b339 50%,#d64545 100%); position:relative; }
.meter .knob { position:absolute; top:-5px; width:6px; height:22px; background:#1a202c;
               border-radius:3px; box-shadow:0 0 0 3px #fff, 0 2px 6px rgba(0,0,0,.3); }
.meter-labels { display:flex; justify-content:space-between; font-size:.72rem; color:#8595a8; font-weight:500; }

.factor { background:#fff; border:1px solid #e6ecf2; border-radius:10px;
          padding:.7rem .9rem; margin-bottom:.5rem; font-size:.86rem; }
.factor .n { color:#64748b; font-size:.75rem; display:block; margin-bottom:.15rem; }
.factor .v { font-weight:600; color:#1a202c; }
.factor.warn { border-left:4px solid #e8b339; background:#fffdf5; }
.factor.bad  { border-left:4px solid #d64545; background:#fff7f7; }
.factor.good { border-left:4px solid #2e9e5b; background:#f7fdf9; }

div.stButton > button {
    background: linear-gradient(135deg,#1e3a5f,#2d5f8a); color:#fff; border:none;
    padding:.75rem 0; border-radius:10px; font-weight:600; font-size:1rem;
    box-shadow:0 4px 14px rgba(30,58,95,.3);
}
div.stButton > button:hover { background: linear-gradient(135deg,#16304f,#24527a); color:#fff; }

.foot { text-align:center; color:#94a3b8; font-size:.78rem; margin-top:2.5rem;
        padding-top:1.2rem; border-top:1px solid #e6ecf2; }
</style>
""", unsafe_allow_html=True)

model = joblib.load("kNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.markdown("""
<div class="hero">
  <span class="pill">MACHINE LEARNING · k-NEAREST NEIGHBOURS</span>
  <h1>CardioScan</h1>
  <p>Heart disease risk assessment from clinical and exercise-test measurements</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.25, 1], gap="large")

with left:
    st.markdown('<p class="sec-label">Patient profile</p>', unsafe_allow_html=True)
    with st.container(border=True):
        a, b = st.columns(2)
        with a:
            age = st.slider("Age", 18, 100, 40)
        with b:
            sex = st.selectbox("Sex", ["M", "F"],
                               format_func=lambda x: "Male" if x == "M" else "Female")

    st.markdown('<p class="sec-label">Clinical measurements</p>', unsafe_allow_html=True)
    with st.container(border=True):
        a, b = st.columns(2)
        with a:
            resting_bp = st.number_input("Resting blood pressure", 80, 200, 120,
                                         help="mm Hg · normal is under 120")
            cholesterol = st.number_input("Cholesterol", 100, 600, 200,
                                          help="mg/dL · desirable is under 200")
        with b:
            fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dL", [0, 1],
                                      format_func=lambda x: "No" if x == 0 else "Yes")
            resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"],
                                       help="ST = ST-T abnormality · LVH = left ventricular hypertrophy")

    st.markdown('<p class="sec-label">Exercise stress test</p>', unsafe_allow_html=True)
    with st.container(border=True):
        a, b = st.columns(2)
        with a:
            chest_pain = st.selectbox("Chest pain type", ["ATA", "NAP", "TA", "ASY"],
                                      help="ATA atypical · NAP non-anginal · TA typical · ASY asymptomatic")
            max_hr = st.slider("Max heart rate achieved", 60, 220, 150)
        with b:
            exercise_angina = st.selectbox("Exercise-induced angina", ["N", "Y"],
                                           format_func=lambda x: "No" if x == "N" else "Yes")
            st_slope = st.selectbox("ST slope", ["Up", "Flat", "Down"],
                                    help="Upsloping is the healthiest pattern")
        oldpeak = st.slider("Oldpeak — ST depression", 0.0, 6.0, 1.0, step=0.1)

    run = st.button("Run assessment", use_container_width=True)

with right:
    st.markdown('<p class="sec-label">Assessment result</p>', unsafe_allow_html=True)

    if not run:
        with st.container(border=True):
            st.markdown(
                "<div style='padding:2.5rem 1rem; text-align:center; color:#94a3b8;'>"
                "<div style='font-size:2.6rem; margin-bottom:.6rem;'>🫀</div>"
                "<div style='font-size:.92rem;'>Enter the measurements and run the "
                "assessment to see the result.</div></div>",
                unsafe_allow_html=True)
    else:
        raw_input = {
            'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol,
            'FastingBS': fasting_bs, 'MaxHR': max_hr, 'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1,
        }
        input_df = pd.DataFrame([raw_input])
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]
        scaled_input = scaler.transform(input_df)
        prediction = int(model.predict(scaled_input)[0])

        risk = None
        if hasattr(model, "predict_proba"):
            risk = float(model.predict_proba(scaled_input)[0][1]) * 100

        if prediction == 1:
            cls, head, note, colr = ("result-high", "Elevated risk",
                                     "The model's nearest cases in the training data were mostly positive for heart disease.",
                                     "#d64545")
        else:
            cls, head, note, colr = ("result-low", "Low risk",
                                     "The model's nearest cases in the training data were mostly negative for heart disease.",
                                     "#2e9e5b")

        st.markdown(f"""
        <div class="result-card {cls}">
          <h2 style="color:{colr};">{head}</h2>
          <p>{note}</p>
        </div>
        """, unsafe_allow_html=True)

        if risk is not None:
            st.markdown(f"""
            <div style="margin-top:1.3rem;">
              <div style="font-size:.78rem; font-weight:600; letter-spacing:1px;
                          text-transform:uppercase; color:#5b7a99;">Model confidence</div>
              <div class="score" style="color:{colr}; margin-top:.35rem;">{risk:.0f}<span
                   style="font-size:1.3rem; font-weight:600;">%</span></div>
              <div style="font-size:.82rem; color:#64748b;">probability of heart disease</div>
              <div class="meter"><div class="knob" style="left:calc({risk:.0f}% - 3px);"></div></div>
              <div class="meter-labels"><span>0% · low</span><span>50%</span><span>100% · high</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<p class="sec-label" style="margin-top:1.6rem;">Key indicators</p>',
                    unsafe_allow_html=True)

        expected_hr = 220 - age
        factors = [
            ("ST slope", st_slope,
             "good" if st_slope == "Up" else ("warn" if st_slope == "Flat" else "bad")),
            ("Chest pain type", chest_pain,
             "bad" if chest_pain == "ASY" else "good"),
            ("Exercise-induced angina", "Yes" if exercise_angina == "Y" else "No",
             "bad" if exercise_angina == "Y" else "good"),
            ("ST depression (oldpeak)", f"{oldpeak:.1f}",
             "good" if oldpeak < 1 else ("warn" if oldpeak < 2 else "bad")),
            ("Max heart rate", f"{max_hr} bpm · age-predicted {expected_hr}",
             "good" if max_hr >= expected_hr * 0.85 else "warn"),
            ("Resting blood pressure", f"{resting_bp} mm Hg",
             "good" if resting_bp < 130 else ("warn" if resting_bp < 140 else "bad")),
            ("Cholesterol", f"{cholesterol} mg/dL",
             "good" if cholesterol < 200 else ("warn" if cholesterol < 240 else "bad")),
        ]
        for name, value, level in factors:
            st.markdown(
                f'<div class="factor {level}"><span class="n">{name}</span>'
                f'<span class="v">{value}</span></div>', unsafe_allow_html=True)

st.markdown("""
<div class="foot">
  Built by <strong>Soumya Manna</strong> · k-Nearest Neighbours trained on the Heart Failure
  Prediction dataset<br>
  Educational project. Not a medical device and not a substitute for professional diagnosis.
</div>
""", unsafe_allow_html=True)