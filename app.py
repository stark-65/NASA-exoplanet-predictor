import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Exoplanet Predictor", page_icon="🪐", layout="wide")

@st.cache_resource
def load_models():
    clf = joblib.load("exoplanet_production/xgb_classifier.pkl")
    reg = joblib.load("exoplanet_production/lgbm_regressor.pkl")
    le  = joblib.load("exoplanet_production/label_encoder.pkl")
    return clf, reg, le

clf, reg, le = load_models()

def predict(raw):
    pl_orbper   = raw['pl_orbper']
    pl_orbsmax  = raw['pl_orbsmax']
    pl_orbeccen = raw['pl_orbeccen']
    pl_orbincl  = raw['pl_orbincl']
    pl_rade     = raw['pl_rade']
    pl_bmasse   = raw['pl_bmasse']
    pl_dens     = raw['pl_dens']
    st_teff     = raw['st_teff']
    st_rad      = raw['st_rad']
    st_mass     = raw['st_mass']
    st_met      = raw['st_met']
    st_lum      = raw['st_lum']
    st_logg     = raw['st_logg']
    st_age      = raw['st_age']
    st_dens     = raw['st_dens']
    sy_dist     = raw['sy_dist']
    sy_snum     = raw['sy_snum']
    sy_pnum     = raw['sy_pnum']
    tran_flag   = raw['tran_flag']
    rv_flag     = raw['rv_flag']
    disc_year   = raw['disc_year']
    disc_enc    = raw['disc_enc']

    log_pl_orbper        = np.log1p(pl_orbper)
    log_pl_orbsmax       = np.log1p(pl_orbsmax)
    log_pl_bmasse        = np.log1p(pl_bmasse)
    log_sy_dist          = np.log1p(sy_dist)
    st_rad_AU            = st_rad * 0.00465
    a_over_Rstar         = pl_orbsmax / st_rad_AU
    orbital_energy_proxy = -st_mass / pl_orbsmax
    hill_radius          = pl_orbsmax * ((pl_bmasse / (3 * st_mass * 333000)) ** (1/3))
    is_circular          = int(pl_orbeccen < 0.05)

    clf_vec = np.array([[
        log_pl_orbper, log_pl_orbsmax, pl_orbeccen,
        pl_orbincl, st_teff, st_rad, st_mass, st_met,
        st_lum, st_logg, st_age, st_dens,
        log_sy_dist, sy_snum, sy_pnum,
        a_over_Rstar, orbital_energy_proxy, hill_radius,
        is_circular, disc_enc, disc_year,
        tran_flag, rv_flag, 0, 0
    ]])

    reg_vec = np.array([[
        log_pl_orbper, log_pl_orbsmax, pl_rade, log_pl_bmasse,
        pl_dens, pl_orbeccen, pl_orbincl,
        st_teff, st_rad, st_mass, st_met, st_lum,
        st_logg, st_age, st_dens, log_sy_dist,
        sy_snum, sy_pnum,
        a_over_Rstar, orbital_energy_proxy, hill_radius,
        is_circular, disc_enc, disc_year,
        tran_flag, rv_flag, 0, 0
    ]])

    planet_class = le.inverse_transform(clf.predict(clf_vec))[0]
    class_proba  = dict(zip(le.classes_, clf.predict_proba(clf_vec)[0].round(3)))
    temperature  = round(float(reg.predict(reg_vec)[0]), 1)

    st_lum_lin = 10 ** st_lum
    hz_inner   = round(0.95 * np.sqrt(st_lum_lin), 3)
    hz_outer   = round(1.67 * np.sqrt(st_lum_lin), 3)
    in_hz      = hz_inner <= pl_orbsmax <= hz_outer

    return planet_class, class_proba, temperature, in_hz, hz_inner, hz_outer

st.title("🪐 NASA Exoplanet Predictor")
st.markdown("Enter planet and star measurements to predict **planet type**, **temperature**, and **habitability**.")
st.markdown("---")

st.subheader("⚡ Quick Presets")
col1, col2, col3, col4 = st.columns(4)

presets = {
    "earth": dict(pl_orbper=365.0, pl_orbsmax=1.0, pl_orbeccen=0.017,
                  pl_orbincl=90.0, pl_rade=1.0, pl_bmasse=1.0, pl_dens=5.51,
                  st_teff=5778.0, st_rad=1.0, st_mass=1.0, st_met=0.0,
                  st_lum=0.0, st_logg=4.44, st_age=4.6, st_dens=1.41,
                  sy_dist=1.3, sy_snum=1, sy_pnum=8, tran_flag=0,
                  rv_flag=0, disc_year=2024, disc_enc=2),
    "hotjupiter": dict(pl_orbper=3.5, pl_orbsmax=0.05, pl_orbeccen=0.01,
                       pl_orbincl=85.0, pl_rade=12.0, pl_bmasse=300.0, pl_dens=1.2,
                       st_teff=5900.0, st_rad=1.1, st_mass=1.05, st_met=0.15,
                       st_lum=0.1, st_logg=4.4, st_age=3.0, st_dens=1.2,
                       sy_dist=200.0, sy_snum=1, sy_pnum=1, tran_flag=1,
                       rv_flag=1, disc_year=2010, disc_enc=9),
    "superearth": dict(pl_orbper=20.0, pl_orbsmax=0.12, pl_orbeccen=0.05,
                       pl_orbincl=88.0, pl_rade=2.0, pl_bmasse=8.0, pl_dens=4.5,
                       st_teff=4200.0, st_rad=0.6, st_mass=0.6, st_met=-0.1,
                       st_lum=-0.6, st_logg=4.6, st_age=6.0, st_dens=2.8,
                       sy_dist=50.0, sy_snum=1, sy_pnum=3, tran_flag=1,
                       rv_flag=0, disc_year=2018, disc_enc=9),
    "neptune": dict(pl_orbper=45.0, pl_orbsmax=0.25, pl_orbeccen=0.08,
                    pl_orbincl=87.0, pl_rade=4.0, pl_bmasse=17.0, pl_dens=1.6,
                    st_teff=5200.0, st_rad=0.9, st_mass=0.85, st_met=0.0,
                    st_lum=-0.15, st_logg=4.5, st_age=5.5, st_dens=1.6,
                    sy_dist=80.0, sy_snum=1, sy_pnum=2, tran_flag=1,
                    rv_flag=0, disc_year=2015, disc_enc=9),
}

if 'preset' not in st.session_state:
    st.session_state.preset = "earth"

if col1.button("🌍 Earth-like"):   st.session_state.preset = "earth"
if col2.button("♃ Hot Jupiter"):   st.session_state.preset = "hotjupiter"
if col3.button("🔴 Super Earth"):  st.session_state.preset = "superearth"
if col4.button("🌊 Neptune-like"): st.session_state.preset = "neptune"

p = presets[st.session_state.preset]

st.markdown("---")
st.subheader("🌍 Planet Parameters")
c1, c2, c3 = st.columns(3)

with c1:
    pl_orbper   = st.slider("Orbital Period (days)",   0.5,   5000.0, float(p['pl_orbper']),   0.5)
    pl_orbsmax  = st.slider("Semi-major Axis (AU)",    0.01,  10.0,   float(p['pl_orbsmax']),  0.01)
    pl_orbeccen = st.slider("Eccentricity",            0.0,   0.95,   float(p['pl_orbeccen']), 0.01)
with c2:
    pl_rade     = st.slider("Planet Radius (R⊕)",      0.3,   25.0,   float(p['pl_rade']),     0.1)
    pl_bmasse   = st.slider("Planet Mass (M⊕)",        0.1,   5000.0, float(p['pl_bmasse']),   0.1)
    pl_dens     = st.slider("Planet Density (g/cm³)",  0.1,   15.0,   float(p['pl_dens']),     0.1)
with c3:
    pl_orbincl  = st.slider("Orbital Inclination (°)", 0.0,   90.0,   float(p['pl_orbincl']),  0.1)
    tran_flag   = st.selectbox("Transit Detected?",    [0,1],          index=int(p['tran_flag']))
    rv_flag     = st.selectbox("RV Detected?",         [0,1],          index=int(p['rv_flag']))

st.subheader("⭐ Star Parameters")
s1, s2, s3 = st.columns(3)

with s1:
    st_teff  = st.slider("Stellar Temperature (K)",  2500.0, 10000.0, float(p['st_teff']),  50.0)
    st_rad   = st.slider("Stellar Radius (Solar)",   0.1,    5.0,     float(p['st_rad']),   0.05)
    st_mass  = st.slider("Stellar Mass (Solar)",     0.1,    3.0,     float(p['st_mass']),  0.05)
with s2:
    st_met   = st.slider("Metallicity [Fe/H]",      -1.0,   1.0,     float(p['st_met']),   0.01)
    st_lum   = st.slider("Luminosity (log Solar)",  -3.0,   3.0,     float(p['st_lum']),   0.05)
    st_logg  = st.slider("Surface Gravity (log g)",  2.0,   5.5,     float(p['st_logg']),  0.05)
with s3:
    st_age   = st.slider("Stellar Age (Gyr)",        0.1,   13.0,    float(p['st_age']),   0.1)
    st_dens  = st.slider("Stellar Density (g/cm³)",  0.01,  10.0,    float(p['st_dens']),  0.05)
    sy_dist  = st.slider("Distance (parsecs)",       0.1,   5000.0,  float(p['sy_dist']),  1.0)

st.subheader("🔭 System Info")
i1, i2, i3 = st.columns(3)
with i1: sy_snum   = st.slider("Number of Stars",   1, 4,    int(p['sy_snum']))
with i2: sy_pnum   = st.slider("Number of Planets", 1, 8,    int(p['sy_pnum']))
with i3: disc_year = st.slider("Discovery Year",    1990, 2025, int(p['disc_year']))

st.markdown("---")

if st.button("🔍  PREDICT", use_container_width=True):
    raw = dict(
        pl_orbper=pl_orbper, pl_orbsmax=pl_orbsmax,
        pl_orbeccen=pl_orbeccen, pl_orbincl=pl_orbincl,
        pl_rade=pl_rade, pl_bmasse=pl_bmasse, pl_dens=pl_dens,
        st_teff=st_teff, st_rad=st_rad, st_mass=st_mass,
        st_met=st_met, st_lum=st_lum, st_logg=st_logg,
        st_age=st_age, st_dens=st_dens, sy_dist=sy_dist,
        sy_snum=sy_snum, sy_pnum=sy_pnum,
        tran_flag=tran_flag, rv_flag=rv_flag,
        disc_year=disc_year, disc_enc=9
    )

    planet_class, class_proba, temperature, in_hz, hz_inner, hz_outer = predict(raw)

    emoji_map = {
        "Rocky/Terrestrial":       "🪨",
        "Sub-Neptune/Super-Earth": "🌊",
        "Neptune-like":            "💙",
        "Gas Giant":               "🟤",
        "Water/Volatile-rich":     "💧",
    }
    emoji = emoji_map.get(planet_class, "🪐")

    st.markdown("---")
    st.subheader("📊 Prediction Results")

    r1, r2, r3 = st.columns(3)
    r1.metric("🪐 Planet Class",    f"{emoji} {planet_class}")
    r2.metric("🌡️ Eq. Temperature", f"{temperature} K",
              delta=f"{round(temperature-288,1)} K vs Earth")
    r3.metric("🌿 Habitable Zone",  "✅ YES" if in_hz else "❌ NO",
              delta=f"{hz_inner}–{hz_outer} AU")

    st.subheader("📈 Class Probabilities")
    for cls, prob in sorted(class_proba.items(), key=lambda x: -x[1]):
        e = emoji_map.get(cls, "🪐")
        st.write(f"{e} **{cls}**")
        st.progress(float(prob), text=f"{prob*100:.1f}%")

    st.markdown("---")
    st.subheader("🌍 Habitability Assessment")

    temp_ok = 200 <= temperature <= 400
    size_ok = pl_rade <= 2.0
    ecc_ok  = pl_orbeccen <= 0.3

    ca, cb, cc, cd = st.columns(4)
    ca.metric("HZ Position", "✅ Inside"  if in_hz   else "❌ Outside")
    cb.metric("Temperature", "✅ OK"      if temp_ok  else "⚠️ Extreme")
    cc.metric("Planet Size", "✅ Rocky"   if size_ok  else "⚠️ Too Large")
    cd.metric("Orbit Shape", "✅ Stable"  if ecc_ok   else "⚠️ Eccentric")

    score = sum([in_hz, temp_ok, size_ok, ecc_ok])
    verdict = {
        4: "🟢 Strong habitability candidate",
        3: "🟡 Moderate candidate — some conditions favourable",
        2: "🟠 Weak candidate — significant challenges",
        1: "🔴 Poor candidate — hostile conditions",
        0: "⛔ Not habitable — extreme environment",
    }
    st.info(f"**Habitability Score: {score}/4** — {verdict[score]}")

st.markdown("---")
st.caption("Built on NASA Exoplanet Archive | XGBoost F1=0.858 | LightGBM R²=0.923")