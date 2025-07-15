import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# --- Page Configuration ---
st.set_page_config(
    page_title="Dynamic Wastewater Simulator",
    page_icon="🌍",
    layout="wide"
)

# --- Dynamic Model (ODEs for a CSTR) ---
# This model's internal units are fixed: time in days, mass in mg, volume in m³
def cstr_model(t, y, Q, V, S_in, Y, k_d, K_s, mu_max, K_La, DO_sat, OTR_factor):
    """
    Defines the ordinary differential equations for the CSTR model.
    y[0] = S (Substrate concentration, mg/L)
    y[1] = X (Biomass concentration, mg/L)
    y[2] = DO (Dissolved Oxygen concentration, mg/L)
    """
    S, X, DO = y

    # Specific growth rate (Monod equation with DO limitation)
    mu = mu_max * (S / (K_s + S)) * (DO / (0.5 + DO))

    # Rates of change (ODEs)
    dS_dt = (Q / V) * (S_in - S) - (mu / Y) * X
    dX_dt = mu * X - k_d * X - (Q / V) * X
    
    # Oxygen Uptake Rate (OUR) is proportional to growth
    OUR = (mu / Y) * X * OTR_factor
    dDO_dt = K_La * (DO_sat - DO) - OUR

    return [dS_dt, dX_dt, dDO_dt]

# --- UI Layout ---
st.title("🌍 Dynamic Wastewater Process Simulator")
st.markdown("This app simulates the dynamic response of an activated sludge process. Use the sidebar to configure parameters in your preferred unit system and run the simulation.")

# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Configuration")

# Unit System Selection
unit_system = st.sidebar.radio(
    "Unit System",
    ["Conventional (m³/day, mg/L)", "SI (m³/s, kg/m³)"],
    help="Select the unit system for inputs and outputs."
)

# --- Input Sections based on Unit System ---
if unit_system == "Conventional (m³/day, mg/L)":
    # Plant Design Parameters
    with st.sidebar.expander("Plant Design Parameters", expanded=True):
        Q = st.number_input("Average Daily Flow (Q, m³/day)", min_value=0.0, value=5000.0, step=100.0)
        V = st.number_input("Aeration Tank Volume (V, m³)", min_value=0.0, value=2500.0, step=100.0)
        S_in_base = st.number_input("Normal Influent BOD (S_in, mg/L)", min_value=0.0, value=250.0, step=10.0)
        X_initial = st.number_input("Initial Biomass (MLVSS, mg/L)", min_value=0.0, value=2800.0, step=100.0)
    # Kinetic & Oxygen Parameters
    with st.sidebar.expander("Kinetic & Oxygen Parameters"):
        k_d = st.slider("Endogenous Decay (k_d, 1/day)", 0.04, 0.1, 0.06, 0.01)
        K_La = st.slider("Oxygen Transfer Coeff. (K_La, 1/day)", 100.0, 300.0, 240.0, 10.0)
        K_s = st.slider("Half-Saturation Constant (K_s, mg/L)", 10.0, 50.0, 20.0, 5.0)
        # Shock Load Inputs
        st.sidebar.header("🕹️ Simulation Control")
        shock_load_mag = st.sidebar.number_input("Shock Load BOD (mg/L)", min_value=0.0, value=600.0, step=50.0)
else: # SI units
    with st.sidebar.expander("Plant Design Parameters", expanded=True):
        Q = st.number_input("Average Flow (Q, m³/s)", min_value=0.0, value=0.058, step=0.01, format="%.3f")
        V = st.number_input("Aeration Tank Volume (V, m³)", min_value=0.0, value=2500.0, step=100.0)
        S_in_base = st.number_input("Normal Influent BOD (S_in, kg/m³)", min_value=0.0, value=0.25, step=0.01, format="%.2f")
        X_initial = st.number_input("Initial Biomass (MLVSS, kg/m³)", min_value=0.0, value=2.8, step=0.1, format="%.2f")
    with st.sidebar.expander("Kinetic & Oxygen Parameters"):
        k_d = st.slider("Endogenous Decay (k_d, 1/s)", 0.0000005, 0.0000012, 0.0000007, 0.0000001, format="%.7f")
        K_La = st.slider("Oxygen Transfer Coeff. (K_La, 1/s)", 0.001, 0.004, 0.0028, 0.0001, format="%.4f")
        K_s = st.slider("Half-Saturation Constant (K_s, kg/m³)", 0.01, 0.05, 0.02, 0.005, format="%.3f")
        # Shock Load Inputs
        st.sidebar.header("🕹️ Simulation Control")
        shock_load_mag = st.sidebar.number_input("Shock Load BOD (kg/m³)", min_value=0.0, value=0.6, step=0.1, format="%.2f")

# Shared Parameters (Unit-agnostic or simple)
with st.sidebar.expander("Shared Kinetic Parameters"):
    Y = st.slider("Biomass Yield (Y, mg VSS/mg BOD)", 0.3, 0.7, 0.5, 0.05)
    mu_max_d = st.slider("Max Specific Growth Rate (μ_max, 1/day)", 2.0, 8.0, 4.0, 0.5, help="This parameter is always entered in 1/day.")
    DO_sat = st.slider("DO Saturation (DO_sat, g/m³ or mg/L)", 8.0, 10.0, 9.2, 0.1)
    OTR_factor = st.slider("Oxygen Usage Factor (1-Y)", 0.3, 0.7, 0.5, 0.05, help="Represents (1-Y), the fraction of substrate used for respiration.")

# Shared Simulation Control
sim_duration = st.sidebar.slider("Simulation Duration (days)", 1, 30, 10)
shock_start_time = st.sidebar.slider("Shock Start Time (day)", 0.0, float(sim_duration-1), 2.0, 0.5)
shock_duration = st.sidebar.slider("Shock Duration (days)", 0.1, 5.0, 1.0, 0.1)


# --- Main Panel for Simulation ---
if st.button("🚀 Run Dynamic Simulation"):

    # --- Unit Conversion to Model's Base Units ---
    if unit_system == "SI (m³/s, kg/m³)":
        Q_model = Q * 86400  # m³/s to m³/day
        S_in_base_model = S_in_base * 1000  # kg/m³ to mg/L
        X_initial_model = X_initial * 1000 # kg/m³ to mg/L
        k_d_model = k_d * 86400  # 1/s to 1/day
        K_La_model = K_La * 86400 # 1/s to 1/day
        K_s_model = K_s * 1000 # kg/m³ to mg/L
        shock_load_mag_model = shock_load_mag * 1000 # kg/m³ to mg/L
    else: # Conventional units are already the model's base units
        Q_model, S_in_base_model, X_initial_model, k_d_model, K_La_model, K_s_model, shock_load_mag_model = \
        Q, S_in_base, X_initial, k_d, K_La, K_s, shock_load_mag

    # --- Simulation Logic ---
    t_span = [0, sim_duration]
    t_eval = np.linspace(t_span[0], t_span[1], num=sim_duration * 200)

    def get_S_in(t):
        if shock_start_time <= t <= shock_start_time + shock_duration:
            return shock_load_mag_model
        return S_in_base_model

    def model_with_shock(t, y):
        S_in_t = get_S_in(t)
        return cstr_model(t, y, Q_model, V, S_in_t, Y, k_d_model, K_s_model, mu_max_d, K_La_model, DO_sat, OTR_factor)

    y0 = [S_in_base_model, X_initial_model, 2.0]

    with st.spinner('Running simulation... this may take a moment.'):
        sol = solve_ivp(model_with_shock, t_span, y0, t_eval=t_eval, method='RK45', dense_output=True)

        # --- Unit Conversion for Outputs ---
        if unit_system == "SI (m³/s, kg/m³)":
            bod_results = sol.y[0] / 1000
            biomass_results = sol.y[1] / 1000
            do_results = sol.y[2] / 1000
            bod_label, biomass_label, do_label = "Effluent BOD (kg/m³)", "Biomass (MLVSS, kg/m³)", "Dissolved Oxygen (kg/m³)"
        else:
            bod_results, biomass_results, do_results = sol.y[0], sol.y[1], sol.y[2]
            bod_label, biomass_label, do_label = "Effluent BOD (mg/L)", "Biomass (MLVSS, mg/L)", "Dissolved Oxygen (mg/L)"

        results_df = pd.DataFrame({
            'Time (days)': sol.t,
            bod_label: bod_results,
            biomass_label: biomass_results,
            do_label: do_results
        })

    st.success("✅ Simulation Complete!")

    # --- Display Results ---
    st.header("📊 Simulation Results")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Substrate & Biomass")
        st.line_chart(results_df, x='Time (days)', y=[bod_label, biomass_label])

    with col2:
        st.subheader("Dissolved Oxygen")
        st.line_chart(results_df, x='Time (days)', y=do_label, color='#FF4B4B')

# --- Instructions and Notes ---
with st.expander("About This Dynamic Simulator"):
    st.markdown("""
        ### What's New?
        You can now select your preferred unit system—**Conventional** (m³/day, mg/L) or **SI** (m³/s, kg/m³)—from the sidebar. The app automatically handles all necessary conversions for the simulation.
        ### The Model
        The simulation is based on a **Completely Stirred Tank Reactor (CSTR)** model using **Monod kinetics**.
        *Disclaimer: This is a simplified educational model. Real-world systems are more complex and professional designs require comprehensive software like GPS-X, BioWin, or WEST.*
    """)
