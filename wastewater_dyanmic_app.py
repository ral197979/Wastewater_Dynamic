import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# --- Page Configuration ---
st.set_page_config(
    page_title="Dynamic Wastewater Simulator",
    page_icon="📈",
    layout="wide"
)

# --- Dynamic Model (ODEs for a CSTR) ---
def cstr_model(t, y, Q, V, S_in, Y, k_d, K_s, mu_max, K_La, DO_sat, OTR_factor):
    """
    Defines the ordinary differential equations for the CSTR model.
    
    y[0] = S (Substrate concentration, mg/L)
    y[1] = X (Biomass concentration, mg/L)
    y[2] = DO (Dissolved Oxygen concentration, mg/L)
    """
    S, X, DO = y
    
    # Specific growth rate (Monod equation)
    mu = mu_max * (S / (K_s + S)) * (DO / (0.5 + DO)) # Added DO limitation
    
    # Rates of change (ODEs)
    dS_dt = (Q / V) * (S_in - S) - (mu / Y) * X
    dX_dt = mu * X - k_d * X - (Q / V) * X
    
    # Oxygen Uptake Rate (OUR) is proportional to growth
    OUR = (mu / Y) * X * OTR_factor 
    dDO_dt = K_La * (DO_sat - DO) - OUR
    
    return [dS_dt, dX_dt, dDO_dt]

# --- UI Layout ---
st.title("📈 Dynamic Wastewater Process Simulator")
st.markdown("""
This app simulates the dynamic response of an activated sludge process in a CSTR (Completely Stirred Tank Reactor).
- **Configure** the plant parameters and simulation settings in the sidebar.
- **Introduce a shock load** to see how the system handles upsets.
- **Run the simulation** to view the results on the time-series charts.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Configuration")

# Plant Design Parameters
with st.sidebar.expander("Plant Design Parameters", expanded=True):
    Q = st.number_input("Average Daily Flow (Q, m³/day)", min_value=0.0, value=5000.0, step=100.0)
    V = st.number_input("Aeration Tank Volume (V, m³)", min_value=0.0, value=2500.0, step=100.0)
    S_in_base = st.number_input("Normal Influent BOD (S_in, mg/L)", min_value=0.0, value=250.0, step=10.0)
    X_initial = st.number_input("Initial Biomass (MLVSS, mg/L)", min_value=0.0, value=2800.0, step=100.0)

# Kinetic & Oxygen Parameters
with st.sidebar.expander("Kinetic & Oxygen Parameters"):
    Y = st.slider("Biomass Yield (Y, mg VSS/mg BOD)", 0.3, 0.7, 0.5, 0.05)
    k_d = st.slider("Endogenous Decay (k_d, 1/day)", 0.04, 0.1, 0.06, 0.01)
    mu_max = st.slider("Max Specific Growth Rate (μ_max, 1/day)", 2.0, 8.0, 4.0, 0.5)
    K_s = st.slider("Half-Saturation Constant (K_s, mg/L)", 10.0, 50.0, 20.0, 5.0)
    K_La = st.slider("Oxygen Transfer Coeff. (K_La, 1/day)", 100.0, 300.0, 240.0, 10.0)
    DO_sat = st.slider("DO Saturation (DO_sat, mg/L)", 8.0, 10.0, 9.2, 0.1)
    OTR_factor = st.slider("Oxygen Usage Factor (1-Y)", 0.3, 0.7, 0.5, 0.05, help="Represents (1-Y), the fraction of substrate used for respiration.")


# Simulation Control
st.sidebar.header("🕹️ Simulation Control")
sim_duration = st.sidebar.slider("Simulation Duration (days)", 1, 30, 10)
shock_load_mag = st.sidebar.number_input("Shock Load BOD (mg/L)", min_value=0.0, value=600.0, step=50.0)
shock_start_time = st.sidebar.slider("Shock Start Time (day)", 0.0, float(sim_duration-1), 2.0, 0.5)
shock_duration = st.sidebar.slider("Shock Duration (days)", 0.1, 5.0, 1.0, 0.1)

# --- Main Panel for Simulation ---
if st.button("🚀 Run Dynamic Simulation"):
    
    # --- Simulation Logic ---
    # Define time span and evaluation points
    t_span = [0, sim_duration]
    t_eval = np.linspace(t_span[0], t_span[1], num=sim_duration * 100)
    
    # Define the shock load influent function
    def get_S_in(t):
        if shock_start_time <= t <= shock_start_time + shock_duration:
            return shock_load_mag
        return S_in_base
        
    # We need to solve the ODE with a time-varying input (S_in)
    # The solver 'solve_ivp' cannot directly handle this, so we solve it in a loop
    # or wrap it in a class. For simplicity, we create a function that re-defines the model at each step.
    # A more efficient way is to pass parameters, but this shows the logic clearly.
    def model_with_shock(t, y):
        S_in_t = get_S_in(t)
        return cstr_model(t, y, Q, V, S_in_t, Y, k_d, K_s, mu_max, K_La, DO_sat, OTR_factor)

    # Initial conditions
    y0 = [S_in_base, X_initial, 2.0] # S, X, DO (start DO at 2.0 mg/L)

    with st.spinner('Running simulation... this may take a moment.'):
        # Solve the ODEs
        sol = solve_ivp(model_with_shock, t_span, y0, t_eval=t_eval, method='RK45', dense_output=True)

        # Process results into a DataFrame for plotting
        results_df = pd.DataFrame({
            'Time (days)': sol.t,
            'Effluent BOD (mg/L)': sol.y[0],
            'Biomass (MLVSS, mg/L)': sol.y[1],
            'Dissolved Oxygen (mg/L)': sol.y[2]
        })

    st.success("✅ Simulation Complete!")

    # --- Display Results ---
    st.header("📊 Simulation Results")
    
    # Create columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Substrate & Biomass")
        st.line_chart(results_df, x='Time (days)', y=['Effluent BOD (mg/L)', 'Biomass (MLVSS, mg/L)'])

    with col2:
        st.subheader("Dissolved Oxygen")
        st.line_chart(results_df, x='Time (days)', y='Dissolved Oxygen (mg/L)', color='#FF4B4B')
    
    # Display summary metrics
    st.header("📈 Key Performance Indicators")
    max_bod = results_df['Effluent BOD (mg/L)'].max()
    min_do = results_df['Dissolved Oxygen (mg/L)'].min()
    recovery_time_df = results_df[results_df['Time (days)'] > shock_start_time + shock_duration]
    try:
        recovery_time = recovery_time_df[recovery_time_df['Effluent BOD (mg/L)'] <= S_in_base * 0.1].iloc[0]['Time (days)']
        st.metric("Time to Recover (days)", f"{recovery_time - (shock_start_time + shock_duration):.2f}")
    except IndexError:
        st.metric("Time to Recover (days)", "Not recovered")

    st.metric("Peak Effluent BOD (mg/L)", f"{max_bod:.2f}")
    st.metric("Minimum Dissolved Oxygen (mg/L)", f"{min_do:.2f}")


# --- Instructions and Notes ---
with st.expander("About This Dynamic Simulator"):
    st.markdown("""
        ### What's Changed?
        This app has been upgraded from a static calculator to a dynamic simulator.
        - **Static Calculator:** Solves algebraic equations for a single, steady-state condition. Useful for initial sizing.
        - **Dynamic Simulator:** Solves ordinary differential equations (ODEs) over time to show how the system responds to changes. This is crucial for assessing process stability and resilience.

        ### The Model
        The simulation is based on a **Completely Stirred Tank Reactor (CSTR)** model using the famous **Monod kinetics** for microbial growth. It tracks the moment-by-moment changes in:
        1.  **Substrate (S):** `dS/dt = Inflow - Outflow - Consumption`
        2.  **Biomass (X):** `dX/dt = Growth - Decay - Washout`
        3.  **Dissolved Oxygen (DO):** `dDO/dt = Aeration - Consumption by Biomass`

        *Disclaimer: This is a simplified educational model. Real-world systems are more complex and professional designs require comprehensive software like GPS-X, BioWin, or WEST.*
    """)