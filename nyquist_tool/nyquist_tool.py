import io
import numpy as np
import streamlit as st

try:
    import control as ctl
except Exception as e:
    st.error("The 'control' library is required. Install with: pip install control")
    raise

try:
    import sympy as sp
except Exception as e:
    st.error("The 'sympy' library is required. Install with: pip install sympy")
    raise

from dataclasses import dataclass
import matplotlib.pyplot as plt

st.set_page_config(page_title="Nyquist Plot Tool", layout="wide")

st.title("Nyquist Plot Tool")
st.caption("Nyquist plots, gain/phase margins, stability analysis, time delays, and compensators.")

# ---------- Helpers ----------
@dataclass
class StabilityInfo:
    """Stores stability analysis results"""
    gm: float  # gain margin (dB)
    pm: float  # phase margin (deg)
    gm_freq: float  # gain crossover frequency
    pm_freq: float  # phase crossover frequency
    stable: bool
    
def parse_native_tf(tf_string: str):
    """Parse native transfer function string like '(1 / (s^2+2*s+2))'"""
    try:
        # Define symbolic variable s
        s = sp.Symbol('s')
        
        # Parse the expression
        expr = sp.sympify(tf_string)
        
        # Get numerator and denominator
        num_expr, den_expr = expr.as_numer_denom()
        
        # Convert to polynomials
        # Use cancel to simplify and ensure proper polynomial form
        num_expr = sp.cancel(num_expr)
        den_expr = sp.cancel(den_expr)
        
        # Convert to Poly objects
        num_poly = sp.Poly(num_expr, s)
        den_poly = sp.Poly(den_expr, s)
        
        # Get coefficients (sympy returns lowest power first, need to reverse)
        num_coeffs = num_poly.all_coeffs()
        den_coeffs = den_poly.all_coeffs()
        
        # Convert to float and reverse to get highest power first (control library format)
        num_c = [float(c) for c in reversed(num_coeffs)]
        den_c = [float(c) for c in reversed(den_coeffs)]
        
        # Create transfer function
        sys = ctl.tf(num_c, den_c)
        return sys, None
    except Exception as e:
        return None, f"Failed to parse native transfer function: {e}"

def parse_tf(mode: str):
    """Parse transfer function from user input"""
    if mode == "Numerator / Denominator":
        num = st.text_input("Numerator coefficients (highest power first)", value="1")
        den = st.text_input("Denominator coefficients (highest power first)", value="1, 1, 1")
        try:
            num_c = [float(x) for x in num.replace(";",",").split(",") if x.strip()!=""]
            den_c = [float(x) for x in den.replace(";",",").split(",") if x.strip()!=""]
            sys = ctl.tf(num_c, den_c)
            return sys, None
        except Exception as e:
            return None, f"Failed to parse numerator/denominator: {e}"
    elif mode == "Zeros / Poles / Gain":
        z = st.text_input("Zeros (comma-separated)", value="")
        p = st.text_input("Poles (comma-separated)", value="-1, -2")
        k = st.number_input("Gain k", value=1.0, step=0.1, format="%.6g")
        try:
            zeros = [float(x) for x in z.replace(";",",").split(",") if x.strip()!=""] if z.strip() else []
            poles = [float(x) for x in p.replace(";",",").split(",") if x.strip()!=""]
            sys = ctl.zpk(zeros, poles, k)
            return sys, None
        except Exception as e:
            return None, f"Failed to parse zeros/poles/gain: {e}"
    elif mode == "Native Transfer Function":
        tf_string = st.text_input(
            "Transfer function (e.g., 1/(s^2+2*s+2) or (s+1)/(s^2+3*s+2))", 
            value="1/(s^2+2*s+2)",
            help="Enter transfer function using 's' as the variable. Examples: 1/(s^2+2*s+2), (s+1)/(s^2+3*s+2), s/(s+1)"
        )
        return parse_native_tf(tf_string)
    else:
        return None, "Unknown TF input mode"

def apply_compensator(sys):
    """Add optional compensator C(s)"""
    add_comp = st.checkbox("Add compensator C(s)", value=False)

    # reset if checkbox unselected
    if not add_comp:
        st.session_state.pop("pid_params", None)
        st.session_state.pop("last_ctype", None)
        return sys, ctl.tf([1], [1]), None

    # what compensator
    ctype = st.selectbox(
        "Compensator type",
        ["PID", "Lead", "Lag", "Lead-Lag", "Dynamic (num/den)"],
        index=0,
    )

    # check if type got changed
    last_ctype = st.session_state.get("last_ctype")
    if last_ctype != ctype:
        # reset all if type gets changed
        st.session_state.pop("pid_params", None)
    st.session_state["last_ctype"] = ctype
    
    if ctype == "PID":
        Kp = st.number_input("Kp", value=1.0, step=0.1, format="%.6g")
        Ki = st.number_input("Ki", value=0.0, step=0.1, format="%.6g")
        Kd = st.number_input("Kd", value=0.0, step=0.1, format="%.6g")
        
        if Ki == 0 and Kd == 0:
            # Pure proportional
            C = ctl.tf([Kp], [1])
        elif Ki == 0:
            # PD controller
            N = st.number_input("Derivative filter N", value=10.0, step=1.0, format="%.6g")
            C = ctl.tf([Kd*N, Kp*N], [1, N])
        else:
            # Full PID
            N = st.number_input("Derivative filter N", value=10.0, step=1.0, format="%.6g")
            C = ctl.tf([Kd*N, Kp*N, Ki*N], [1, N, 0])
        
        st.session_state["pid_params"] = {"Kp": Kp, "Ki": Ki, "Kd": Kd}
        return C*sys, C, None
    
    if ctype in ["Lead", "Lag", "Lead-Lag"]:
        if ctype == "Lead":
            z = st.number_input("Lead zero (rad/s)", value=1.0, step=0.1, format="%.6g")
            p = st.number_input("Lead pole (rad/s)", value=10.0, step=0.1, format="%.6g")
            k = st.number_input("Lead gain", value=1.0, step=0.1, format="%.6g")
            C = k * ctl.tf([1/z, 1], [1/p, 1])
        elif ctype == "Lag":
            z = st.number_input("Lag zero (rad/s)", value=0.1, step=0.1, format="%.6g")
            p = st.number_input("Lag pole (rad/s)", value=0.01, step=0.01, format="%.6g")
            k = st.number_input("Lag gain", value=1.0, step=0.1, format="%.6g")
            C = k * ctl.tf([1/z, 1], [1/p, 1])
        else:  # Lead-Lag
            z1 = st.number_input("Lead zero (rad/s)", value=1.0, key="llz1", step=0.1, format="%.6g")
            p1 = st.number_input("Lead pole (rad/s)", value=10.0, key="llp1", step=0.1, format="%.6g")
            z2 = st.number_input("Lag  zero (rad/s)", value=0.1, key="llz2", step=0.1, format="%.6g")
            p2 = st.number_input("Lag  pole (rad/s)", value=0.01, key="llp2", step=0.01, format="%.6g")
            k = st.number_input("Lead-Lag gain", value=1.0, step=0.1, format="%.6g")
            Lead = ctl.tf([1/z1, 1], [1/p1, 1])
            Lag  = ctl.tf([1/z2, 1], [1/p2, 1])
            C = k * Lead * Lag
        return C*sys, C, None
    
    if ctype == "Dynamic (num/den)":
        num_c = st.text_input("Compensator numerator", value="1, 2")
        den_c = st.text_input("Compensator denominator", value="1, 1")
        try:
            num_cc = [float(x) for x in num_c.replace(";",",").split(",") if x.strip()!=""]
            den_cc = [float(x) for x in den_c.replace(";",",").split(",") if x.strip()!=""]
            C = ctl.tf(num_cc, den_cc)
            return C*sys, C, None
        except Exception as e:
            return sys, ctl.tf([1],[1]), f"Failed to parse compensator: {e}"
    
    return sys, ctl.tf([1],[1]), None

def apply_time_delay(sys, tau):
    """Apply time delay using Pade approximation"""
    if tau <= 0:
        return sys
    
    # Use Pade approximation for time delay
    order = st.number_input("Pade approximation order", min_value=1, max_value=10, value=3, step=1)
    
    try:
        # Create Pade approximation for e^(-tau*s)
        num_pade, den_pade = ctl.pade(tau, order)
        delay_sys = ctl.tf(num_pade, den_pade)
        return sys * delay_sys
    except Exception as e:
        st.error(f"Error applying time delay: {e}")
        return sys

def compute_nyquist_data(sys, omega_range):
    """Compute Nyquist plot data"""
    try:
        # Compute frequency response
        response = ctl.frequency_response(sys, omega_range)
        mag = np.abs(response.fresp[0, 0, :])
        phase = np.angle(response.fresp[0, 0, :])
        
        # Convert to real and imaginary parts
        real = mag * np.cos(phase)
        imag = mag * np.sin(phase)
        
        return real, imag, mag, phase
    except Exception as e:
        st.error(f"Error computing Nyquist data: {e}")
        return None, None, None, None

def compute_stability_margins(sys):
    """Compute gain and phase margins"""
    try:
        gm, pm, wpc, wgc = ctl.margin(sys)
        
        # Convert to dB
        gm_db = 20 * np.log10(gm) if gm > 0 else np.inf
        
        # Check stability using Nyquist criterion
        # Count poles in RHP
        poles = ctl.pole(sys)
        rhp_poles = np.sum(np.real(poles) > 0)
        
        # Simple stability check: positive margins and no RHP poles
        stable = (gm > 1) and (pm > 0) and (rhp_poles == 0)
        
        return StabilityInfo(
            gm=gm_db,
            pm=pm,
            gm_freq=wgc,
            pm_freq=wpc,
            stable=stable
        )
    except Exception as e:
        st.warning(f"Could not compute margins: {e}")
        return None

def plot_nyquist(sys, omega_range, show_unity_circle=True, show_margins=True):
    """Create Nyquist plot"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute Nyquist data
    real, imag, mag, phase = compute_nyquist_data(sys, omega_range)
    
    if real is None:
        return None
    
    # Plot Nyquist curve for positive frequencies
    ax.plot(real, imag, 'b-', linewidth=2, label='L(jω) for ω > 0')
    
    # Plot Nyquist curve for negative frequencies (mirror)
    ax.plot(real, -imag, 'b--', linewidth=1.5, alpha=0.6, label='L(jω) for ω < 0')
    
    # Mark start and end points
    ax.plot(real[0], imag[0], 'go', markersize=10, label=f'ω = {omega_range[0]:.3f} rad/s')
    ax.plot(real[-1], imag[-1], 'rs', markersize=10, label=f'ω = {omega_range[-1]:.3f} rad/s')
    
    # Plot critical point (-1, 0)
    ax.plot(-1, 0, 'rx', markersize=15, markeredgewidth=3, label='Critical point (-1, 0)')
    
    # Plot unity circle
    if show_unity_circle:
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1, label='Unit circle')
    
    # Add gain and phase margins visualization
    if show_margins:
        stability = compute_stability_margins(sys)
        if stability:
            # Find phase crossover (where phase = -180°)
            phase_180_idx = np.argmin(np.abs(phase + np.pi))
            if mag[phase_180_idx] > 0.01:  # Valid crossover
                ax.plot([0, real[phase_180_idx]], [0, imag[phase_180_idx]], 
                       'g--', linewidth=2, alpha=0.5)
                ax.plot(real[phase_180_idx], imag[phase_180_idx], 
                       'go', markersize=8)
                ax.text(real[phase_180_idx], imag[phase_180_idx], 
                       f'  PM point\n  ω={stability.pm_freq:.2f}', 
                       fontsize=9, color='green')
            
            # Find gain crossover (where |L(jω)| = 1)
            gain_1_idx = np.argmin(np.abs(mag - 1))
            if abs(mag[gain_1_idx] - 1) < 0.1:  # Valid crossover
                ax.plot([0, real[gain_1_idx]], [0, imag[gain_1_idx]], 
                       'm--', linewidth=2, alpha=0.5)
                ax.plot(real[gain_1_idx], imag[gain_1_idx], 
                       'mo', markersize=8)
                ax.text(real[gain_1_idx], imag[gain_1_idx], 
                       f'  GM point\n  ω={stability.gm_freq:.2f}', 
                       fontsize=9, color='magenta')
    
    # Grid and labels
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='k', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Real Axis', fontsize=12)
    ax.set_ylabel('Imaginary Axis', fontsize=12)
    ax.set_title('Nyquist Plot', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.set_aspect('equal', adjustable='datalim')
    
    plt.tight_layout()
    return fig

# ---------- Main App ----------
st.sidebar.header("Plant Transfer Function P(s)")

# Input mode
input_mode = st.sidebar.radio(
    "Input format",
    ["Native Transfer Function", "Numerator / Denominator", "Zeros / Poles / Gain"],
    index=0
)

with st.sidebar:
    plant_sys, err = parse_tf(input_mode)
    
    if err:
        st.error(err)
        st.stop()
    
    if plant_sys is None:
        st.error("Could not create transfer function")
        st.stop()

st.sidebar.header("Controller C(s)")

with st.sidebar:
    loop_sys, comp_sys, comp_err = apply_compensator(plant_sys)
    
    if comp_err:
        st.warning(comp_err)

st.sidebar.header("Time Delay")

with st.sidebar:
    add_delay = st.checkbox("Add time delay e^(-τs)", value=False)
    if add_delay:
        tau = st.number_input("Time delay τ (seconds)", value=0.1, min_value=0.0, step=0.01, format="%.6g")
        loop_sys = apply_time_delay(loop_sys, tau)
    else:
        tau = 0.0

st.sidebar.header("Nyquist Plot Settings")

with st.sidebar:
    omega_min = st.number_input("Min frequency (rad/s)", value=0.001, min_value=1e-6, step=0.001, format="%.6g")
    omega_max = st.number_input("Max frequency (rad/s)", value=100.0, min_value=0.1, step=1.0, format="%.6g")
    n_points = st.number_input("Number of points", value=1000, min_value=100, max_value=10000, step=100)
    
    log_scale = st.checkbox("Logarithmic frequency scale", value=True)
    
    if log_scale:
        omega_range = np.logspace(np.log10(omega_min), np.log10(omega_max), n_points)
    else:
        omega_range = np.linspace(omega_min, omega_max, n_points)
    
    show_unity = st.checkbox("Show unit circle", value=True)
    show_margins = st.checkbox("Show gain/phase margins", value=True)

# ---------- Display Results ----------

# Display transfer functions
col1, col2 = st.columns(2)

with col1:
    st.subheader("Plant P(s)")
    st.code(str(plant_sys), language=None)

with col2:
    if comp_sys.num[0][0][0] != 1 or len(comp_sys.num[0][0]) > 1 or len(comp_sys.den[0][0]) > 1:
        st.subheader("Controller C(s)")
        st.code(str(comp_sys), language=None)

st.subheader("Open-Loop Transfer Function L(s) = C(s)·P(s)")
st.code(str(loop_sys), language=None)

if add_delay:
    st.info(f"Time delay τ = {tau} s is included using Pade approximation")

# Compute stability margins
stability = compute_stability_margins(loop_sys)

if stability:
    st.subheader("Stability Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if np.isinf(stability.gm):
            st.metric("Gain Margin", "∞ dB")
        else:
            st.metric("Gain Margin", f"{stability.gm:.2f} dB")
    
    with col2:
        st.metric("Phase Margin", f"{stability.pm:.2f}°")
    
    with col3:
        st.metric("GM Frequency", f"{stability.gm_freq:.3f} rad/s")
    
    with col4:
        st.metric("PM Frequency", f"{stability.pm_freq:.3f} rad/s")
    
    # Stability indicator
    if stability.stable:
        st.success("✓ System appears to be stable (positive margins, no RHP poles)")
    else:
        st.error("✗ System may be unstable (check Nyquist encirclements)")

# Plot Nyquist diagram
st.subheader("Nyquist Diagram")

fig = plot_nyquist(loop_sys, omega_range, show_unity_circle=show_unity, show_margins=show_margins)

if fig:
    st.pyplot(fig)
    plt.close(fig)
else:
    st.error("Could not generate Nyquist plot")

# Additional information
with st.expander("ℹ️ How to interpret the Nyquist plot"):
    st.markdown("""
    ### Nyquist Stability Criterion
    
    The Nyquist plot shows the open-loop frequency response L(jω) = C(jω)·P(jω) in the complex plane.
    
    **Key Points:**
    - **Critical Point (-1, 0)**: The system is stable if the Nyquist curve does not encircle this point
    - **Gain Margin (GM)**: Distance from the curve to (-1, 0) along the negative real axis
    - **Phase Margin (PM)**: Angular distance from the curve to (-1, 0) at unity gain
    
    **Stability Rules:**
    - For systems with no RHP poles: No encirclements of (-1, 0) → Stable
    - For systems with N RHP poles: N counter-clockwise encirclements → Stable
    - Positive GM and PM generally indicate stability
    
    **The Plot Shows:**
    - Blue solid line: L(jω) for positive frequencies
    - Blue dashed line: L(jω) for negative frequencies (mirror image)
    - Red X: Critical point (-1, 0)
    - Green circle: Low frequency start point
    - Red square: High frequency end point
    """)

with st.expander("📝 Tips"):
    st.markdown("""
    - Adjust the frequency range to see more detail in specific regions
    - Use logarithmic scale for a wider frequency range
    - Enable margins visualization to see GM and PM graphically
    - The unit circle helps identify the unity gain crossover
    - Add compensators to improve stability margins
    - Time delay rotates the phase, which can destabilize the system
    """)

