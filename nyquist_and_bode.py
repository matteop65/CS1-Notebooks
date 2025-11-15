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

try:
    import plotly.graph_objects as go
except Exception as e:
    st.error("The 'plotly' library is required. Install with: pip install plotly")
    raise

import matplotlib.pyplot as plt

st.set_page_config(page_title="Nyquist & Bode Plot Tool", layout="wide")

st.title("Nyquist & Bode Plot Tool")
st.caption("Combined Nyquist and Bode plots for frequency domain analysis.")

# ---------- Helpers ----------
def parse_native_tf(tf_string: str):
    """Parse native transfer function string like '(1 / (s^2+2*s+2))'"""
    try:
        s = sp.Symbol('s')
        expr = sp.sympify(tf_string)
        num_expr, den_expr = expr.as_numer_denom()
        num_expr = sp.cancel(num_expr)
        den_expr = sp.cancel(den_expr)
        num_expr = sp.expand(num_expr)
        den_expr = sp.expand(den_expr)
        num_poly = sp.Poly(num_expr, s)
        den_poly = sp.Poly(den_expr, s)
        num_degree = num_poly.degree()
        den_degree = den_poly.degree()
        num_c = []
        for i in range(num_degree, -1, -1):
            coeff = num_poly.coeff_monomial(s**i)
            num_c.append(float(coeff))
        den_c = []
        for i in range(den_degree, -1, -1):
            coeff = den_poly.coeff_monomial(s**i)
            den_c.append(float(coeff))
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
    elif mode == "Transfer Function (s-form)":
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
    if not add_comp:
        st.session_state.pop("pid_params", None)
        st.session_state.pop("last_ctype", None)
        return sys, ctl.tf([1], [1]), None

    ctype = st.selectbox(
        "Compensator type",
        ["PID", "Lead", "Lag", "Lead-Lag", "Custom"],
        index=0,
    )

    last_ctype = st.session_state.get("last_ctype")
    if last_ctype != ctype:
        st.session_state.pop("pid_params", None)
    st.session_state["last_ctype"] = ctype
    
    if ctype == "PID":
        Kp = st.number_input("Kp", value=1.0, step=0.1, format="%.6g")
        Ki = st.number_input("Ki", value=0.0, step=0.1, format="%.6g")
        Kd = st.number_input("Kd", value=0.0, step=0.1, format="%.6g")
        
        if Ki == 0 and Kd == 0:
            C = ctl.tf([Kp], [1])
            N = None
        elif Ki == 0:
            N = st.number_input("Derivative filter N", value=10.0, step=1.0, format="%.6g")
            C = ctl.tf([Kd*N, Kp*N], [1, N])
        else:
            N = st.number_input("Derivative filter N", value=10.0, step=1.0, format="%.6g")
            C = ctl.tf([Kd*N, Kp*N, Ki*N], [1, N, 0])
        
        st.session_state["pid_params"] = {"Kp": Kp, "Ki": Ki, "Kd": Kd}
        if N is not None:
            st.session_state["pid_N"] = N
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
    
    if ctype == "Custom":
        tf_string = st.text_input(
            "Compensator transfer function (e.g., (s+1)/(s+2) or 1/(s+1))", 
            value="1",
            help="Enter compensator transfer function using 's' as the variable. Examples: (s+1)/(s+2), 1/(s+1), s/(s^2+2*s+1)"
        )
        C, err = parse_native_tf(tf_string)
        if err:
            return sys, ctl.tf([1],[1]), f"Failed to parse compensator: {err}"
        if C is None:
            return sys, ctl.tf([1],[1]), "Could not create compensator transfer function"
        return C*sys, C, None
    
    return sys, ctl.tf([1],[1]), None

def apply_time_delay(sys, tau):
    """Apply time delay using Pade approximation or exact delay"""
    if tau <= 0:
        return sys, None, None
    
    st.markdown(
        r"<script type='text/javascript' src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'></script>",
        unsafe_allow_html=True
    )
    
    method = st.radio(
        "Delay model",
        [
            "Padé approximation",
            "Exact $e^{-s\\tau}$ (frequency-domain only)"
        ],
        index=0,
        horizontal=True,
    )
    
    if method.startswith("Padé"):
        order = st.number_input("Padé order", min_value=1, max_value=10, value=3, step=1)
        st.session_state["pade_order"] = order
        try:
            num_d, den_d = ctl.pade(tau, order)
            D = ctl.tf(num_d, den_d)
            return sys * D, ("pade", tau, order, D), None
        except Exception as e:
            st.error(f"Error applying time delay: {e}")
            return sys, None, None
    else:
        return sys, ("exact", tau, None, None), None

def bode_np(sys, w):
    """Compute Bode plot data"""
    mag, phase, omega = ctl.freqresp(sys, w)
    mag = np.squeeze(mag)
    phase = np.squeeze(phase) * 180/np.pi
    return mag, phase

def compute_nyquist_data(sys, omega_range, delay_info=None):
    """Compute Nyquist plot data with optional exact time delay"""
    try:
        response = ctl.frequency_response(sys, omega_range)
        mag = np.abs(response.fresp[0, 0, :])
        phase = np.angle(response.fresp[0, 0, :])
        
        if delay_info is not None and delay_info[0] == "exact":
            tau = delay_info[1]
            phase = phase - omega_range * tau
        
        real = mag * np.cos(phase)
        imag = mag * np.sin(phase)
        
        return real, imag, mag, phase
    except Exception as e:
        st.error(f"Error computing Nyquist data: {e}")
        return None, None, None, None

def plot_nyquist(sys, omega_range, show_unity_circle=True, delay_info=None):
    """Create interactive Nyquist plot using Plotly"""
    real, imag, mag, phase = compute_nyquist_data(sys, omega_range, delay_info)
    
    if real is None:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=real,
        y=imag,
        mode='lines',
        name='L(jω) for ω > 0',
        line=dict(color='blue', width=2),
        hovertemplate='Real: %{x:.4f}<br>Imag: %{y:.4f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=real,
        y=-imag,
        mode='lines',
        name='L(jω) for ω < 0',
        line=dict(color='blue', width=1.5, dash='dash'),
        opacity=0.6,
        hovertemplate='Real: %{x:.4f}<br>Imag: %{y:.4f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[-1],
        y=[0],
        mode='markers',
        name='Critical point (-1, 0)',
        marker=dict(symbol='x', size=15, color='red', line=dict(width=2, color='red')),
        hovertemplate='Critical point (-1, 0)<extra></extra>'
    ))
    
    if show_unity_circle:
        theta = np.linspace(0, 2*np.pi, 200)
        fig.add_trace(go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode='lines',
            name='Unit circle',
            line=dict(color='gray', width=1, dash='dash'),
            opacity=0.3,
            hovertemplate='Unit circle<extra></extra>'
        ))
    
    fig.update_layout(
        title='Nyquist Plot',
        xaxis_title='Real Axis',
        yaxis_title='Imaginary Axis',
        hovermode='closest',
        width=600,
        height=600,
        showlegend=True,
        plot_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='white',
            bordercolor='gray',
            borderwidth=1,
            font=dict(color='black')
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=0.8
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=0.8
        )
    )
    
    return fig

def plot_bode(sys, omega_range, delay_info=None):
    """Create Bode plots (magnitude and phase)"""
    mag, phase = bode_np(sys, omega_range)
    
    if delay_info is not None and delay_info[0] == "exact":
        tau = delay_info[1]
        phase = phase - omega_range * tau * 180/np.pi
    
    mag_db = 20 * np.log10(mag)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    
    # Magnitude plot
    ax1.semilogx(omega_range, mag_db, 'b-', linewidth=2)
    ax1.axhline(0, color="black", lw=0.8, ls="-", alpha=0.7)
    ax1.set_ylabel("Magnitude (dB)", fontsize=12)
    ax1.set_xlabel("ω (rad/s)", fontsize=12)
    ax1.grid(True, which="both", ls=":")
    ax1.set_title("Bode Plot - Magnitude", fontsize=14, fontweight='bold')
    
    # Phase plot
    ax2.semilogx(omega_range, phase, 'b-', linewidth=2)
    ax2.axhline(-180, color="black", lw=0.8, ls="-", alpha=0.7)
    ax2.set_ylabel("Phase (deg)", fontsize=12)
    ax2.set_xlabel("ω (rad/s)", fontsize=12)
    ax2.grid(True, which="both", ls=":")
    ax2.set_title("Bode Plot - Phase", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

# ---------- Main App ----------
st.sidebar.header("Plant Transfer Function P(s)")

input_mode = st.sidebar.radio(
    "Input format",
    ["Transfer Function (s-form)", "Numerator / Denominator", "Zeros / Poles / Gain"],
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
        loop_sys, delay_info, delay_err = apply_time_delay(loop_sys, tau)
        if delay_err:
            st.warning(delay_err)
    else:
        tau = 0.0
        delay_info = None

st.sidebar.header("Plot Settings")

with st.sidebar:
    n_points = st.number_input("Number of points", value=1000, min_value=100, max_value=10000, step=100)
    
    omega_min = 0.001
    omega_max = 100.0
    omega_range = np.logspace(np.log10(omega_min), np.log10(omega_max), n_points)
    
    show_unity = st.checkbox("Show unit circle", value=True)

# ---------- Display Results ----------

st.markdown(
    r"<script type='text/javascript' src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'></script>",
    unsafe_allow_html=True
)

# Helper functions for LaTeX formatting
def clean_num(x):
    if abs(x - round(x)) < 1e-10:
        return str(int(round(x)))
    else:
        return f"{x:.4g}"

def poly_to_latex(coeffs):
    terms = []
    n = len(coeffs)
    for i, c in enumerate(coeffs):
        power = n - i - 1
        if abs(c) < 1e-12:
            continue
        c_str = clean_num(c)
        if power == 0:
            terms.append(f"{c_str}")
        elif power == 1:
            terms.append(f"{c_str} s")
        else:
            terms.append(f"{c_str} s^{{{power}}}")
    return " + ".join(terms) if terms else "0"

def tf_to_latex(tf_obj):
    num, den = ctl.tfdata(tf_obj)
    num = [float(n) for n in num[0][0]]
    den = [float(d) for d in den[0][0]]
    num_str = poly_to_latex(num)
    den_str = poly_to_latex(den)
    return r"\frac{" + num_str + "}{" + den_str + "}"

# Display resulting transfer function
st.subheader("Resulting Transfer Function")

comp_active = comp_sys is not None and not (
    len(ctl.tfdata(comp_sys)[0][0][0]) == 1 and ctl.tfdata(comp_sys)[0][0][0][0] == 1
)

if not comp_active:
    if not add_delay:
        st.latex(r"L(s) = P(s) = " + tf_to_latex(plant_sys))
    else:
        if delay_info is not None and delay_info[0] == "exact":
            st.latex(
                rf"""\begin{{aligned}}
                L(s) &= P(s)\,e^{{-s\tau}},\quad \tau={tau:.3g}\\[6pt]
                L(s) &= {tf_to_latex(plant_sys)}\,e^{{-{tau:.3g}s}}
                \end{{aligned}}"""
            )
        elif delay_info is not None:
            order = delay_info[2]
            D = delay_info[3]
            num_d, den_d = ctl.pade(tau, order)
            num_str = poly_to_latex(num_d)
            den_str = poly_to_latex(den_d)
            pade_formula = rf"e^{{-s\tau}} \approx \frac{{{num_str}}}{{{den_str}}},\quad n={order},\;\tau={tau:.3g}"
            st.latex(
                rf"""\begin{{aligned}}
                &{pade_formula}\\[6pt]
                L(s) &= P(s)\,D_{{\text{{Padé}}}}(s)
                = \big({tf_to_latex(plant_sys)}\big)\,\big({tf_to_latex(D)}\big)\\[8pt]
                L(s) &= {tf_to_latex(loop_sys)}
                \end{{aligned}}"""
            )
else:
    L = comp_sys * plant_sys
    pid = st.session_state.get("pid_params", None)
    
    if pid is not None:
        def cn(x):
            return str(int(round(x))) if abs(x - round(x)) < 1e-10 else f"{x:.4g}"
        Kp, Ki, Kd = pid["Kp"], pid["Ki"], pid["Kd"]
        
        terms = []
        if abs(Kp) > 1e-12:
            terms.append(cn(Kp))
        if abs(Ki) > 1e-12:
            terms.append(rf"\frac{{{cn(Ki)}}}{{s}}")
        if abs(Kd) > 1e-12:
            N = st.session_state.get("pid_N", 10.0)
            terms.append(rf"\frac{{{cn(Kd)}\,\cdot {cn(N)}\,s}}{{s+{cn(N)}}}")
        if not terms:
            terms = ["0"]
        pid_expr = " + ".join(terms)
        
        if not add_delay:
            st.latex(
                rf"""\begin{{aligned}}
                L(s) &= C(s)\,P(s) = \big({pid_expr}\big)\cdot{tf_to_latex(plant_sys)}\\[10pt]
                L(s) &= {tf_to_latex(L)}
                \end{{aligned}}"""
            )
        else:
            if delay_info is not None and delay_info[0] == "exact":
                st.latex(
                    rf"""\begin{{aligned}}
                    L(s) &= C(s)\,P(s)\,e^{{-s\tau}},\quad \tau={tau:.3g}\\[8pt]
                    L(s) &= {tf_to_latex(L)}\,e^{{-{tau:.3g}s}}
                    \end{{aligned}}"""
                )
            elif delay_info is not None:
                order = delay_info[2]
                D = delay_info[3]
                num_d, den_d = ctl.pade(tau, order)
                num_str = poly_to_latex(num_d)
                den_str = poly_to_latex(den_d)
                pade_formula = rf"e^{{-s\tau}} \approx \frac{{{num_str}}}{{{den_str}}},\quad n={order},\;\tau={tau:.3g}"
                st.latex(
                    rf"""\begin{{aligned}}
                    &{pade_formula}\\[6pt]
                    L(s) &= C(s)\,P(s)\,D_{{\text{{Padé}}}}(s)
                    = \big({pid_expr}\big)\,\big({tf_to_latex(plant_sys)}\big)\,\big({tf_to_latex(D)}\big)\\[10pt]
                    L(s) &= {tf_to_latex(loop_sys)}
                    \end{{aligned}}"""
                )
    else:
        if not add_delay:
            st.latex(
                rf"""\begin{{aligned}}
                L(s) &= C(s)\,P(s) = {tf_to_latex(comp_sys)}\cdot{tf_to_latex(plant_sys)}\\[10pt]
                L(s) &= {tf_to_latex(L)}
                \end{{aligned}}"""
            )
        else:
            if delay_info is not None and delay_info[0] == "exact":
                st.latex(
                    rf"""\begin{{aligned}}
                    L(s) &= C(s)\,P(s)\,e^{{-s\tau}},\quad \tau={tau:.3g}\\[8pt]
                    L(s) &= {tf_to_latex(L)}\,e^{{-{tau:.3g}s}}
                    \end{{aligned}}"""
                )
            elif delay_info is not None:
                order = delay_info[2]
                D = delay_info[3]
                num_d, den_d = ctl.pade(tau, order)
                num_str = poly_to_latex(num_d)
                den_str = poly_to_latex(den_d)
                pade_formula = rf"e^{{-s\tau}} \approx \frac{{{num_str}}}{{{den_str}}},\quad n={order},\;\tau={tau:.3g}"
                st.latex(
                    rf"""\begin{{aligned}}
                    &{pade_formula}\\[6pt]
                    L(s) &= C(s)\,P(s)\,D_{{\text{{Padé}}}}(s)
                    = \big({tf_to_latex(comp_sys)}\big)\,\big({tf_to_latex(plant_sys)}\big)\,\big({tf_to_latex(D)}\big)\\[10pt]
                    L(s) &= {tf_to_latex(loop_sys)}
                    \end{{aligned}}"""
                )

# ---------- Plots ----------
st.subheader("Frequency Domain Plots")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Bode Plot")
    bode_fig = plot_bode(loop_sys, omega_range, delay_info)
    if bode_fig:
        st.pyplot(bode_fig)
        plt.close(bode_fig)

with col2:
    st.markdown("### Nyquist Plot")
    nyquist_fig = plot_nyquist(loop_sys, omega_range, show_unity_circle=show_unity, delay_info=delay_info)
    if nyquist_fig:
        st.plotly_chart(nyquist_fig, width='stretch')
    else:
        st.error("Could not generate Nyquist plot")

