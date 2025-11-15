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

st.set_page_config(page_title="Nyquist Plot Tool", layout="wide")

st.title("Nyquist Plot Tool")
st.caption("Nyquist plots, stability analysis, time delays, and compensators.")

# Instructions box
with st.expander("📖 Instructions", expanded=False):
    st.markdown("""
    **How to use this tool:**
    
    1. **Enter Plant Transfer Function**: In the sidebar, select an input format and enter your plant transfer function P(s).
       - **Transfer Function (s-form)**: Enter directly using 's' (e.g., `1/(s^2+2*s+2)`)
       - **Numerator / Denominator**: Enter coefficients as comma-separated values
       - **Zeros / Poles / Gain**: Specify zeros, poles, and gain
    
    2. **Add Compensator (Optional)**: Check "Add compensator C(s)" to include a controller.
       - Choose from PID, Lead, Lag, Lead-Lag, or Custom (s-form)
    
    3. **Add Time Delay (Optional)**: Check "Add time delay e^(-τs)" to include time delay.
       - Choose between Padé approximation or exact delay (frequency-domain only)
    
    4. **Adjust Settings**: Configure the number of points for the Nyquist plot and toggle the unit circle display.
    
    5. **View Results**: The resulting transfer function L(s) = C(s)·P(s) is displayed, and the interactive Nyquist plot shows the frequency response in the complex plane.
    
    **Tips:**
    - Use the interactive plot to zoom and pan for detailed analysis
    - The critical point (-1, 0) is marked in red
    - The unit circle helps identify unity gain crossover
    - Hover over the plot to see exact coordinates
    """)

# ---------- Helpers ----------
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
        
        # Expand to ensure all terms are explicit
        num_expr = sp.expand(num_expr)
        den_expr = sp.expand(den_expr)
        
        # Convert to Poly objects - this will handle missing terms correctly
        num_poly = sp.Poly(num_expr, s)
        den_poly = sp.Poly(den_expr, s)
        
        # Get the degree to ensure we extract all coefficients
        num_degree = num_poly.degree()
        den_degree = den_poly.degree()
        
        # Extract coefficients explicitly for each power
        # Build coefficient lists from highest to lowest power (control library format)
        num_c = []
        for i in range(num_degree, -1, -1):
            coeff = num_poly.coeff_monomial(s**i)
            num_c.append(float(coeff))
        
        den_c = []
        for i in range(den_degree, -1, -1):
            coeff = den_poly.coeff_monomial(s**i)
            den_c.append(float(coeff))
        
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

    # reset if checkbox unselected
    if not add_comp:
        st.session_state.pop("pid_params", None)
        st.session_state.pop("last_ctype", None)
        return sys, ctl.tf([1], [1]), None

    # what compensator
    ctype = st.selectbox(
        "Compensator type",
        ["PID", "Lead", "Lag", "Lead-Lag", "Custom"],
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
            N = None
        elif Ki == 0:
            # PD controller
            N = st.number_input("Derivative filter N", value=10.0, step=1.0, format="%.6g")
            C = ctl.tf([Kd*N, Kp*N], [1, N])
        else:
            # Full PID
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
    
    # Load MathJax for LaTeX rendering in radio labels
    st.markdown(
        r"<script type='text/javascript' src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'></script>",
        unsafe_allow_html=True
    )
    
    # Use LaTeX directly in radio labels
    method = st.radio(
        "Delay model",
        [
            "Padé approximation",
            "Exact $e^{-s\\tau}$ (frequency-domain only)"
        ],
        index=0,
        horizontal=True,
    )
    
    # Handle the two delay types
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

def compute_nyquist_data(sys, omega_range, delay_info=None):
    """Compute Nyquist plot data with optional exact time delay"""
    try:
        # Compute frequency response
        response = ctl.frequency_response(sys, omega_range)
        mag = np.abs(response.fresp[0, 0, :])
        phase = np.angle(response.fresp[0, 0, :])
        
        # Apply exact time delay in frequency domain if specified
        if delay_info is not None and delay_info[0] == "exact":
            tau = delay_info[1]
            # Exact delay: multiply by e^(-j*omega*tau) = cos(-omega*tau) + j*sin(-omega*tau)
            # This rotates the phase by -omega*tau
            phase = phase - omega_range * tau
        
        # Convert to real and imaginary parts
        real = mag * np.cos(phase)
        imag = mag * np.sin(phase)
        
        return real, imag, mag, phase
    except Exception as e:
        st.error(f"Error computing Nyquist data: {e}")
        return None, None, None, None

def plot_nyquist(sys, omega_range, show_unity_circle=True, delay_info=None):
    """Create interactive Nyquist plot using Plotly"""
    # Compute Nyquist data
    real, imag, mag, phase = compute_nyquist_data(sys, omega_range, delay_info)
    
    if real is None:
        return None
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Plot Nyquist curve for positive frequencies
    fig.add_trace(go.Scatter(
        x=real,
        y=imag,
        mode='lines',
        name='L(jω) for ω > 0',
        line=dict(color='blue', width=2),
        hovertemplate='Real: %{x:.4f}<br>Imag: %{y:.4f}<extra></extra>'
    ))
    
    # Plot Nyquist curve for negative frequencies (mirror)
    fig.add_trace(go.Scatter(
        x=real,
        y=-imag,
        mode='lines',
        name='L(jω) for ω < 0',
        line=dict(color='blue', width=1.5, dash='dash'),
        opacity=0.6,
        hovertemplate='Real: %{x:.4f}<br>Imag: %{y:.4f}<extra></extra>'
    ))
    
    # Plot critical point (-1, 0)
    fig.add_trace(go.Scatter(
        x=[-1],
        y=[0],
        mode='markers',
        name='Critical point (-1, 0)',
        marker=dict(symbol='x', size=15, color='red', line=dict(width=2, color='red')),
        hovertemplate='Critical point (-1, 0)<extra></extra>'
    ))
    
    # Plot unity circle
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
    
    # Update layout
    fig.update_layout(
        title='Nyquist Plot',
        xaxis_title='Real Axis',
        yaxis_title='Imaginary Axis',
        hovermode='closest',
        width=700,
        height=600,
        showlegend=True,
        plot_bgcolor='white',
        # paper_bgcolor='white',
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

# ---------- Main App ----------
st.sidebar.header("Plant Transfer Function P(s)")

# Input mode
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

st.sidebar.header("Nyquist Plot Settings")

with st.sidebar:
    n_points = st.number_input("Number of points", value=1000, min_value=100, max_value=10000, step=100)
    
    # Use default frequency range with logarithmic scale
    omega_min = 0.001
    omega_max = 100.0
    omega_range = np.logspace(np.log10(omega_min), np.log10(omega_max), n_points)
    
    show_unity = st.checkbox("Show unit circle", value=True)

# ---------- Display Results ----------

# Load MathJax for LaTeX rendering
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

# Check if compensator is active
comp_active = comp_sys is not None and not (
    len(ctl.tfdata(comp_sys)[0][0][0]) == 1 and ctl.tfdata(comp_sys)[0][0][0][0] == 1
)

# No compensator
if not comp_active:
    if not add_delay:
        st.latex(r"L(s) = P(s) = " + tf_to_latex(plant_sys))
    else:
        if delay_info is not None and delay_info[0] == "exact":
            # Exact delay
            st.latex(
                rf"""\begin{{aligned}}
                L(s) &= P(s)\,e^{{-s\tau}},\quad \tau={tau:.3g}\\[6pt]
                L(s) &= {tf_to_latex(plant_sys)}\,e^{{-{tau:.3g}s}}
                \end{{aligned}}"""
            )
        elif delay_info is not None:
            # With Pade delay
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

# With compensator
else:
    L = comp_sys * plant_sys
    pid = st.session_state.get("pid_params", None)
    
    # Special format for PID
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
                # Exact delay
                st.latex(
                    rf"""\begin{{aligned}}
                    L(s) &= C(s)\,P(s)\,e^{{-s\tau}},\quad \tau={tau:.3g}\\[8pt]
                    L(s) &= {tf_to_latex(L)}\,e^{{-{tau:.3g}s}}
                    \end{{aligned}}"""
                )
            elif delay_info is not None:
                # With Pade delay
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
        # Compensator without PID
        if not add_delay:
            st.latex(
                rf"""\begin{{aligned}}
                L(s) &= C(s)\,P(s) = {tf_to_latex(comp_sys)}\cdot{tf_to_latex(plant_sys)}\\[10pt]
                L(s) &= {tf_to_latex(L)}
                \end{{aligned}}"""
            )
        else:
            if delay_info is not None and delay_info[0] == "exact":
                # Exact delay
                st.latex(
                    rf"""\begin{{aligned}}
                    L(s) &= C(s)\,P(s)\,e^{{-s\tau}},\quad \tau={tau:.3g}\\[8pt]
                    L(s) &= {tf_to_latex(L)}\,e^{{-{tau:.3g}s}}
                    \end{{aligned}}"""
                )
            elif delay_info is not None:
                # With Pade delay
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

# Plot Nyquist diagram
st.subheader("Nyquist Diagram")

fig = plot_nyquist(loop_sys, omega_range, show_unity_circle=show_unity, delay_info=delay_info)

if fig:
    # Use a column to control the plot width on the webpage
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Could not generate Nyquist plot")

# Additional information
with st.expander("ℹ️ How to interpret the Nyquist plot"):
    st.markdown("""
    ### Nyquist Stability Criterion
    
    The Nyquist plot shows the open-loop frequency response L(jω) = C(jω)·P(jω) in the complex plane.
    
    **Key Points:**
    - **Critical Point (-1, 0)**: The system is stable if the Nyquist curve does not encircle this point
    
    **Stability Rules:**
    - For systems with no RHP poles: No encirclements of (-1, 0) → Stable
    - For systems with N RHP poles: N counter-clockwise encirclements → Stable
    
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
    - The unit circle helps identify the unity gain crossover
    - Add compensators to improve stability
    - Time delay rotates the phase, which can destabilize the system
    """)

