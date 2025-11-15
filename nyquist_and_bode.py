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
    from plotly.subplots import make_subplots
except Exception as e:
    st.error("The 'plotly' library is required. Install with: pip install plotly")
    raise

# matplotlib not needed for Plotly plots

from dataclasses import dataclass

@dataclass
class SpecConfig:
    show_specs: bool
    min_phase_margin_deg: float | None
    min_gain_margin_db: float | None
    target_bw_rad: float | None

st.set_page_config(page_title="Frequency Domain Plots", layout="wide")

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

def compute_margins_and_bw(L, w):
    """Compute gain margin, phase margin, and bandwidth"""
    try:
        gm, pm, wgc, wpc = ctl.margin(L)
    except Exception:
        gm = pm = wgc = wpc = np.nan
    T = ctl.feedback(L, 1)
    try:
        magT, phaseT = bode_np(T, w)
        magT_db = 20*np.log10(magT)
        idx = np.where(magT_db <= -3.0)[0]
        bw = w[idx[0]] if len(idx) > 0 else np.nan
    except Exception:
        bw = np.nan
    return gm, pm, wgc, wpc, bw

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

def plot_bode(sys, omega_range, delay_info=None, show_margins=True, specs=None, freq_bounds=None):
    """Create interactive Bode plots using Plotly"""
    mag, phase = bode_np(sys, omega_range)
    
    if delay_info is not None and delay_info[0] == "exact":
        tau = delay_info[1]
        phase = phase - omega_range * tau * 180/np.pi
    
    mag_db = 20 * np.log10(mag)
    
    # Generate logarithmic grid lines
    omega_min = omega_range[0]
    omega_max = omega_range[-1]
    
    # Major divisions (powers of 10)
    major_freqs = []
    exp_min = int(np.floor(np.log10(omega_min)))
    exp_max = int(np.ceil(np.log10(omega_max)))
    for exp in range(exp_min, exp_max + 1):
        freq = 10**exp
        if omega_min <= freq <= omega_max:
            major_freqs.append(freq)
    
    # Minor divisions (2, 3, 4, 5, 6, 7, 8, 9 times powers of 10)
    minor_freqs = []
    for exp in range(exp_min, exp_max + 1):
        for mult in [2, 3, 4, 5, 6, 7, 8, 9]:
            freq = mult * 10**exp
            if omega_min <= freq <= omega_max:
                minor_freqs.append(freq)
    
    # Compute margins and bandwidth
    gm, pm, wgc, wpc, bw = compute_margins_and_bw(sys, omega_range)
    
    if specs is None:
        specs = SpecConfig(False, None, None, None)
    if freq_bounds is None:
        freq_bounds = {"show": False}
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("", ""),  # Empty titles, we'll add custom ones
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5],
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # Magnitude plot
    fig.add_trace(
        go.Scatter(
            x=omega_range,
            y=mag_db,
            mode='lines',
            name='Magnitude',
            line=dict(color='blue', width=2),
            hovertemplate='ω: %{x:.4f} rad/s<br>Magnitude: %{y:.4f} dB<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 0 dB reference line (solid, prominent)
    fig.add_hline(
        y=0, line_dash="solid", line_color="black", 
        line_width=1.0, opacity=0.8, row=1, col=1
    )
    
    # Common magnitude reference lines (dashed, lighter)
    for db_line in [-20, -40, 20, 40]:
        fig.add_hline(
            y=db_line, line_dash="dash", line_color="gray", 
            line_width=0.5, opacity=0.4, row=1, col=1
        )
    
    # Logarithmic grid lines - major divisions (powers of 10)
    for freq in major_freqs:
        fig.add_vline(
            x=freq, line_dash="solid", line_color="lightgray", 
            line_width=0.8, opacity=0.6, row=1, col=1
        )
    
    # Logarithmic grid lines - minor divisions
    for freq in minor_freqs:
        fig.add_vline(
            x=freq, line_dash="dot", line_color="lightgray", 
            line_width=0.3, opacity=0.4, row=1, col=1
        )
    
    # Gain margin indicator
    if show_margins and np.isfinite(gm) and np.isfinite(wgc) and wgc > 0:
        try:
            mag_at_wgc = 20 * np.log10(np.abs(ctl.freqresp(sys, [wgc])[0][0]))
            gm_db = 20 * np.log10(gm)
            # Vertical line at gain crossover frequency
            fig.add_vline(
                x=wgc, line_dash="dot", line_color="blue", 
                line_width=1.0, opacity=0.8, row=1, col=1
            )
            # Line connecting to 0 dB
            fig.add_trace(
                go.Scatter(
                    x=[wgc, wgc],
                    y=[mag_at_wgc, 0],
                    mode='lines',
                    name=f'GM = {gm_db:.2f} dB',
                    line=dict(color='blue', width=1.2, dash='dot'),
                    showlegend=True,
                    hovertemplate=f'Gain Crossover: ω={wgc:.4f} rad/s<br>GM={gm_db:.2f} dB<extra></extra>'
                ),
                row=1, col=1
            )
            # Marker at gain crossover
            fig.add_trace(
                go.Scatter(
                    x=[wgc],
                    y=[mag_at_wgc],
                    mode='markers',
                    name='',
                    marker=dict(symbol='circle', size=8, color='blue'),
                    showlegend=False,
                    hovertemplate=f'Gain Crossover: ω={wgc:.4f} rad/s<br>GM={gm_db:.2f} dB<extra></extra>'
                ),
                row=1, col=1
            )
            # Text annotation for GM
            fig.add_annotation(
                x=wgc,
                y=(mag_at_wgc + 0) / 2,
                text=f"GM<br>{gm_db:.2f} dB",
                showarrow=False,
                font=dict(size=9, color="blue"),
                bgcolor="white",
                bordercolor="blue",
                borderwidth=1,
                row=1, col=1
            )
        except Exception:
            pass
    
    # Min Gain Margin specification
    if specs.show_specs and specs.min_gain_margin_db is not None:
        min_gm_db = specs.min_gain_margin_db
        fig.add_hline(
            y=min_gm_db, line_dash="dash", line_color="orange", 
            line_width=2, opacity=0.7, row=1, col=1
        )
        # Find where magnitude > min_gm_db and add green regions
        satisfied = mag_db > min_gm_db
        if np.any(satisfied):
            # Create segments where spec is satisfied
            for i in range(len(omega_range) - 1):
                if satisfied[i] and satisfied[i+1]:
                    fig.add_shape(
                        type="rect",
                        x0=omega_range[i], x1=omega_range[i+1],
                        y0=-200, y1=200,
                        fillcolor="green",
                        opacity=0.15,
                        layer="below",
                        line_width=0,
                        row=1, col=1
                    )
    
    # Frequency-dependent bounds
    if freq_bounds["show"]:
        freq_low_max = freq_bounds["freq_low_max"]
        mag_low_min = freq_bounds["mag_low_min"]
        freq_high_min = freq_bounds["freq_high_min"]
        mag_high_max = freq_bounds["mag_high_max"]
        
        # Low frequency bound
        low_mask = omega_range <= freq_low_max
        if np.any(low_mask):
            fig.add_trace(
                go.Scatter(
                    x=omega_range[low_mask],
                    y=[mag_low_min] * np.sum(low_mask),
                    mode='lines',
                    name='Disturbance attenuation',
                    line=dict(color='red', width=2, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    row=1, col=1
                )
            )
            fig.add_vline(
                x=freq_low_max, line_dash="dash", line_color="gray", 
                line_width=1, opacity=0.5, row=1, col=1
            )
        
        # High frequency bound
        high_mask = omega_range >= freq_high_min
        if np.any(high_mask):
            fig.add_trace(
                go.Scatter(
                    x=omega_range[high_mask],
                    y=[mag_high_max] * np.sum(high_mask),
                    mode='lines',
                    name='Measurement noise',
                    line=dict(color='red', width=2, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    row=1, col=1
                )
            )
            fig.add_vline(
                x=freq_high_min, line_dash="dash", line_color="gray", 
                line_width=1, opacity=0.5, row=1, col=1
            )
    
    # Phase plot
    fig.add_trace(
        go.Scatter(
            x=omega_range,
            y=phase,
            mode='lines',
            name='Phase',
            line=dict(color='blue', width=2),
            hovertemplate='ω: %{x:.4f} rad/s<br>Phase: %{y:.4f}°<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Common phase reference lines
    for phase_line in [-90, -180, -270]:
        if phase_line == -180:
            # -180° line (solid, prominent)
            fig.add_hline(
                y=phase_line, line_dash="solid", line_color="black", 
                line_width=1.0, opacity=0.8, row=2, col=1
            )
        else:
            # -90° and -270° lines (dashed, lighter)
            fig.add_hline(
                y=phase_line, line_dash="dash", line_color="gray", 
                line_width=0.5, opacity=0.4, row=2, col=1
            )
    
    # Logarithmic grid lines - major divisions (powers of 10)
    for freq in major_freqs:
        fig.add_vline(
            x=freq, line_dash="solid", line_color="lightgray", 
            line_width=0.8, opacity=0.6, row=2, col=1
        )
    
    # Logarithmic grid lines - minor divisions
    for freq in minor_freqs:
        fig.add_vline(
            x=freq, line_dash="dot", line_color="lightgray", 
            line_width=0.3, opacity=0.4, row=2, col=1
        )
    
    # Bandwidth indicator
    if np.isfinite(bw) and bw > 0:
        fig.add_vline(
            x=bw, line_dash="dash", line_color="green", 
            line_width=1.5, opacity=0.6, row=2, col=1
        )
        fig.add_annotation(
            x=bw,
            y=np.max(phase) if len(phase) > 0 else 0,
            text=f"BW<br>ω_bw={bw:.3g}",
            showarrow=False,
            font=dict(size=9, color="green"),
            bgcolor="white",
            bordercolor="green",
            borderwidth=1,
            row=2, col=1
        )
    
    # Phase margin indicator
    if show_margins and np.isfinite(pm) and np.isfinite(wpc) and wpc > 0:
        try:
            phase_at_wpc = np.interp(wpc, omega_range, phase)
            # Vertical line at phase crossover frequency
            fig.add_vline(
                x=wpc, line_dash="dot", line_color="red", 
                line_width=1.0, opacity=0.8, row=2, col=1
            )
            # Line connecting to -180°
            fig.add_trace(
                go.Scatter(
                    x=[wpc, wpc],
                    y=[phase_at_wpc, -180],
                    mode='lines',
                    name=f'PM = {pm:.2f}°',
                    line=dict(color='red', width=1.2, dash='dot'),
                    showlegend=True,
                    hovertemplate=f'Phase Crossover: ω={wpc:.4f} rad/s<br>PM={pm:.2f}°<extra></extra>'
                ),
                row=2, col=1
            )
            # Marker at phase crossover
            fig.add_trace(
                go.Scatter(
                    x=[wpc],
                    y=[phase_at_wpc],
                    mode='markers',
                    name='',
                    marker=dict(symbol='circle', size=8, color='red'),
                    showlegend=False,
                    hovertemplate=f'Phase Crossover: ω={wpc:.4f} rad/s<br>PM={pm:.2f}°<extra></extra>'
                ),
                row=2, col=1
            )
            # Text annotation for PM
            y_mid = (phase_at_wpc - 180) / 2
            fig.add_annotation(
                x=wpc,
                y=y_mid,
                text=f"PM<br>{pm:.2f}°",
                showarrow=False,
                font=dict(size=9, color="red"),
                bgcolor="white",
                bordercolor="red",
                borderwidth=1,
                row=2, col=1
            )
        except Exception:
            pass
    
    # Min Phase Margin specification
    if specs.show_specs and specs.min_phase_margin_deg is not None:
        min_pm = specs.min_phase_margin_deg
        spec_phase_line = -180 + min_pm
        fig.add_hline(
            y=spec_phase_line, line_dash="dash", line_color="orange", 
            line_width=2, opacity=0.7, row=2, col=1
        )
        # Find where phase > spec_phase_line and add green regions
        satisfied = phase > spec_phase_line
        if np.any(satisfied):
            # Create segments where spec is satisfied
            for i in range(len(omega_range) - 1):
                if satisfied[i] and satisfied[i+1]:
                    fig.add_shape(
                        type="rect",
                        x0=omega_range[i], x1=omega_range[i+1],
                        y0=-400, y1=400,
                        fillcolor="green",
                        opacity=0.15,
                        layer="below",
                        line_width=0,
                        row=2, col=1
                    )
    
    # Target bandwidth
    if specs.show_specs and specs.target_bw_rad is not None:
        target_bw = specs.target_bw_rad
        fig.add_vline(
            x=target_bw, line_dash="dash", line_color="purple", 
            line_width=2, opacity=0.7, row=2, col=1
        )
        fig.add_annotation(
            x=target_bw,
            y=np.max(phase) if len(phase) > 0 else 0,
            text=f"Target BW<br>{target_bw:.3g} rad/s",
            showarrow=False,
            font=dict(size=9, color="purple"),
            bgcolor="white",
            bordercolor="purple",
            borderwidth=1,
            row=2, col=1
        )
    
    # Update x-axis for both subplots (log scale with proper grid)
    # Configure major and minor grid lines for logarithmic scale
    fig.update_xaxes(
        type="log",
        title_text="ω (rad/s)",
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=0.8,
        minor=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.3,
            griddash='dot'
        ),
        row=1, col=1
    )
    fig.update_xaxes(
        type="log",
        title_text="ω (rad/s)",
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=0.8,
        minor=dict(
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=0.3,
            griddash='dot'
        ),
        row=2, col=1
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
    fig.update_yaxes(title_text="Phase (deg)", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        width=600,
        showlegend=True,
        plot_bgcolor='white',
        legend=dict(
            bgcolor='white',
            bordercolor='gray',
            borderwidth=1,
            font=dict(color='black'),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=0.5),
        yaxis2=dict(showgrid=True, gridcolor='lightgray', gridwidth=0.5),
        # Add custom subtitles with proper spacing
        annotations=[
            dict(
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                text="<b>Bode Plot - Magnitude</b>",
                showarrow=False,
                font=dict(size=14, color="white"),
                xanchor="center",
                bgcolor="black",
                bordercolor="black",
                borderwidth=0
            ),
            dict(
                x=0.5,
                y=0.47,
                xref="paper",
                yref="paper",
                text="<b>Bode Plot - Phase</b>",
                showarrow=False,
                font=dict(size=14, color="white"),
                xanchor="center",
                bgcolor="black",
                bordercolor="black",
                borderwidth=0
            )
        ]
    )
    
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
    
    show_margins = st.checkbox("Show margins on Bode plot", value=True)

st.sidebar.header("Frequency Domain Specifications")
st.sidebar.caption("(Bode obstacle course)")

with st.sidebar:
    show_specs = st.checkbox("Enable specifications", value=False)
    
    # Phase Margin
    show_pm = st.checkbox("Show phase margin specification", value=False) if show_specs else False
    min_pm = (
        st.number_input("Min phase margin (deg)", value=45.0, step=1.0, format="%.6g")
        if show_pm else None
    )
    
    # Gain Margin
    show_gm = st.checkbox("Show gain margin specification", value=False) if show_specs else False
    min_gm_db = (
        st.number_input("Min gain margin (dB)", value=10.0, step=1.0, format="%.6g")
        if show_gm else None
    )
    
    # Target Bandwidth
    show_bw = st.checkbox("Show target bandwidth specification", value=False) if show_specs else False
    target_bw = (
        st.number_input("Target closed-loop BW ω_B (rad/s)", value=10.0, step=0.5, format="%.6g")
        if show_bw else None
    )
    
    # Frequency-dependent bounds
    show_bounds = st.checkbox("Show frequency-dependent bounds", value=False) if show_specs else False
    if show_bounds:
        st.markdown("**Low-frequency (disturbance attenuation):**")
        freq_low_max = st.number_input("ω_low_max (rad/s)", value=0.1, step=0.01, format="%.6g")
        mag_low_min = st.number_input("Min magnitude at low freq (dB)", value=-20.0, step=1.0, format="%.6g")
        
        st.markdown("**High-frequency (measurement noise):**")
        freq_high_min = st.number_input("ω_high_min (rad/s)", value=10.0, step=10.0, format="%.6g")
        mag_high_max = st.number_input("Max magnitude at high freq (dB)", value=-40.0, step=1.0, format="%.6g")
        
        freq_bounds = {
            "show": True,
            "freq_low_max": freq_low_max,
            "mag_low_min": mag_low_min,
            "freq_high_min": freq_high_min,
            "mag_high_max": mag_high_max,
        }
    else:
        freq_bounds = {"show": False}
    
    specs = SpecConfig(
        show_specs=show_specs,
        min_phase_margin_deg=min_pm,
        min_gain_margin_db=min_gm_db,
        target_bw_rad=target_bw
    )

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
    bode_fig = plot_bode(loop_sys, omega_range, delay_info, show_margins, specs, freq_bounds)
    if bode_fig:
        st.plotly_chart(bode_fig, width='stretch')
    
    # Display margins and bandwidth
    if show_margins:
        gm, pm, wgc, wpc, bw = compute_margins_and_bw(loop_sys, omega_range)
        st.markdown("**Margins & Bandwidth**")
        colA, colB, colC, colD, colE = st.columns(5)
        
        if np.isfinite(gm):
            gm_db = 20 * np.log10(gm)
            colA.metric("Gain margin (×)", f"{gm:.4g}", f"{gm_db:.2f} dB")
        else:
            colA.metric("Gain margin (×)", "∞", "∞ dB")
        
        if np.isfinite(pm):
            colB.metric("Phase margin", f"{pm:.2f}°", "")
        else:
            colB.metric("Phase margin", "∞", "")
        
        if np.isfinite(wgc):
            colC.metric("ω_gc (rad/s)", f"{wgc:.4g}", "")
        else:
            colC.metric("ω_gc (rad/s)", "N/A", "")
        
        if np.isfinite(wpc):
            colD.metric("ω_pc (rad/s)", f"{wpc:.4g}", "")
        else:
            colD.metric("ω_pc (rad/s)", "N/A", "")
        
        if np.isfinite(bw):
            colE.metric("BW (rad/s)", f"{bw:.4g}", "")
        else:
            colE.metric("BW (rad/s)", "N/A", "")

with col2:
    st.markdown("### Nyquist Plot")
    nyquist_fig = plot_nyquist(loop_sys, omega_range, show_unity_circle=show_unity, delay_info=delay_info)
    if nyquist_fig:
        st.plotly_chart(nyquist_fig, width='stretch')
    else:
        st.error("Could not generate Nyquist plot")

