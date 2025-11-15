# Nyquist Plot Tool

An interactive Nyquist plot tool for control systems analysis, built with Streamlit.

## Features

- **Plant Input**: Enter plant transfer function P(s) as:
  - Numerator/Denominator coefficients
  - Zeros/Poles/Gain format
  
- **Controller Options**: Add various compensators C(s):
  - PID controller with adjustable Kp, Ki, Kd
  - Lead compensator
  - Lag compensator
  - Lead-Lag compensator
  - Custom dynamic compensator (num/den)
  
- **Time Delay**: Add transport delay e^(-τs) using Padé approximation

- **Stability Analysis**:
  - Automatic computation of gain and phase margins
  - Frequency at gain/phase crossover
  - Stability indicator
  - Visual representation of margins on the plot

- **Interactive Nyquist Plot**:
  - Shows L(jω) for positive and negative frequencies
  - Critical point (-1, 0) marked
  - Unit circle overlay
  - Start/end frequency markers
  - Gain and phase margin visualization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run nyquist_tool.py
```

Then open your browser to the URL shown (typically http://localhost:8501)

## How to Use

1. **Enter Plant Transfer Function**:
   - Select input format (Numerator/Denominator or Zeros/Poles/Gain)
   - Enter the coefficients or values

2. **Add Controller (Optional)**:
   - Check "Add compensator C(s)"
   - Select controller type
   - Adjust parameters

3. **Add Time Delay (Optional)**:
   - Check "Add time delay"
   - Enter delay τ in seconds
   - Adjust Padé approximation order if needed

4. **Adjust Plot Settings**:
   - Set frequency range (min/max)
   - Choose logarithmic or linear frequency scale
   - Toggle unit circle and margin visualization

5. **Analyze Results**:
   - View stability margins (GM, PM)
   - Check stability indicator
   - Examine Nyquist plot for encirclements

## Interpreting the Nyquist Plot

### Nyquist Stability Criterion

The closed-loop system is stable if:
- **For systems with no RHP poles**: The Nyquist curve does NOT encircle the critical point (-1, 0)
- **For systems with N RHP poles**: The Nyquist curve makes N counter-clockwise encirclements of (-1, 0)

### Key Indicators

- **Gain Margin (GM)**: Distance from the Nyquist curve to (-1, 0) along the negative real axis
  - GM > 0 dB is desirable
  - Larger GM = more robust to gain variations

- **Phase Margin (PM)**: Angular separation from the curve to (-1, 0) at unity gain
  - PM > 0° is desirable
  - PM ≈ 45-60° typically gives good transient response

### Plot Elements

- **Blue solid line**: Open-loop frequency response for ω > 0
- **Blue dashed line**: Mirror image for ω < 0
- **Red X**: Critical point (-1, 0)
- **Green circle**: Starting point (low frequency)
- **Red square**: Ending point (high frequency)
- **Dashed lines**: Vectors to GM and PM points (when enabled)

## Examples

### Example 1: Simple Plant
```
Plant: P(s) = 1/(s+1)
No controller
Result: Stable with infinite GM and 90° PM
```

### Example 2: Plant with PID
```
Plant: P(s) = 1/(s(s+1))
Controller: PID with Kp=2, Ki=1, Kd=0.5
Result: Stabilized with improved margins
```

### Example 3: Lead Compensator
```
Plant: P(s) = 1/(s^2+2s+10)
Controller: Lead with zero=1, pole=10, gain=5
Result: Increased phase margin
```

### Example 4: System with Delay
```
Plant: P(s) = 1/(s+1)
Time delay: τ = 0.5s
Result: Reduced phase margin due to delay
```

## Notes

- The tool uses the Python Control Systems Library
- Time delays are approximated using Padé approximation
- For accurate results with time delays, use higher Padé orders
- Very high or very low frequencies may cause numerical issues
- Increase the number of points for smoother curves

## Troubleshooting

**"Could not compute margins"**
- System may be marginally stable or have numerical issues
- Try adjusting frequency range or number of points

**"System may be unstable"**
- Check if Nyquist curve encircles (-1, 0)
- Verify pole locations and encirclement count
- Add or adjust compensator to improve stability

**Irregular curve**
- Increase number of frequency points
- Adjust frequency range to capture important dynamics
- Check for very fast or very slow poles

## License

See main repository LICENSE file.

