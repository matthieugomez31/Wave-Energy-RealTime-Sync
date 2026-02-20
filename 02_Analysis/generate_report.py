import sys
import os
import datetime
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    from fpdf import FPDF
except ImportError:
    print("[ERROR] The 'fpdf' library is required to generate PDF reports.")
    print("Please install it by running: pip install fpdf")
    sys.exit(1)

# Import the data analysis pipeline (assumes data_analysis.py is in the same folder or python path)
import data_analysis as da

# ==========================================
# CONFIGURATION & FILE PATH
# ==========================================
DEFAULT_FILE_PATH = "03_Data_Samples/log_experience_dummy.csv"
OUTPUT_REPORT_DIR = "04_Reports/Experiments"

# ==========================================
# REPORT TARGETS & CRITERIA
# ==========================================
TARGET_MIN_HEAVE_MM = 50.0      # Minimum acceptable vertical motion
TARGET_MAX_DRIFT_RATIO = 0.30   # Drift (X) should be less than 30% of Heave (Y)
TARGET_PEAK_POWER_W = 2.5       # Target power for phone charging
TARGET_AVG_POWER_W = 0.5        # Minimum continuous power generation
TARGET_MIN_R2 = 0.75            # Minimum R-squared for Power vs Velocity (efficiency/loss check)

# ==========================================
# ANALYSIS FUNCTIONS
# ==========================================
def perform_frequency_analysis(df):
    """ Computes the dominant frequency of the heave motion (Pos_Y) using FFT. """
    if 'Pos_Y_Smooth_mm' not in df.columns or len(df) < 10:
        return 0.0, None, None
        
    dt = np.mean(np.diff(df['Time_sec']))
    if dt == 0 or np.isnan(dt): return 0.0, None, None
    fs = 1.0 / dt
    
    y_data = df['Pos_Y_Smooth_mm'].values
    y_data_centered = y_data - np.mean(y_data)
    
    n = len(y_data_centered)
    fft_values = np.fft.fft(y_data_centered)
    fft_freqs = np.fft.fftfreq(n, d=dt)
    
    pos_mask = fft_freqs > 0
    fft_freqs = fft_freqs[pos_mask]
    fft_amplitudes = np.abs(fft_values)[pos_mask] / n
    
    if len(fft_amplitudes) == 0:
        return 0.0, None, None
        
    max_amp_idx = np.argmax(fft_amplitudes)
    dominant_freq = fft_freqs[max_amp_idx]
    
    return dominant_freq, fft_freqs, fft_amplitudes

def calculate_kinematics_and_r2(df):
    """ Calculates velocity norm, fits a Power vs Velocity curve, and finds R^2. """
    dt = np.mean(np.diff(df['Time_sec']))
    
    # Calculate velocities if not present
    vel_x = np.gradient(df['Pos_X_Smooth_mm'], dt)
    vel_y = np.gradient(df['Pos_Y_Smooth_mm'], dt)
    
    # Velocity norm: ||V|| = sqrt(Vx^2 + Vy^2)
    v_norm = np.sqrt(vel_x**2 + vel_y**2)
    df['Vel_Norm_mm_s'] = v_norm
    
    power = df['Power_W'].values
    
    # --- Filter out NaN or Inf values before polyfit to prevent SVD convergence error ---
    valid_mask = np.isfinite(v_norm) & np.isfinite(power)
    v_norm_clean = v_norm[valid_mask]
    power_clean = power[valid_mask]
    
    if len(v_norm_clean) < 3:
        print("[WARNING] Not enough valid data points to fit polynomial.")
        return v_norm, power, np.zeros_like(v_norm), 0.0, [0, 0, 0]
    
    # Fit a 2nd degree polynomial: Power = a*v^2 + b*v + c on clean data
    coeffs = np.polyfit(v_norm_clean, power_clean, 2)
    
    # Evaluate fit on original data for plotting (ignoring NaNs visually)
    p_fit = np.polyval(coeffs, v_norm) 
    
    # Calculate R-squared using only the clean, valid data
    p_fit_clean = np.polyval(coeffs, v_norm_clean)
    ss_res = np.sum((power_clean - p_fit_clean)**2)
    ss_tot = np.sum((power_clean - np.mean(power_clean))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return v_norm, power, p_fit, r_squared, coeffs

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================
def save_mechanical_plots(df, output_filepath):
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['Time_sec'], df['Pos_Y_Smooth_mm'], 'b-', label='Heave (Y)')
    ax1.set_title("Vertical Displacement (Heave) vs Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Y Position (mm)")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['Time_sec'], df['Pos_X_Smooth_mm'], 'r-', label='Drift (X)')
    ax2.set_title("Lateral Displacement (Drift) vs Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("X Position (mm)")
    ax2.grid(True)
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, :], projection='3d')
    img = ax3.scatter(df['Time_sec'], df['Pos_X_Smooth_mm'], df['Pos_Y_Smooth_mm'], 
                      c=df['Time_sec'], cmap='viridis', s=10)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Disp. X (mm)')
    ax3.set_zlabel('Disp. Y (mm)')
    ax3.set_title('3D Spatio-Temporal Trajectory')
    fig.colorbar(img, ax=ax3, label='Time (s)')

    plt.tight_layout()
    plt.savefig(output_filepath, dpi=150)
    plt.close()

def save_electrical_plots(df, v_norm, power, p_fit, r_squared, output_filepath):
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Voltage & Current
    ax0 = fig.add_subplot(gs[0, 0])
    color = 'tab:red'
    ax0.set_xlabel('Time (s)')
    ax0.set_ylabel('Voltage (V)', color=color)
    ax0.plot(df['Time_sec'], df['Voltage_V'], color=color, label='Voltage')
    ax0.tick_params(axis='y', labelcolor=color)
    ax0.grid(True, alpha=0.3)
    ax0_twin = ax0.twinx()
    color = 'tab:blue'
    ax0_twin.set_ylabel('Current (mA)', color=color)
    ax0_twin.plot(df['Time_sec'], df['Current_mA'], color=color, label='Current', alpha=0.7)
    ax0_twin.tick_params(axis='y', labelcolor=color)
    ax0.set_title('Voltage & Current vs Time')

    # 2. Power vs Time
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.fill_between(df['Time_sec'], df['Power_W'], color='purple', alpha=0.3)
    ax1.plot(df['Time_sec'], df['Power_W'], color='purple')
    ax1.axhline(y=TARGET_PEAK_POWER_W, color='r', linestyle='--', label=f'Target ({TARGET_PEAK_POWER_W}W)')
    ax1.set_title('Instantaneous Power vs Time')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Power (W)')
    ax1.grid(True)
    ax1.legend()

    # 3. Position vs Power
    ax2 = fig.add_subplot(gs[1, 0])
    scatter = ax2.scatter(df['Pos_Y_Smooth_mm'], df['Power_W'], c=df['Time_sec'], cmap='plasma', alpha=0.6, s=15)
    ax2.set_title('Power Generated vs Vertical Position (Heave)')
    ax2.set_xlabel('Y Position (mm)')
    ax2.set_ylabel('Power (W)')
    ax2.grid(True)
    fig.colorbar(scatter, ax=ax2, label='Time (s)')

    # 4. Power vs Velocity Norm with R^2
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(v_norm, power, color='gray', alpha=0.5, s=10, label='Data points')
    
    # Sort for plotting the line nicely (ignore NaNs for plotting the line)
    valid_plot_mask = np.isfinite(v_norm) & np.isfinite(p_fit)
    v_norm_plot = v_norm[valid_plot_mask]
    p_fit_plot = p_fit[valid_plot_mask]
    
    sort_idx = np.argsort(v_norm_plot)
    ax3.plot(v_norm_plot[sort_idx], p_fit_plot[sort_idx], color='red', linewidth=2, label='Polynomial Fit (Deg 2)')
    
    ax3.set_title(f'Power vs Velocity Norm ($R^2$ = {r_squared:.4f})')
    ax3.set_xlabel('Velocity Norm ||V|| (mm/s)')
    ax3.set_ylabel('Power (W)')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_filepath, dpi=150)
    plt.close()

def save_frequency_plot(freqs, amps, dominant_f, output_filepath):
    if freqs is None: return
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, amps, 'g-')
    plt.axvline(x=dominant_f, color='r', linestyle='--', label=f'Dominant: {dominant_f:.2f} Hz')
    plt.title('Wave Frequency Analysis (FFT on Heave)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(0, max(5, dominant_f * 3))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=150)
    plt.close()

# ==========================================
# PDF GENERATOR CLASS
# ==========================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(0, 51, 102) # Dark Blue
        self.cell(0, 10, 'Wave Energy Converter - Performance Report', border=0, ln=1, align='C')
        self.set_line_width(0.5)
        self.line(10, 20, 200, 20)
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# ==========================================
# MAIN ROUTINE
# ==========================================
def generate_report(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        print("Please check the path in DEFAULT_FILE_PATH or pass the correct path as an argument.")
        return

    print(f"\n[REPORT GENERATOR] Analyzing data from: {filepath}")
    
    # --- 1. Load & Process Data ---
    df = da.load_and_inspect_data(filepath)
    if df is None or df.empty: return
    df = da.preprocess_data(df)
    df = da.calculate_physics(df)
    
    # Drop completely invalid rows before analysis
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Pos_Y_Smooth_mm', 'Power_W'])
    
    if df.empty: 
        print("[ERROR] DataFrame is empty after processing and cleaning NaNs.")
        return

    # --- 2. Extract Metrics ---
    duration = df['Time_sec'].iloc[-1] - df['Time_sec'].iloc[0]
    
    max_heave = df['Pos_Y_Smooth_mm'].max() - df['Pos_Y_Smooth_mm'].min()
    max_drift = df['Pos_X_Smooth_mm'].max() - df['Pos_X_Smooth_mm'].min()
    drift_ratio = max_drift / max_heave if max_heave > 0 else 0
    
    max_power = df['Power_W'].max()
    avg_power = df['Power_W'].mean()
    total_energy = df['Energy_J'].iloc[-1]
    
    dominant_freq, freqs, amps = perform_frequency_analysis(df)
    v_norm, power, p_fit, r_squared, coeffs = calculate_kinematics_and_r2(df)

    # --- 3. Evaluate Criteria ---
    crit_heave = "PASS" if max_heave >= TARGET_MIN_HEAVE_MM else "FAIL"
    crit_drift = "PASS" if drift_ratio <= TARGET_MAX_DRIFT_RATIO else "FAIL"
    crit_peak_p = "PASS" if max_power >= TARGET_PEAK_POWER_W else "FAIL"
    crit_avg_p = "PASS" if avg_power >= TARGET_AVG_POWER_W else "FAIL"
    crit_r2 = "PASS" if r_squared >= TARGET_MIN_R2 else "FAIL"

    # Ensure output directory exists
    os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)
    base_name = os.path.basename(filepath).replace('.csv', '')
    pdf_filepath = os.path.join(OUTPUT_REPORT_DIR, f"{base_name}_Performance_Report.pdf")

    print("[REPORT GENERATOR] Generating plots and compiling PDF...")

    # --- 4. Use a Temporary Directory for Images ---
    with tempfile.TemporaryDirectory() as temp_dir:
        mech_plot_path = os.path.join(temp_dir, "plot_mechanical.png")
        elec_plot_path = os.path.join(temp_dir, "plot_electrical.png")
        freq_plot_path = os.path.join(temp_dir, "plot_frequency.png")
        
        save_mechanical_plots(df, mech_plot_path)
        save_electrical_plots(df, v_norm, power, p_fit, r_squared, elec_plot_path)
        if freqs is not None:
            save_frequency_plot(freqs, amps, dominant_freq, freq_plot_path)

        # --- 5. Build PDF Report ---
        pdf = PDFReport()
        pdf.add_page()
        
        # Metadata
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 6, f"Source File: {filepath}", ln=1)
        pdf.cell(0, 6, f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
        pdf.cell(0, 6, f"Dataset Duration: {duration:.2f} seconds", ln=1)
        pdf.ln(5)
        
        # Criteria Table
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "1. Validation Criteria Summary", ln=1)
        
        pdf.set_font("Arial", 'B', 10)
        pdf.set_fill_color(200, 220, 255)
        
        # Table Header
        col_w = [45, 45, 45, 30]
        pdf.cell(col_w[0], 8, "Metric", border=1, fill=True, align='C')
        pdf.cell(col_w[1], 8, "Target", border=1, fill=True, align='C')
        pdf.cell(col_w[2], 8, "Actual", border=1, fill=True, align='C')
        pdf.cell(col_w[3], 8, "Status", border=1, fill=True, align='C', ln=1)
        
        # Table Content
        pdf.set_font("Arial", '', 10)
        metrics = [
            ("Heave Amplitude", f">= {TARGET_MIN_HEAVE_MM} mm", f"{max_heave:.1f} mm", crit_heave),
            ("Lateral Drift", f"<= {TARGET_MAX_DRIFT_RATIO*100}% of Y", f"{drift_ratio*100:.1f}%", crit_drift),
            ("Peak Power", f">= {TARGET_PEAK_POWER_W} W", f"{max_power:.2f} W", crit_peak_p),
            ("Average Power", f">= {TARGET_AVG_POWER_W} W", f"{avg_power:.2f} W", crit_avg_p),
            ("Power-Velocity R-Sq", f">= {TARGET_MIN_R2}", f"{r_squared:.3f}", crit_r2)
        ]
        
        for name, target, actual, status in metrics:
            pdf.cell(col_w[0], 8, name, border=1)
            pdf.cell(col_w[1], 8, target, border=1, align='C')
            pdf.cell(col_w[2], 8, actual, border=1, align='C')
            
            if status == "PASS":
                pdf.set_text_color(0, 128, 0) # Green
                pdf.set_font("Arial", 'B', 10)
            else:
                pdf.set_text_color(200, 0, 0) # Red
                pdf.set_font("Arial", 'B', 10)
                
            pdf.cell(col_w[3], 8, status, border=1, align='C', ln=1)
            pdf.set_text_color(0, 0, 0) # Reset
            pdf.set_font("Arial", '', 10)
        
        pdf.ln(5)
        
        # Electrical Section
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "2. Electrical Performance & Kinematic Correlation", ln=1)
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 6, f"- Total Energy Harvested: {total_energy:.2f} Joules", ln=1)
        pdf.cell(0, 6, f"- Power vs Velocity Norm Fit R-Squared: {r_squared:.4f}", ln=1)
        
        # Insert Electrical Plot
        pdf.image(elec_plot_path, x=10, w=190)
        
        # --- PAGE 2 ---
        pdf.add_page()
        
        # Mechanical Section
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "3. Kinematic & Mechanical Stability", ln=1)
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 6, f"- Maximum Heave (Y): {max_heave:.1f} mm", ln=1)
        pdf.cell(0, 6, f"- Maximum Drift (X): {max_drift:.1f} mm", ln=1)
        
        # Insert Mechanical Plot
        pdf.image(mech_plot_path, x=10, w=190)
        pdf.ln(5)
        
        # Frequency Section
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "4. Frequency Analysis (FFT)", ln=1)
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 6, f"- Dominant Wave Frequency detected: {dominant_freq:.3f} Hz", ln=1)
        
        # Insert Frequency Plot
        if os.path.exists(freq_plot_path):
            pdf.image(freq_plot_path, x=10, w=190)

        # --- Save PDF ---
        pdf.output(pdf_filepath)
        
        # The temporary directory (and the .png plots inside it) is automatically deleted here
    
    print(f"[REPORT GENERATOR] Success! Full PDF report generated: '{pdf_filepath}'")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = DEFAULT_FILE_PATH
        
    generate_report(target_file)