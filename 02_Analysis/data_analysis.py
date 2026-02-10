import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Path to your file
FILE_PATH = '03_Data_Samples/log_experience_20251125_163304.csv' 

# CSV Format Settings
CSV_SEP = ';'       # Separator is semicolon
CSV_DECIMAL = ','   # Decimal is comma (e.g., 0,01)

# --- TIME CROPPING (New!) ---
# Adjust these values based on what you see in the raw plot
ANALYSIS_START_SEC = 0    # Start analysis at X seconds (skip setup time)
ANALYSIS_END_SEC = None   # Set to a number (e.g. 60) to stop early, or None to read until end

# Physics Constants
SMOOTHING_WINDOW = 15  # Window size for filtering (must be odd)
SMOOTHING_POLY = 2     # Polynomial order for filtering

def load_and_inspect_data(filepath):
    """
    Loads the CSV file and prints basic info.
    """
    print(f"📂 Loading file: {filepath}...")

    try:
        df = pd.read_csv(filepath, sep=CSV_SEP, decimal=CSV_DECIMAL)
        df.columns = [c.strip() for c in df.columns]
        
        print("✅ File loaded successfully!")
        print(f"📊 Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    except FileNotFoundError:
        print("❌ Error: File not found. Check the path.")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def preprocess_data(df):
    """
    Basic cleaning: Time normalization, CROPPING, NaN handling, and ZEROING.
    """
    # 1. Normalize Time (Absolute from 0)
    if 'Timestamp_ms' in df.columns:
        start_time = df['Timestamp_ms'].iloc[0]
        df['Time_sec'] = (df['Timestamp_ms'] - start_time) / 1000.0
    else:
        print("⚠️ Warning: No 'Timestamp_ms'. Creating artificial time.")
        df['Time_sec'] = np.arange(len(df)) * 0.01

    # --- 1.5 TIME CROPPING (The filter) ---
    original_len = len(df)
    
    # Filter Start
    df = df[df['Time_sec'] >= ANALYSIS_START_SEC]
    
    # Filter End (if set)
    if ANALYSIS_END_SEC is not None:
        df = df[df['Time_sec'] <= ANALYSIS_END_SEC]
        
    print(f"✂️ Cropped data: Kept {len(df)}/{original_len} rows ({ANALYSIS_START_SEC}s to {ANALYSIS_END_SEC if ANALYSIS_END_SEC else 'End'}s)")
    
    # If empty after cropping, stop safely
    if df.empty:
        print("❌ Error: No data left after cropping! Check your START/END times.")
        return df

    # Re-normalize time so the graph starts at 0 relative to the crop
    # (Optional: keep this if you want the graph to show '0' at the new start)
    df['Time_sec'] = df['Time_sec'] - df['Time_sec'].iloc[0]

    # 2. Handle Lost Tracking (-1 to NaN)
    for col in ['Vision_Y_mm', 'Vision_X_mm']:
        if col in df.columns:
            clean_col = col.replace('_mm', '_Clean')
            df[clean_col] = df[col].replace(-1, np.nan).replace(-1.0, np.nan)

    # 3. Zeroing / Relative Displacement
    # Now that we cropped the "setup phase", the first point IS the experiment start.
    if 'Vision_X_Clean' in df.columns:
        first_valid_x = df['Vision_X_Clean'].dropna().iloc[0] if not df['Vision_X_Clean'].dropna().empty else 0
        df['Vision_X_Rel'] = df['Vision_X_Clean'] - first_valid_x
    
    if 'Vision_Y_Clean' in df.columns:
        first_valid_y = df['Vision_Y_Clean'].dropna().iloc[0] if not df['Vision_Y_Clean'].dropna().empty else 0
        df['Vision_Y_Rel'] = df['Vision_Y_Clean'] - first_valid_y

    # 4. Absolute Displacement
    if 'Vision_X_Rel' in df.columns and 'Vision_Y_Rel' in df.columns:
        df['Vision_Abs_Disp'] = np.sqrt(df['Vision_X_Rel']**2 + df['Vision_Y_Rel']**2)

    return df

def calculate_physics(df):
    """
    The Core Physics Engine:
    1. Interpolate missing data
    2. Smooth position (X and Y)
    3. Calculate Velocity (Derivative)
    4. Calculate Power & Energy
    """
    if df.empty: return df
    
    print("⚙️ Running Physics Engine...")
    
    # --- A. Kinematics Y (Heave - Useful Motion) ---
    if 'Vision_Y_Rel' in df.columns:
        # Interpolate
        df['Pos_Y_Interp'] = df['Vision_Y_Rel'].interpolate(method='linear')
        
        # Smooth
        if len(df) > SMOOTHING_WINDOW:
            df['Pos_Y_Smooth_mm'] = savgol_filter(df['Pos_Y_Interp'], SMOOTHING_WINDOW, SMOOTHING_POLY)
        else:
            df['Pos_Y_Smooth_mm'] = df['Pos_Y_Interp']
            
        # Velocity Y
        pos_y_m = df['Pos_Y_Smooth_mm'] / 1000.0
        df['Velocity_Y_m_s'] = np.gradient(pos_y_m, df['Time_sec'])

    # --- B. Kinematics X (Surge/Sway - Parasitic Motion) ---
    if 'Vision_X_Rel' in df.columns:
        # Interpolate
        df['Pos_X_Interp'] = df['Vision_X_Rel'].interpolate(method='linear')
        
        # Smooth
        if len(df) > SMOOTHING_WINDOW:
            df['Pos_X_Smooth_mm'] = savgol_filter(df['Pos_X_Interp'], SMOOTHING_WINDOW, SMOOTHING_POLY)
        else:
            df['Pos_X_Smooth_mm'] = df['Pos_X_Interp']
            
        # Velocity X
        pos_x_m = df['Pos_X_Smooth_mm'] / 1000.0
        df['Velocity_X_m_s'] = np.gradient(pos_x_m, df['Time_sec'])
    
    # --- C. Electrical (Puissance) ---
    if 'Voltage_V' in df.columns and 'Current_mA' in df.columns:
        df['Power_W'] = df['Voltage_V'] * (df['Current_mA'] / 1000.0)
        dt = np.gradient(df['Time_sec'])
        df['Energy_J'] = np.cumsum(df['Power_W'] * dt)
        
    return df

def plot_physics_dashboard(df):
    """
    Displays the advanced physics analysis (Energy focus).
    """
    if df.empty: return

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: Kinematics Y
    axs[0].plot(df['Time_sec'], df['Vision_Y_Rel'], '.', color='lightgray', label='Raw Y (Relative)')
    if 'Pos_Y_Smooth_mm' in df.columns:
        axs[0].plot(df['Time_sec'], df['Pos_Y_Smooth_mm'], color='blue', linewidth=2, label='Smoothed Y')
    axs[0].set_ylabel('Position Y (mm)')
    axs[0].set_title(f'1. Heave Motion (Start @ {ANALYSIS_START_SEC}s)')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: Power
    if 'Power_W' in df.columns:
        axs[1].fill_between(df['Time_sec'], df['Power_W'], color='purple', alpha=0.3)
        axs[1].plot(df['Time_sec'], df['Power_W'], color='purple', label='Inst. Power (W)')
        axs[1].axhline(y=2.5, color='red', linestyle=':', label='Target: Phone Charge (2.5W)')
    axs[1].set_ylabel('Power (Watts)')
    axs[1].set_title('2. Electrical Generation')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, alpha=0.3)
    
    # Plot 3: Energy
    if 'Energy_J' in df.columns:
        final_energy = df['Energy_J'].iloc[-1]
        axs[2].plot(df['Time_sec'], df['Energy_J'], color='green', linewidth=3, label=f'Total: {final_energy:.2f} J')
    axs[2].set_ylabel('Energy (Joules)')
    axs[2].set_xlabel('Time (seconds)')
    axs[2].set_title('3. Cumulative Energy')
    axs[2].legend(loc='upper left')
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_mechanical_analysis(df):
    """
    NEW: Analyzes mechanical stability (X vs Y and 3D).
    Allows to see if the buoy moves straight or wobbles.
    """
    if df.empty: return

    print("📊 Generating Mechanical Analysis...")
    fig = plt.figure(figsize=(14, 10))
    
    # --- 1. 2D Trajectory (Frontal View: Y vs X) ---
    # Ideally, this should be a vertical line.
    ax1 = fig.add_subplot(2, 2, 1)
    if 'Pos_X_Smooth_mm' in df.columns and 'Pos_Y_Smooth_mm' in df.columns:
        ax1.plot(df['Pos_X_Smooth_mm'], df['Pos_Y_Smooth_mm'], color='purple', alpha=0.6, linewidth=1)
        # Mark Start and End
        ax1.plot(df['Pos_X_Smooth_mm'].iloc[0], df['Pos_Y_Smooth_mm'].iloc[0], 'go', label='Start')
        ax1.plot(df['Pos_X_Smooth_mm'].iloc[-1], df['Pos_Y_Smooth_mm'].iloc[-1], 'rs', label='End')
        
    ax1.set_xlabel('Horiz. Displacement X (mm)')
    ax1.set_ylabel('Vert. Displacement Y (mm)')
    ax1.set_title('2D Trajectory (Relative to Start)')
    ax1.axis('equal') # Crucial to see real shape
    ax1.grid(True)
    ax1.legend()

    # --- 2. X Stability over Time ---
    ax2 = fig.add_subplot(2, 2, 2)
    if 'Pos_X_Smooth_mm' in df.columns:
        ax2.plot(df['Time_sec'], df['Pos_X_Smooth_mm'], color='orange', label='Position X (Smoothed)')
        
        # Calculate drift stats
        drift_range = df['Pos_X_Smooth_mm'].max() - df['Pos_X_Smooth_mm'].min()
        ax2.text(0.05, 0.9, f"Max Lateral Drift: {drift_range:.1f} mm", transform=ax2.transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8))

        # Add absolute displacement comparison
        if 'Vision_Abs_Disp' in df.columns:
            # We filter/smooth abs disp just for plotting comparison or plot raw
            ax2.plot(df['Time_sec'], df['Vision_Abs_Disp'], color='black', linestyle=':', alpha=0.5, label='Total 2D Motion (Abs)')
                 
    ax2.set_ylabel('Displacement (mm)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Lateral Drift vs Total Motion')
    ax2.grid(True)
    ax2.legend()

    # --- 3. 3D Plot (Time, X, Y) ---
    # Shows the evolution of the trajectory in space-time
    ax3 = fig.add_subplot(2, 1, 2, projection='3d')
    if 'Pos_X_Smooth_mm' in df.columns and 'Pos_Y_Smooth_mm' in df.columns:
        # Scatter plot colored by time
        img = ax3.scatter(df['Time_sec'], df['Pos_X_Smooth_mm'], df['Pos_Y_Smooth_mm'], 
                          c=df['Time_sec'], cmap='viridis', s=2)
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Disp. X (mm)')
        ax3.set_zlabel('Disp. Y (mm)')
        ax3.set_title('3D Motion Visualization (Space-Time)')
        fig.colorbar(img, ax=ax3, label='Time (s)')

    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load
    df = load_and_inspect_data(FILE_PATH)
    
    if df is not None:
        # 2. Clean, Crop & Zeroing
        df = preprocess_data(df)
        
        # 3. Compute Physics
        df = calculate_physics(df)
        
        # 4. Visualize - Dashboard 1: Energy & Power
        plot_physics_dashboard(df)
        
        # 5. Visualize - Dashboard 2: Mechanical Stability
        plot_mechanical_analysis(df)