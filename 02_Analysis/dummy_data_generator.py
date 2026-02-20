import os
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = "03_Data_Samples"
OUTPUT_FILENAME = "log_experience_dummy.csv"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

DURATION_SEC = 60.0       # 60 seconds of data
SAMPLE_RATE_HZ = 20.0     # 20 samples per second
DT = 1.0 / SAMPLE_RATE_HZ

# ==========================================
# GENERATE SIMULATED DATA
# ==========================================
def generate_data():
    print("[DUMMY DATA] Generating physical simulation...")
    
    # Time vector
    t = np.arange(0, DURATION_SEC, DT)
    n_samples = len(t)
    
    # 1. Heave (Pos_Y) - Main vertical wave motion
    # Dominant frequency = 0.6 Hz, Amplitude = ~70 mm (Passes > 50mm criteria)
    freq_wave = 0.6
    pos_y = 70.0 * np.sin(2 * np.pi * freq_wave * t) 
    # Add a secondary smaller wave and some noise for realism
    pos_y += 15.0 * np.sin(2 * np.pi * (freq_wave * 2.1) * t)
    pos_y += np.random.normal(0, 1.5, n_samples)
    
    # 2. Drift (Pos_X) - Lateral motion
    # Slow drift frequency = 0.05 Hz, Amplitude = ~15 mm (Passes < 30% of Heave criteria)
    pos_x = 15.0 * np.sin(2 * np.pi * 0.05 * t)
    pos_x += np.random.normal(0, 0.8, n_samples)
    
    # 3. Kinematics (Velocity)
    # Velocity is the derivative of position
    vel_y = np.gradient(pos_y, DT)
    
    # 4. Electrical Generation
    # Assuming a rectified generator where Voltage is proportional to absolute velocity
    # Max velocity is approx 70 * 2*pi*0.6 = 263 mm/s
    # We want max voltage around 6V
    voltage_ideal = np.abs(vel_y) * 0.023 
    voltage = voltage_ideal + np.random.normal(0, 0.2, n_samples)
    voltage = np.clip(voltage, 0, None) # No negative voltage
    
    # Current is proportional to voltage (Ohm's law for a fixed load)
    # We want max current around 500 mA, so Power peak is around 6V * 0.5A = 3.0W (Passes > 2.5W criteria)
    current_ideal = voltage * 85.0 
    current = current_ideal + np.random.normal(0, 5.0, n_samples)
    current = np.clip(current, 0, None)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Time_sec': t,
        'Vision_X_mm': pos_x,
        'Vision_Y_mm': pos_y,
        'Voltage_V': voltage,
        'Current_mA': current
    })
    
    return df

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate data
    df_dummy = generate_data()
    
    # Save to CSV
    # Using French/European locale formatting (; for separator, , for decimal)
    # to match what 'data_analysis.py' seems to expect based on configuration.
    print(f"[DUMMY DATA] Saving to {OUTPUT_PATH}...")
    df_dummy.to_csv(OUTPUT_PATH, sep=';', decimal=',', index=False)
    
    print("[DUMMY DATA] Success! You can now test the report generator by running:")
    print(f"             python generate_report.py {OUTPUT_PATH}")