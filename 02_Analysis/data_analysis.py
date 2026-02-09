import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Path to your file
# Updated to point to the correct folder in the repository structure
FILE_PATH = '03_Data_Samples/log_experience_20251125_163304.csv' 

# CSV Format Settings
# FOR YOUR CURRENT FILE (Legacy Excel format):
CSV_SEP = ';'       # Separator is semicolon
CSV_DECIMAL = ','   # Decimal is comma (e.g., 0,01)

# FOR FUTURE PYTHON FILES (Standard Universal format):
# CSV_SEP = ','     
# CSV_DECIMAL = '.' 

def load_and_inspect_data(filepath):
    """
    Loads the CSV file and prints basic info.
    """
    print(f"📂 Loading file: {filepath}...")

    try:
        # Load data with the specified configuration
        df = pd.read_csv(filepath, sep=CSV_SEP, decimal=CSV_DECIMAL)
        
        # Clean column names (remove spaces like ' Voltage_V')
        df.columns = [c.strip() for c in df.columns]
        
        print("✅ File loaded successfully!")
        print(f"📊 Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
        print("🔍 First 5 rows of raw data:")
        print(df.head())
        
        return df

    except FileNotFoundError:
        print("❌ Error: File not found. Check the path.")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def preprocess_data(df):
    """
    Basic cleaning before plotting.
    """
    # 1. Normalize Time (Start at t=0 seconds)
    # Convert ms to seconds and shift start to 0
    if 'Timestamp_ms' in df.columns:
        start_time = df['Timestamp_ms'].iloc[0]
        df['Time_sec'] = (df['Timestamp_ms'] - start_time) / 1000.0
    else:
        # Fallback if no timestamp: Create a fake time vector (assuming 100Hz)
        print("⚠️ Warning: No 'Timestamp_ms' column. Creating artificial time.")
        df['Time_sec'] = np.arange(len(df)) * 0.01

    # 2. Handle Lost Tracking (-1 values) for Y
    # Replace -1 with NaN (Not a Number) so Matplotlib ignores them instead of plotting 0
    if 'Vision_Y_mm' in df.columns:
        df['Vision_Y_Clean'] = df['Vision_Y_mm'].replace(-1, np.nan)
        df['Vision_Y_Clean'] = df['Vision_Y_Clean'].replace(-1.0, np.nan)
    
    # 3. Handle Lost Tracking for X (New)
    if 'Vision_X_mm' in df.columns:
        df['Vision_X_Clean'] = df['Vision_X_mm'].replace(-1, np.nan)
        df['Vision_X_Clean'] = df['Vision_X_Clean'].replace(-1.0, np.nan)

    # 4. Calculate Absolute Displacement (2D Magnitude)
    # Calculates sqrt(X^2 + Y^2). Useful to see total movement.
    if 'Vision_X_Clean' in df.columns and 'Vision_Y_Clean' in df.columns:
        # We assume origin (0,0) is meaningful (top-left of camera usually).
        # For analysis, we might want variation from start, but raw magnitude is good for inspection.
        df['Vision_Abs_Disp'] = np.sqrt(df['Vision_X_Clean']**2 + df['Vision_Y_Clean']**2)

    return df

def plot_raw_data(df):
    """
    Displays the raw data (Position X/Y/Abs, Voltage, Current).
    """
    plt.figure(figsize=(12, 12)) # Increased height for more plots

    # --- Subplot 1: Vision Y (Heave - Useful Motion) ---
    plt.subplot(4, 1, 1)
    if 'Vision_Y_Clean' in df.columns:
        plt.plot(df['Time_sec'], df['Vision_Y_Clean'], label='Y (Heave)', color='blue', marker='o', markersize=2, linestyle='None')
    plt.title('RAW DATA: Vertical Position (Y)')
    plt.ylabel('Y Position (mm)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # --- Subplot 2: Vision X & Absolute (Parasitic/Total Motion) ---
    plt.subplot(4, 1, 2)
    if 'Vision_X_Clean' in df.columns:
        plt.plot(df['Time_sec'], df['Vision_X_Clean'], label='X (Surge/Drift)', color='orange', alpha=0.7)
    if 'Vision_Abs_Disp' in df.columns:
        plt.plot(df['Time_sec'], df['Vision_Abs_Disp'], label='Absolute (2D)', color='purple', linestyle='--', alpha=0.5)
    plt.title('RAW DATA: Horizontal (X) & Absolute Displacement')
    plt.ylabel('Position (mm)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # --- Subplot 3: Voltage ---
    plt.subplot(4, 1, 3)
    if 'Voltage_V' in df.columns:
        plt.plot(df['Time_sec'], df['Voltage_V'], label='Voltage', color='green')
    plt.title('RAW DATA: Generator Voltage')
    plt.ylabel('Voltage (V)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # --- Subplot 4: Current ---
    plt.subplot(4, 1, 4)
    if 'Current_mA' in df.columns:
        plt.plot(df['Time_sec'], df['Current_mA'], label='Current', color='red')
    plt.title('RAW DATA: Generator Current')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Current (mA)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load
    df = load_and_inspect_data(FILE_PATH)
    
    if df is not None:
        # 2. Process
        df = preprocess_data(df)
        
        # 3. Visualize
        plot_raw_data(df)