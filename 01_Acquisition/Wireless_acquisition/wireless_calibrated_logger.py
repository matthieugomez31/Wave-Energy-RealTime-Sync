import cv2
import cv2.aruco as aruco
import numpy as np
import time
import math
import socket
import threading
from datetime import datetime
import os

# =========================================================================
# CONFIGURATION SECTION
# =========================================================================

# --- Wireless UDP Settings ---
UDP_IP = "0.0.0.0"   # Listen on all available network interfaces
UDP_PORT = 4210      # Must match the ESP32 port

# --- ArUco Marker IDs & Physics ---
REFERENCE_ID_1 = 10  # Origin point (0,0)
REFERENCE_ID_2 = 11  # Second point for scaling
MOBILE_ID = 12       # The marker on the moving object
KNOWN_DISTANCE_MM = 160.0 # Distance between REF_1 and REF_2

# --- Camera Settings ---
CAMERA_INDEX = 1 # 0 for internal, 1 for external

# --- Data Logging Setup ---
OUTPUT_DIR = "03_Data_Samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
FILENAME = os.path.join(OUTPUT_DIR, f"log_experience_wireless_{TIMESTAMP}.csv")

# =========================================================================
# WIRELESS LISTENER THREAD (UDP)
# =========================================================================
# Global variables shared between the UDP thread and the Main Camera Loop
latest_voltage = float('nan')
latest_current = float('nan')
last_packet_time = 0.0
stop_thread = False
data_lock = threading.Lock() # Prevents memory read/write collisions

def udp_listener():
    """ Background thread that constantly listens for ESP32 packets. """
    global latest_voltage, latest_current, last_packet_time
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0) # 1 second timeout to allow clean thread exit
    
    print(f"[WIFI] Listening for ESP32 data on port {UDP_PORT}...")
    
    while not stop_thread:
        try:
            data, addr = sock.recvfrom(1024) # Buffer size is 1024 bytes
            decoded_str = data.decode('utf-8').strip()
            parts = decoded_str.split(',')
            
            if len(parts) == 2:
                v = float(parts[0])
                i = float(parts[1])
                
                # Safely update globals
                with data_lock:
                    latest_voltage = v
                    latest_current = i
                    last_packet_time = time.time()
                    
        except socket.timeout:
            pass # Normal behavior, just loops again
        except Exception as e:
            print(f"[WIFI ERROR] {e}")

# =========================================================================
# MAIN CAMERA & TIME MASTER LOOP
# =========================================================================
def get_pixel_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    global stop_thread
    
    # 1. Start the Wi-Fi listener thread
    listener = threading.Thread(target=udp_listener, daemon=True)
    listener.start()

    # 2. Initialize Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        stop_thread = True
        return

    # ArUco Setup (Compatible with OpenCV 4.7+)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    # 3. Create CSV File and write header
    with open(FILENAME, mode='w') as f:
        # Using Semicolon to match European standards requested previously
        f.write("Time_sec;Pos_X_mm;Pos_Y_mm;Voltage_V;Current_mA\n")
    
    print(f"[LOG] Saving to {FILENAME}")
    print("[INFO] Press 'q' on the video window to stop and save.")

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        current_time_sec = time.time() - start_time
        
        # --- A. COMPUTER VISION (Kinematics) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        val_x_mm = -1.0
        val_y_mm = -1.0
        mm_per_pixel = -1.0

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            ids_flat = ids.flatten()

            # Calibration: Find scale if both reference markers are visible
            if REFERENCE_ID_1 in ids_flat and REFERENCE_ID_2 in ids_flat:
                idx1 = np.where(ids_flat == REFERENCE_ID_1)[0][0]
                idx2 = np.where(ids_flat == REFERENCE_ID_2)[0][0]
                
                center1 = np.mean(corners[idx1][0], axis=0)
                center2 = np.mean(corners[idx2][0], axis=0)
                
                pixel_dist = get_pixel_distance(center1, center2)
                if pixel_dist > 0:
                    mm_per_pixel = KNOWN_DISTANCE_MM / pixel_dist

                # Tracking: Find the mobile buoy marker
                if MOBILE_ID in ids_flat and mm_per_pixel > 0:
                    idx_mobile = np.where(ids_flat == MOBILE_ID)[0][0]
                    center_mobile = np.mean(corners[idx_mobile][0], axis=0)
                    
                    # Calculate position relative to Origin (REF 1)
                    dx_px = center_mobile[0] - center1[0]
                    dy_px = center1[1] - center_mobile[1] # Y axis inverted in image
                    
                    val_x_mm = dx_px * mm_per_pixel
                    val_y_mm = dy_px * mm_per_pixel

        # --- B. ELECTRICAL DATA ACQUISITION & WATCHDOG ---
        with data_lock:
            v_val = latest_voltage
            i_val = latest_current
            t_packet = last_packet_time

        # THE WATCHDOG: If no data received for 0.5s, the buoy is disconnected
        if time.time() - t_packet > 0.5:
            v_val = float('nan')
            i_val = float('nan')
            wifi_status = "DISCONNECTED"
            wifi_color = (0, 0, 255) # Red
        else:
            wifi_status = "CONNECTED"
            wifi_color = (0, 255, 0) # Green

        # Format correctly for logging (force NaN format if missing)
        log_x = f"{val_x_mm:.1f}" if val_x_mm != -1.0 else "NaN"
        log_y = f"{val_y_mm:.1f}" if val_y_mm != -1.0 else "NaN"
        log_v = f"{v_val:.2f}" if not math.isnan(v_val) else "NaN"
        log_i = f"{i_val:.1f}" if not math.isnan(i_val) else "NaN"

        # --- C. LOGGING ---
        # Replacing dots with commas for European CSV format compatibility (as in Dummy Data)
        log_line = f"{current_time_sec:.3f};{log_x};{log_y};{log_v};{log_i}\n".replace('.', ',')
        
        with open(FILENAME, mode='a') as f:
            f.write(log_line)

        # --- D. VISUAL FEEDBACK (GUI) ---
        cv2.putText(frame, f"Wi-Fi: {wifi_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, wifi_color, 2)
        
        if val_x_mm != -1.0:
            cv2.putText(frame, f"Pos: X={val_x_mm:.1f} Y={val_y_mm:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "TRACKING LOST", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if not math.isnan(v_val):
            cv2.putText(frame, f"Elec: {v_val:.2f}V | {i_val:.1f}mA", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "ELEC: NO DATA", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Wireless Calibrated Logger", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    stop_thread = True
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Experiment saved successfully to: {FILENAME}")

if __name__ == "__main__":
    main()