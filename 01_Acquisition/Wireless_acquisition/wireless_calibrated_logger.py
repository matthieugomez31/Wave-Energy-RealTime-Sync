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
UDP_IP = "0.0.0.0"   # Listen on all network interfaces
UDP_PORT = 4210      # Must match the ESP32 port

# --- ArUco Marker IDs & Physics ---
REFERENCE_ID_1 = 10  # Origin point (0,0)
REFERENCE_ID_2 = 11  # Second point for scaling
MOBILE_ID = 12       # The marker on the moving object
KNOWN_DISTANCE_MM = 160.0 # Physical distance between REF_1 and REF_2

# --- Camera Settings ---
CAMERA_INDEX = 1 # 0 for internal webcam, 1 for external USB webcam

# --- Data Logging Setup ---
OUTPUT_DIR = "03_Data_Samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
FILENAME = os.path.join(OUTPUT_DIR, f"log_experience_wireless_{TIMESTAMP}.csv")

# =========================================================================
# GLOBAL VARIABLES (Shared between the UDP Thread and the Camera)
# =========================================================================
latest_voltage = float('nan')
latest_current = float('nan')
latest_data_time = 0

# =========================================================================
# WIRELESS LISTENER THREAD (UDP)
# =========================================================================
def udp_listener():
    global latest_voltage, latest_current, latest_data_time
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0)
    
    print(f"[WIFI] Listening for ESP32 on UDP port {UDP_PORT}...")
    
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            message = data.decode('utf-8').strip()
            # Expecting format "Voltage,Current" (e.g., "5.12,120.5")
            parts = message.split(',')
            if len(parts) >= 2:
                latest_voltage = float(parts[0])
                latest_current = float(parts[1])
                latest_data_time = time.time()
        except socket.timeout:
            pass
        except Exception as e:
            # Ignore minor parsing errors
            pass

# =========================================================================
# MAIN LOOP (CAMERA)
# =========================================================================
def main():
    # 1. Start Wi-Fi reception in the background
    udp_thread = threading.Thread(target=udp_listener, daemon=True)
    udp_thread.start()

    # 2. Initialize the camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 3. ArUco Configuration (THE FIX IS HERE)
    # We specifically force the 6x6 dictionary to match your markers
    try:
        # OpenCV 4.7+ syntax
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        use_detector = True
    except AttributeError:
        # Older OpenCV syntax
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters_create()
        use_detector = False

    print("[SYSTEM] Starting Camera and Tracking...")
    print("[INFO]Press 'q' on the video window to stop and save.")
    start_time = time.time()

    # Create CSV file with headers
    with open(FILENAME, mode='w') as f:
        f.write("Time_sec;Vision_X_mm;Vision_Y_mm;Voltage_V;Current_mA\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read error.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        if use_detector:
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        val_x_mm = -1.0
        val_y_mm = -1.0
        mm_per_pixel = 0.0

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            ids = ids.flatten()

            # Search for specific IDs
            ref1_idx = np.where(ids == REFERENCE_ID_1)[0]
            ref2_idx = np.where(ids == REFERENCE_ID_2)[0]
            mob_idx = np.where(ids == MOBILE_ID)[0]

            # If both references are present, calculate the scale (mm/pixel)
            if len(ref1_idx) > 0 and len(ref2_idx) > 0:
                c1 = corners[ref1_idx[0]][0]
                c2 = corners[ref2_idx[0]][0]
                
                center1 = (int(np.mean(c1[:, 0])), int(np.mean(c1[:, 1])))
                center2 = (int(np.mean(c2[:, 0])), int(np.mean(c2[:, 1])))
                
                pixel_dist = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                if pixel_dist > 0:
                    mm_per_pixel = KNOWN_DISTANCE_MM / pixel_dist
                    
                cv2.line(frame, center1, center2, (0, 255, 255), 2)

                # If the scale is known and the mobile object is present, calculate its position
                if mm_per_pixel > 0 and len(mob_idx) > 0:
                    c_mob = corners[mob_idx[0]][0]
                    center_mob = (int(np.mean(c_mob[:, 0])), int(np.mean(c_mob[:, 1])))
                    
                    pixel_x = center_mob[0] - center1[0]
                    pixel_y = center_mob[1] - center1[1]

                    val_x_mm = pixel_x * mm_per_pixel
                    val_y_mm = pixel_y * mm_per_pixel
                    
                    cv2.line(frame, center1, center_mob, (255, 0, 255), 2)

        # --- DATA MANAGEMENT AND LOGGING ---
        current_time_sec = time.time() - start_time
        
        # If no Wi-Fi data received for > 2 seconds, consider it disconnected
        v_val = latest_voltage if (time.time() - latest_data_time < 2.0) else float('nan')
        i_val = latest_current if (time.time() - latest_data_time < 2.0) else float('nan')

        wifi_status = "OK" if not math.isnan(v_val) else "DISCONNECTED"
        wifi_color = (0, 255, 0) if wifi_status == "OK" else (0, 0, 255)

        log_x = f"{val_x_mm:.1f}" if val_x_mm != -1.0 else "NaN"
        log_y = f"{val_y_mm:.1f}" if val_y_mm != -1.0 else "NaN"
        log_v = f"{v_val:.2f}" if not math.isnan(v_val) else "NaN"
        log_i = f"{i_val:.1f}" if not math.isnan(i_val) else "NaN"

        # Write to CSV (Replacing dots with commas for European CSV format compatibility)
        log_line = f"{current_time_sec:.3f};{log_x};{log_y};{log_v};{log_i}\n".replace('.', ',')
        with open(FILENAME, mode='a') as f:
            f.write(log_line)

        # --- VISUAL FEEDBACK (GUI) ---
        cv2.putText(frame, f"Wi-Fi: {wifi_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, wifi_color, 2)
        
        if val_x_mm != -1.0:
            cv2.putText(frame, f"Pos: X={val_x_mm:.1f} Y={val_y_mm:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "TRACKING LOST", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if not math.isnan(v_val):
            cv2.putText(frame, f"V: {v_val:.2f} V | I: {i_val:.1f} mA", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Wireless Calibrated Logger", frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] Program stopped.")

if __name__ == "__main__":
    main()