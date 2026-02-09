import cv2
import cv2.aruco as aruco
import numpy as np
import time
import serial
import math
from datetime import datetime

# =========================================================================
# CONFIGURATION SECTION
# =========================================================================

# --- Serial Communication Settings ---
# CRITICAL: Update this to match your new PC's COM port (e.g., 'COM5')
SERIAL_PORT = 'COM3' 
BAUD_RATE = 115200
SERIAL_TIMEOUT = 0.1 # Fast timeout for non-blocking reads

# --- ArUco Marker IDs ---
# Update these to match the printed markers you are using
REFERENCE_ID_1 = 10  # Origin point (0,0)
REFERENCE_ID_2 = 11  # Second point for scaling
MOBILE_ID = 12       # The marker on the moving object

# --- Physical Calibration ---
# The REAL distance in millimeters between center of REF_1 and center of REF_2
KNOWN_DISTANCE_MM = 160.0 

# --- Camera Settings ---
CAMERA_INDEX = 1 # 0 for internal webcam, 1 for external USB webcam

# =========================================================================

def get_pixel_distance(p1, p2):
    """Calculates Euclidean distance between two (x, y) points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    # --- 1. Setup Serial Connection (Robust Method) ---
    port_serie = None
    try:
        print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
        port_serie = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=SERIAL_TIMEOUT,
            rtscts=False # Critical for ESP32
        )
        
        # WAIT for ESP32 to reboot (The critical fix)
        print("Waiting 4 seconds for ESP32 to initialize...")
        time.sleep(4)
        
        # CLEAN the input buffer (Remove boot messages)
        print("Cleaning serial buffer...")
        port_serie.reset_input_buffer()
        
        print("Serial communication established.")

    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        print("Running in VIDEO ONLY mode (No data logging).")

    # --- 2. Setup ArUco Detector ---
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    aruco_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    # --- 3. Setup Video Capture ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}.")
        return
    
    # Optional: Set resolution (e.g., 1280x720)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # --- 4. Setup CSV Logging ---
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"log_experience_{timestamp_str}.csv"
    print(f"Data will be saved to: {log_filename}")

    with open(log_filename, 'w', newline='') as log_file:
        
        # Write CSV Header
        # Matches the format sent by ESP32
        header = "Timestamp_ms,Vision_X_mm,Vision_Y_mm,Voltage_V,Current_mA\n"
        log_file.write(header)
        
        print("\n--- STARTING ACQUISITION ---")
        print("Press 'q' to quit.")

        mm_per_pixel = 0.0 # Dynamic calibration ratio

        while True:
            # --- A. Image Capture ---
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect markers
            corners, ids, rejected = detector.detectMarkers(gray)
            
            # Dictionary to store centers: {ID: (x, y)}
            marker_centers = {}
            
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
                
                # Calculate centers for all found markers
                for i, marker_id in enumerate(ids):
                    c_x = int(np.mean(corners[i][0][:, 0]))
                    c_y = int(np.mean(corners[i][0][:, 1]))
                    marker_centers[marker_id[0]] = (c_x, c_y)
                    
                    # Visual debug: draw center
                    cv2.circle(frame, (c_x, c_y), 4, (0, 0, 255), -1)

            # --- B. Dynamic Calibration ---
            # We need both reference markers to calculate the ratio
            if REFERENCE_ID_1 in marker_centers and REFERENCE_ID_2 in marker_centers:
                ref1 = marker_centers[REFERENCE_ID_1]
                ref2 = marker_centers[REFERENCE_ID_2]
                
                pixel_dist = get_pixel_distance(ref1, ref2)
                
                if pixel_dist > 0:
                    mm_per_pixel = KNOWN_DISTANCE_MM / pixel_dist

            # --- C. Calculate Displacement ---
            # Default values (sent if marker not found)
            val_x_mm = -1.0
            val_y_mm = -1.0
            
            # We need the Mobile marker AND the Origin (Ref 1) AND a valid ratio
            if MOBILE_ID in marker_centers and REFERENCE_ID_1 in marker_centers and mm_per_pixel > 0:
                mobile = marker_centers[MOBILE_ID]
                origin = marker_centers[REFERENCE_ID_1]
                
                # Calculate distance in pixels
                dx_px = mobile[0] - origin[0]
                dy_px = mobile[1] - origin[1] # Y goes down in images, check orientation!
                
                # Convert to millimeters
                val_x_mm = dx_px * mm_per_pixel
                val_y_mm = dy_px * mm_per_pixel

            # --- D. Serial Communication (Send & Receive) ---
            if port_serie is not None and port_serie.is_open:
                try:
                    # 1. SEND Data to ESP32
                    # Format: "X.XX,Y.YY\r\n" (Using \r\n for compatibility)
                    data_str = f"{val_x_mm:.2f},{val_y_mm:.2f}\r\n"
                    port_serie.write(data_str.encode('utf-8'))
                    
                    # 2. READ Data from ESP32
                    # The ESP32 sends data at 100Hz. We might have multiple lines waiting.
                    # We read all of them to keep the buffer empty.
                    while port_serie.in_waiting > 0:
                        line = port_serie.readline().decode('utf-8', errors='ignore').strip()
                        
                        if line:
                            # Log to file
                            log_file.write(line + "\n")
                            
                            # Print to console (Monitoring)
                            print(f"LOG: {line} | Ratio: {mm_per_pixel:.3f}", end='\r')

                except Exception as e:
                    print(f"\nSerial Error: {e}")

            # --- E. Visual Feedback ---
            # Display info on screen
            cv2.putText(frame, f"Ratio: {mm_per_pixel:.3f} mm/px", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if val_x_mm != -1.0:
                cv2.putText(frame, f"Pos: X={val_x_mm:.1f} Y={val_y_mm:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "TRACKING LOST", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Calibrated Logger", frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # --- Cleanup ---
    print("\nStopping acquisition...")
    cap.release()
    cv2.destroyAllWindows()
    if port_serie is not None:
        port_serie.close()
    print(f"File saved: {log_filename}")

if __name__ == "__main__":
    main()