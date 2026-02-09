# Synchronized Data Acquisition for Wave Energy Converter

## 🌊 Project Overview
**Role:** Electronics & Control Lead (Capstone Project)
**Context:** Renewable Energy from Ocean Waves (Point Absorber Prototype)

This project implements a high-precision Data Acquisition System (DAQ) to validate the efficiency of a wave energy converter. The system synchronizes mechanical data (Computer Vision) with electrical telemetry (Voltage/Current) to generate accurate power curves.

## 🛠️ The Challenge: OS Latency
To measure the generator's efficiency ($\eta = P_{elec} / P_{mech}$), we needed to correlate the buoy's movement with the power output.
* **Problem:** Using Windows OS to timestamp high-frequency sensor data introduced variable latency (drift), making precise synchronization impossible.
* **Consequence:** Data skews rendered the mechanical-to-electrical efficiency models invalid.

## 💡 The Solution: Hardware "Time Master"
I designed a closed-loop architecture where the **ESP32 Microcontroller acts as the Time Master**, bypassing the OS scheduler entirely.

1.  **PC (Python):** Tracks markers via webcam and sends position $(X, Y)$ to the ESP32.
2.  **ESP32 (C++):** Waits for the position packet.
3.  **Hardware Sync:** Upon reception, the ESP32 *immediately* triggers the electrical sensor reading (INA219).
4.  **Data Return:** The ESP32 packages the synchronized tuple `(Timestamp, X, Y, V, I)` and sends it back to the PC for logging.

### System Architecture
![System Diagram](99_Assets/system_architecture.png)
*(Figure: Data flow ensuring <10ms synchronization delay)*

## ⚙️ Key Technical Features

### 1. Dynamic Calibration Algorithm
Hardcoding a pixel-to-mm ratio is error-prone due to camera vibrations. I implemented a **per-frame calibration** system using a 3-marker setup:
* **Reference Markers (ID 10 & 11):** Placed at a fixed known distance.
* **Algorithm:** The script recalculates the scale factor ($mm/px$) for every single frame based on the distance between Ref 1 and Ref 2.
* **Result:** Robust measurement accuracy ($\pm$ 0.5 mm) even if the camera focus or position shifts slightly.

### 2. Electrical Hardware Setup
* **Microcontroller:** ESP32-DevKitC (Dual-core, High speed).
* **Sensor:** INA219 (High-Side DC Current/Voltage) over I2C.
* **Power Stage:** Full-bridge rectifier connecting the generator to the load.

![Wiring Diagram](99_Assets/wiring_diagram.png)

## 💻 Tech Stack
* **Embedded:** C++ (Arduino Framework), Serial Communication (115200 baud).
* **Computer Vision:** Python, OpenCV (ArUco Dictionary), NumPy.
* **Protocols:** UART (Serial), I2C.

## 🚀 How to Run
*(Detailed operational guide available in `04_Reports/Report.pdf`)*

1.  **Hardware:** Connect ESP32 via Data USB. Ensure markers 10, 11, and 12 are visible.
2.  **Software Sequence:**
    * Close any open Serial Monitors (Arduino IDE) to free the COM port.
    * Run the script: `python calibrated_logger.py`
    * Wait 4 seconds for buffer cleaning.
3.  **Output:** Data is saved to `log_experience_YYYYMMDD.csv` for MATLAB/Excel analysis.

---
*Full Engineering Report available in the `04_Reports/` folder.*
