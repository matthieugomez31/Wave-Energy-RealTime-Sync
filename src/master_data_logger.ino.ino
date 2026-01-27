/*
 * =======================================================
 * CODE FOR STEP 4 (v2.1): The "Master Data Logger"
 * =======================================================
 * VERSION 2.1: Fix for older INA219 library versions.
 * - Handles FLOATING POINT (decimal) values from Python.
 * - Acts as the TIME MASTER (100Hz).
 * - Listens for (X_mm, Y_mm) coordinates.
 * - Reads INA219 sensor.
 * - Combines and sends log line back to PC.
 */

// --- 1. Libraries ---
#include <Wire.h>
#include <Adafruit_INA219.h>

// --- 2. Globals ---
Adafruit_INA219 ina219;
float busVoltage = 0.0;
float current_mA = 0.0;

String incomingData = "";
float vision_X_mm = -1.0;
float vision_Y_mm = -1.0;

const int SAMPLING_RATE_MS = 10;
unsigned long lastSampleTime = 0;

void setup() {

  // --- NOUVELLE SECTION: VIDER LE BUFFER SÉRIE ---
  // Attend un moment que l'ESP32 soit prêt, puis vide tout ce qu'il a déjà reçu.
  delay(2000); 
  while (Serial.available()) {
    Serial.read(); // Lit et jette les données
  }
  
  // --- 3. Start Serial Port ---
  Serial.begin(115200);
  Serial.println("ESP32 Master Logger (v2.1-float) Initialized. Sending data at 100Hz.");
  Serial.println("Timestamp(ms),Vision_X(mm),Vision_Y(mm),Voltage(V),Current(mA)");

  // --- 4. Start Electrical Sensor (INA219) ---
  
  // --- THIS IS THE FIX ---
  // Step 1: Initialize the I2C bus (Wire) on the correct pins
  Wire.begin(21, 22); // (SDA = 21, SCL = 22)

  // Step 2: Initialize the sensor (with 1 argument)
  if (!ina219.begin(&Wire)) {
    Serial.println("Error: Failed to find INA219. Check wiring.");
  } else {
    Serial.println("INA219 sensor found.");
  }
  // --- END OF FIX ---
  
  lastSampleTime = millis();
}

void loop() {
  
  // --- 5. Check for new data from Python ---
  if (Serial.available() > 0) {
    incomingData = Serial.readStringUntil('\n');
    incomingData.trim();
    
    // --- LIGNE DE DÉBOGAGE À AJOUTER ---
    Serial.println("RX: " + incomingData); 
    // ------------------------------------
    
    int commaIndex = incomingData.indexOf(',');
    if (commaIndex != -1) {
      vision_X_mm = incomingData.substring(0, commaIndex).toFloat();
      vision_Y_mm = incomingData.substring(commaIndex + 1).toFloat();
    }
  }
  
  // --- 6. Time Master Loop ---
  if (millis() - lastSampleTime >= SAMPLING_RATE_MS) {
    lastSampleTime = millis(); // Reset the timer
    
    // --- 7. Read Local Sensors (Electrical) ---
    
    // This is now active
    busVoltage = ina219.getBusVoltage_V();
    current_mA = ina219.getCurrent_mA();
    
    // --- 8. Assemble and Send the Master Log Line ---
    String log_line = String(lastSampleTime) + "," +
                      String(vision_X_mm) + "," +
                      String(vision_Y_mm) + "," +
                      String(busVoltage) + "," +
                      String(current_mA);
                      
    Serial.println(log_line);
  }
}