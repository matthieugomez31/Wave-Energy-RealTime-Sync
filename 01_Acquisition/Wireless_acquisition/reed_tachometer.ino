// Reed Switch Tachometer
// Wiring: Reed switch between Pin 2 and GND (uses internal pull-up)
//         One magnet per revolution on the axle
//
// Serial output: CSV  →  time_ms, rpm
// Open Serial Monitor at 115200 baud, copy/paste into Excel or any plotter.

// ── Config ─────────────────────────────────────────────────────────────────
const int           REED_PIN         = 2;     // Must be interrupt pin (2 or 3 on Uno)
const unsigned long DEBOUNCE_MS      = 15;    // Ignore re-triggers within this window
const unsigned long STALE_TIMEOUT_MS = 3000;  // Report 0 RPM after this long with no pulse
const unsigned long REPORT_MS        = 250;   // How often to print a reading (ms)
const int           AVG_SAMPLES      = 4;     // Rolling average over this many intervals
// ───────────────────────────────────────────────────────────────────────────

volatile unsigned long lastPulseTime  = 0;
volatile unsigned long intervals[AVG_SAMPLES];
volatile int           intervalIndex  = 0;
volatile int           sampleCount    = 0;

unsigned long lastReportTime = 0;

// ── ISR ────────────────────────────────────────────────────────────────────
void  onPulse() {
  unsigned long now = millis();
  unsigned long gap = now - lastPulseTime;

  if (gap < DEBOUNCE_MS) return;   // Debounce: discard chatter

  intervals[intervalIndex] = gap;
  intervalIndex = (intervalIndex + 1) % AVG_SAMPLES;
  if (sampleCount < AVG_SAMPLES) sampleCount++;

  lastPulseTime = now;
}

// ── Setup ──────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  pinMode(REED_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(REED_PIN), onPulse, FALLING);

  Serial.println(F("time_ms,rpm"));   // CSV header
}

// ── Loop ───────────────────────────────────────────────────────────────────
void loop() {
  unsigned long now = millis();

  if (now - lastReportTime < REPORT_MS) return;
  lastReportTime = now;

  // Snapshot volatile data with interrupts paused
  noInterrupts();
  unsigned long snapshotIntervals[AVG_SAMPLES];
  int           snapshotCount = sampleCount;
  unsigned long snapshotLast  = lastPulseTime;
  for (int i = 0; i < AVG_SAMPLES; i++) snapshotIntervals[i] = intervals[i];
  interrupts();

  float rpm = 0.0;

  bool signalPresent = (snapshotCount > 0) &&
                       ((now - snapshotLast) < STALE_TIMEOUT_MS);

  if (signalPresent) {
    // Average the captured intervals (only use filled slots)
    unsigned long sum = 0;
    for (int i = 0; i < snapshotCount; i++) sum += snapshotIntervals[i];
    float avgInterval_ms = (float)sum / snapshotCount;
    rpm = 60000.0f / avgInterval_ms;
  }

  Serial.print(now);
  Serial.print(',');
  Serial.println(rpm, 1);
}
