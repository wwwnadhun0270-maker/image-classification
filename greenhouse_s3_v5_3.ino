/*
=====================================================
    🌿 SMART GREENHOUSE CONTROL SYSTEM 🌿
    ESP32-S3 WROOM-1 — Blynk IoT — v5.3

    TEC DRIVER  : K3878 MOSFET + 2x Relay
    EXHAUST FAN : Relay ON/OFF
    LDR         : Digital DO pin
    SOIL        : HW-103 Digital DO pin ← v5.3

    PIN MAP:
      DHT22        → GPIO 5   (digital)
      Soil HW-103  → GPIO 1   (DIGITAL DO) ← changed!
      LDR DO       → GPIO 2   (DIGITAL IN)
      MQ135 ADC1   → GPIO 4   (ADC1_CH3)
      OLED SDA     → GPIO 17
      OLED SCL     → GPIO 18
      K3878 Gate   → GPIO 6   (PWM)
      TEC Relay1   → GPIO 7   (Active LOW)
      TEC Relay2   → GPIO 8   (Active LOW)
      TEC Fan      → GPIO 9   (Digital)
      Exhaust Fan  → GPIO 10  (Relay ON/OFF)
      Mist Relay   → GPIO 11  (Active LOW)
      LED MOSFET   → GPIO 12  (PWM)
      Pump Relay   → GPIO 13  (Active LOW)
      Fert Relay   → GPIO 14  (Active LOW)
      MODE Button  → GPIO 38
      UP   Button  → GPIO 39
      DOWN Button  → GPIO 40
      SEL  Button  → GPIO 41

    HW-103 DO logic:
      DO = LOW  → Wet soil  → Pump OFF
      DO = HIGH → Dry soil  → Pump ON

    LDR DO logic:
      DO = LOW  → Dark      → LED ON
      DO = HIGH → Bright    → LED OFF

    Blynk V2 → Label "WET" or "DRY"
    Blynk V3 → Label "BRIGHT" or "DARK"

    v5.3 changes from v5.2:
      - Soil: analogRead → digitalRead (HW-103 DO)
      - soilMoisture/soilPercent → bool soilWet
      - autoControl: % threshold → wet/dry boolean
      - Pump: fixed 10s when dry (adjustable)
      - Blynk V2: sends "WET" or "DRY" text
      - OLED: soil shows WET/DRY
      - setup(): pinMode(SOIL_PIN, INPUT)
      - No map() calibration needed!
=====================================================
*/

// ── BLYNK ─────────────────────────────────────────
#define BLYNK_TEMPLATE_ID    "TMPL6MXqqt7Xt"
#define BLYNK_TEMPLATE_NAME  "Smart Green House"
#define BLYNK_AUTH_TOKEN     "ZV0fj5B5H2TFWimdDXonwLQNLcqM7BhN"
#define BLYNK_PRINT Serial

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <DHT.h>
#include <WiFi.h>
#include <BlynkSimpleEsp32.h>

// ── WiFi ──────────────────────────────────────────
char ssid[] = "HUAWEI Y7a";
char pass[] = "shamikaab";

// ── OLED ──────────────────────────────────────────
#define SCREEN_WIDTH   128
#define SCREEN_HEIGHT   64
#define OLED_ADDR     0x3C
#define I2C_SDA         17
#define I2C_SCL         18
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// ══════════════════════════════════════════════════
//  PIN DEFINITIONS — v5.3
// ══════════════════════════════════════════════════

// Sensors
#define DHT_PIN          5
#define SOIL_PIN         1    // DIGITAL IN (HW-103 DO) ← v5.3
#define LDR_PIN          2    // DIGITAL IN (LDR DO)
#define MQ135_PIN        4    // ADC1_CH3

// K3878 TEC speed (PWM)
#define TEC_MOSFET_PIN   6

// TEC polarity relays (Active LOW)
#define TEC_RELAY1_PIN   7
#define TEC_RELAY2_PIN   8

// TEC heatsink fan
#define TEC_FAN_PIN      9

// Exhaust fan relay (Active LOW — ON/OFF only)
#define EXHAUST_FAN_PIN 10

// Actuator relays (Active LOW)
#define MIST_PIN        11
#define LED_PIN         12    // MOSFET PWM
#define PUMP_PIN        13
#define FERT_PIN        14

// Buttons (INPUT_PULLUP)
#define MODE_BTN_PIN    38
#define UP_BTN_PIN      39
#define DOWN_BTN_PIN    40
#define SELECT_BTN_PIN  41

// ── PWM ───────────────────────────────────────────
#define PWM_FREQ  1000
#define PWM_BITS     8

// ── DHT ───────────────────────────────────────────
#define DHT_TYPE DHT22
DHT dht(DHT_PIN, DHT_TYPE);

// ── Blynk Virtual Pins ────────────────────────────
#define VPIN_TEMP      V0
#define VPIN_HUMIDITY  V1
#define VPIN_SOIL      V2   // Label: "WET" or "DRY"
#define VPIN_LIGHT     V3   // Label: "BRIGHT" or "DARK"
#define VPIN_AIR       V4
#define VPIN_TEC_MODE  V5
#define VPIN_TEC_SPEED V6
#define VPIN_PUMP      V7
#define VPIN_MIST      V8
#define VPIN_LED       V9
#define VPIN_FERT      V10
#define VPIN_FAN       V11  // Switch 0/1
#define VPIN_AUTO_MODE V12
#define VPIN_STATUS    V15

// ── Sensor data ───────────────────────────────────
float temperature = 0.0;
float humidity    = 0.0;
int   airQuality  = 0;
bool  soilWet     = false;  // HW-103: true=wet, false=dry
bool  isDark      = false;  // LDR:    true=dark, false=bright

// ── System state ──────────────────────────────────
bool autoMode      = true;
int  tecMode       = 0;    // 0=OFF 1=COOL 2=HEAT
int  tecSpeed      = 200;
int  pumpDuration  = 10;   // seconds (manual mode)
int  mistDuration  = 0;
int  ledBrightness = 0;
int  fertDuration  = 0;
bool fanOn         = false;

// ── Auto settings ─────────────────────────────────
#define LED_AUTO_BRIGHTNESS  200  // 0-255 adjust as needed
#define PUMP_DRY_SECONDS      10  // pump ON duration when dry

// ── Auto thresholds ───────────────────────────────
const float COOL_THRESH   = 30.0;
const float HEAT_THRESH   = 20.0;
const float HYSTERESIS    =  2.0;
const int   AIR_THRESH    =  800;
const float HUM_THRESH    = 85.0;
const float MIST_THRESH   = 50.0;  // humidity below → mist

// ── Non-blocking timer ────────────────────────────
struct DeviceTimer {
  bool          active    = false;
  unsigned long startTime = 0;
  unsigned long duration  = 0;
  int           pin       = -1;
  int           vpin      = -1;
};
DeviceTimer pumpTimer, mistTimer, fertTimer;

// ── Timing ────────────────────────────────────────
unsigned long tSensor  = 0, tDisplay = 0, tBlynk  = 0;
unsigned long tPumpCD  = 0, tMistCD  = 0, tFertCD = 0;
unsigned long tBtn     = 0;

const unsigned long T_SENSOR   =   2000UL;
const unsigned long T_DISPLAY  =    500UL;
const unsigned long T_BLYNK    =   3000UL;
const unsigned long COOLDOWN   = 300000UL;  // 5 min
const unsigned long FERT_INT   = 600000UL;  // 10 min
const int           DEBOUNCE   =    200;

// ── Alert flags ───────────────────────────────────
bool altTemp = false, altSoil = false, altAir = false;

// ── OLED menu ─────────────────────────────────────
int  menuIdx = 0;
bool inMenu  = false;
const int MENU_N = 8;
const char* menuItems[] = {
  "TEC Mode","TEC Speed","Water Pump",
  "Mist Maker","Grow LED","Fertilizer",
  "Exhaust Fan","Back to Auto"
};


// ══════════════════════════════════════════════════
//  EXHAUST FAN — RELAY ON/OFF
// ══════════════════════════════════════════════════

void setExhaustFan(bool on) {
  fanOn = on;
  digitalWrite(EXHAUST_FAN_PIN, on ? LOW : HIGH);
  Blynk.virtualWrite(VPIN_FAN, on ? 1 : 0);
}


// ══════════════════════════════════════════════════
//  K3878 + RELAY TEC CONTROL
// ══════════════════════════════════════════════════

void controlTEC(int mode, int speed) {
  // Safety: cut PWM before relay switch
  ledcWrite(TEC_MOSFET_PIN, 0);
  delay(10);

  switch (mode) {
    case 0:  // OFF
      digitalWrite(TEC_RELAY1_PIN, HIGH);
      digitalWrite(TEC_RELAY2_PIN, HIGH);
      digitalWrite(TEC_FAN_PIN,    LOW);
      break;
    case 1:  // COOL — NC path (forward)
      digitalWrite(TEC_RELAY1_PIN, HIGH);
      digitalWrite(TEC_RELAY2_PIN, HIGH);
      delay(5);
      ledcWrite(TEC_MOSFET_PIN, speed);
      digitalWrite(TEC_FAN_PIN, HIGH);
      break;
    case 2:  // HEAT — NO path (reverse)
      digitalWrite(TEC_RELAY1_PIN, LOW);
      digitalWrite(TEC_RELAY2_PIN, LOW);
      delay(5);
      ledcWrite(TEC_MOSFET_PIN, speed);
      digitalWrite(TEC_FAN_PIN, HIGH);
      break;
  }
  tecMode = mode;
}


// ══════════════════════════════════════════════════
//  TIMER HELPERS
// ══════════════════════════════════════════════════

void startTimer(DeviceTimer &t, int pin,
                unsigned long ms, int vp = -1) {
  if (!t.active) {
    t.pin = pin; t.duration = ms;
    t.startTime = millis();
    t.active = true; t.vpin = vp;
    digitalWrite(pin, LOW);   // Active LOW = ON
  }
}

void tickTimer(DeviceTimer &t) {
  if (t.active && millis() - t.startTime >= t.duration) {
    digitalWrite(t.pin, HIGH); // OFF
    t.active = false;
    if (t.vpin >= 0) Blynk.virtualWrite(t.vpin, 0);
  }
}

void allOFF() {
  controlTEC(0, 0);
  setExhaustFan(false);
  ledcWrite(LED_PIN, 0);
  digitalWrite(PUMP_PIN, HIGH);
  digitalWrite(MIST_PIN, HIGH);
  digitalWrite(FERT_PIN, HIGH);
  pumpTimer.active = false;
  mistTimer.active = false;
  fertTimer.active = false;
}


// ══════════════════════════════════════════════════
//  SENSORS — v5.3 ALL digital except MQ135
// ══════════════════════════════════════════════════

void readSensors() {
  // DHT22 — temp + humidity
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  if (!isnan(t) && t > -40 && t < 80)   temperature = t;
  if (!isnan(h) && h >= 0  && h <= 100) humidity    = h;

  // HW-103 DO — Wet/Dry digital ← v5.3
  // HW-103: HIGH = Dry soil · LOW = Wet soil
  soilWet = (digitalRead(SOIL_PIN) == LOW);

  // LDR DO — Bright/Dark digital
  // LDR:    HIGH = Bright  · LOW = Dark
  isDark = (digitalRead(LDR_PIN) == LOW);

  // MQ135 — Air quality (only ADC pin left)
  airQuality = analogRead(MQ135_PIN);
}


// ══════════════════════════════════════════════════
//  BLYNK PUSH — v5.3
// ══════════════════════════════════════════════════

void pushBlynk() {
  if (!Blynk.connected()) return;

  Blynk.virtualWrite(VPIN_TEMP,     temperature);
  Blynk.virtualWrite(VPIN_HUMIDITY, humidity);
  Blynk.virtualWrite(VPIN_AIR,      airQuality);

  // V2 — Soil: "WET" or "DRY" ← v5.3
  Blynk.virtualWrite(VPIN_SOIL,  soilWet ? "WET" : "DRY");

  // V3 — Light: "BRIGHT" or "DARK"
  Blynk.virtualWrite(VPIN_LIGHT, isDark ? "DARK" : "BRIGHT");

  // Status string
  String st = autoMode ? "AUTO | " : "MANUAL | ";
  st += "T:" + String(temperature, 1) + "C";
  st += " H:" + String(humidity, 0) + "%";
  st += " SOIL:" + String(soilWet ? "WET" : "DRY");
  st += " TEC:" + String(tecMode==0?"OFF":tecMode==1?"COOL":"HEAT");
  Blynk.virtualWrite(VPIN_STATUS, st);

  // ── Push Alerts ───────────────────────────────────
  if (temperature > 38.0 && !altTemp) {
    Blynk.logEvent("high_temp",
      String("Temp: ") + temperature + "C!");
    altTemp = true;
  } else if (temperature < 35.0) altTemp = false;

  // Soil alert — dry ← v5.3
  if (!soilWet && !altSoil) {
    Blynk.logEvent("dry_soil", "Soil is DRY! Watering needed.");
    altSoil = true;
  } else if (soilWet) altSoil = false;

  if (airQuality > 1000 && !altAir) {
    Blynk.logEvent("bad_air",
      String("Air: ") + airQuality);
    altAir = true;
  } else if (airQuality < 900) altAir = false;
}


// ══════════════════════════════════════════════════
//  BLYNK APP → ESP32-S3
// ══════════════════════════════════════════════════

BLYNK_WRITE(V12) {
  autoMode = (param.asInt() == 1);
  inMenu   = !autoMode;
  if (autoMode) allOFF();
}
BLYNK_WRITE(V5) {
  if (!autoMode) controlTEC(param.asInt(), tecSpeed);
}
BLYNK_WRITE(V6) {
  if (!autoMode) {
    tecSpeed = param.asInt();
    if (tecMode > 0) controlTEC(tecMode, tecSpeed);
  }
}
BLYNK_WRITE(V7) {
  // Manual pump — slider = seconds
  if (!autoMode) {
    pumpDuration = param.asInt();
    if (pumpDuration == 0) {
      digitalWrite(PUMP_PIN, HIGH);
      pumpTimer.active = false;
    } else {
      startTimer(pumpTimer, PUMP_PIN,
                 pumpDuration * 1000UL, VPIN_PUMP);
    }
  }
}
BLYNK_WRITE(V8) {
  if (!autoMode) {
    mistDuration = param.asInt();
    if (mistDuration == 0) {
      digitalWrite(MIST_PIN, HIGH);
      mistTimer.active = false;
    } else {
      startTimer(mistTimer, MIST_PIN,
                 mistDuration * 1000UL, VPIN_MIST);
    }
  }
}
BLYNK_WRITE(V9) {
  if (!autoMode) {
    ledBrightness = param.asInt();
    ledcWrite(LED_PIN, ledBrightness);
  }
}
BLYNK_WRITE(V10) {
  if (!autoMode) {
    fertDuration = param.asInt();
    if (fertDuration == 0) {
      digitalWrite(FERT_PIN, HIGH);
      fertTimer.active = false;
    } else {
      startTimer(fertTimer, FERT_PIN,
                 fertDuration * 1000UL, VPIN_FERT);
    }
  }
}
BLYNK_WRITE(V11) {
  if (!autoMode) setExhaustFan(param.asInt() == 1);
}

BLYNK_CONNECTED() { Blynk.syncAll(); }


// ══════════════════════════════════════════════════
//  AUTO CONTROL — v5.3
// ══════════════════════════════════════════════════

void autoControl() {
  if (!autoMode) return;

  // ── TEC climate (hysteresis) ──────────────────────
  if (temperature > COOL_THRESH + HYSTERESIS) {
    controlTEC(1, 255);
    Blynk.virtualWrite(VPIN_TEC_MODE,  1);
    Blynk.virtualWrite(VPIN_TEC_SPEED, 255);
  } else if (temperature < HEAT_THRESH - HYSTERESIS) {
    controlTEC(2, 180);
    Blynk.virtualWrite(VPIN_TEC_MODE,  2);
    Blynk.virtualWrite(VPIN_TEC_SPEED, 180);
  } else if (temperature >= HEAT_THRESH &&
             temperature <= COOL_THRESH) {
    controlTEC(0, 0);
    Blynk.virtualWrite(VPIN_TEC_MODE,  0);
    Blynk.virtualWrite(VPIN_TEC_SPEED, 0);
  }

  // ── Exhaust fan ───────────────────────────────────
  if (airQuality > AIR_THRESH || humidity > HUM_THRESH) {
    setExhaustFan(true);
  } else {
    setExhaustFan(false);
  }

  // ── Grow LED — LDR digital ────────────────────────
  if (isDark) {
    ledcWrite(LED_PIN, LED_AUTO_BRIGHTNESS);
    Blynk.virtualWrite(VPIN_LED, LED_AUTO_BRIGHTNESS);
  } else {
    ledcWrite(LED_PIN, 0);
    Blynk.virtualWrite(VPIN_LED, 0);
  }

  // ── Water Pump — HW-103 digital ← v5.3 ───────────
  // soilWet = false (DRY) → run pump
  // soilWet = true  (WET) → skip
  if (!soilWet &&
      !pumpTimer.active &&
      millis() - tPumpCD > COOLDOWN) {
    startTimer(pumpTimer, PUMP_PIN,
               PUMP_DRY_SECONDS * 1000UL, VPIN_PUMP);
    tPumpCD = millis();
    Blynk.virtualWrite(VPIN_PUMP, PUMP_DRY_SECONDS);
  }
  if (!pumpTimer.active) Blynk.virtualWrite(VPIN_PUMP, 0);

  // ── Mist — humidity based ─────────────────────────
  if (humidity < MIST_THRESH &&
      soilWet &&               // only mist if soil wet enough
      !mistTimer.active &&
      millis() - tMistCD > COOLDOWN) {
    int s = constrain(
      map((int)humidity, 20, (int)MIST_THRESH, 15, 3), 3, 15);
    startTimer(mistTimer, MIST_PIN,
               s * 1000UL, VPIN_MIST);
    tMistCD = millis();
    Blynk.virtualWrite(VPIN_MIST, s);
  }
  if (!mistTimer.active) Blynk.virtualWrite(VPIN_MIST, 0);

  // ── Fertilizer — fixed schedule ───────────────────
  if (!fertTimer.active &&
      millis() - tFertCD > FERT_INT) {
    startTimer(fertTimer, FERT_PIN, 4000UL, VPIN_FERT);
    tFertCD = millis();
    Blynk.virtualWrite(VPIN_FERT, 4);
  }
  if (!fertTimer.active) Blynk.virtualWrite(VPIN_FERT, 0);
}


// ══════════════════════════════════════════════════
//  OLED DISPLAY — v5.3
// ══════════════════════════════════════════════════

void updateDisplay() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  display.setCursor(0, 0);
  display.println("* Greenhouse S3 v5*");
  display.drawLine(0, 9, 128, 9, SSD1306_WHITE);

  if (inMenu) {
    display.setCursor(0, 12);
    display.println("[MANUAL MODE]");
    display.drawLine(0, 21, 128, 21, SSD1306_WHITE);
    display.setCursor(0, 25);
    display.print("> ");
    display.println(menuItems[menuIdx]);
    display.setCursor(0, 42);
    switch (menuIdx) {
      case 0:
        display.print("TEC: ");
        display.print(tecMode==0?"OFF":tecMode==1?"COOL":"HEAT");
        break;
      case 1:
        display.print("Speed: ");
        display.print(tecSpeed);
        break;
      case 2:
        display.print("Pump: ");
        display.print(pumpTimer.active ?
          String(pumpDuration)+"s" : "OFF");
        break;
      case 3:
        display.print("Mist: ");
        display.print(mistTimer.active ?
          String(mistDuration)+"s" : "OFF");
        break;
      case 4:
        display.print("LED: ");
        display.print(ledBrightness);
        break;
      case 5:
        display.print("Fert: ");
        display.print(fertTimer.active ?
          String(fertDuration)+"s" : "OFF");
        break;
      case 6:
        display.print("Fan: ");
        display.print(fanOn ? "ON" : "OFF");
        break;
      default:
        display.print("SELECT = confirm");
        break;
    }
  } else {
    display.setCursor(0, 12);
    display.print("Mode: ");
    display.println(autoMode ? "AUTO" : "MANUAL");

    display.setCursor(0, 22);
    display.print("T:"); display.print(temperature, 1);
    display.print("C H:"); display.print(humidity, 0);
    display.print("%");

    display.setCursor(0, 32);
    // Soil → WET/DRY  Light → DARK/BRGT ← v5.3
    display.print("Soil:");
    display.print(soilWet ? "WET " : "DRY ");
    display.print("L:");
    display.print(isDark ? "DARK" : "BRGT");

    display.setCursor(0, 42);
    display.print("Air:"); display.print(airQuality);
    display.print(" Fan:");
    display.print(fanOn ? "ON" : "OFF");

    display.setCursor(0, 52);
    display.print("TEC:");
    display.print(tecMode==0?"OFF":tecMode==1?"CL":"HT");
    display.print(" Blynk:");
    display.print(Blynk.connected() ? "OK" : "X");
  }
  display.display();
}


// ══════════════════════════════════════════════════
//  BUTTONS
// ══════════════════════════════════════════════════

void executeMenu() {
  switch (menuIdx) {
    case 0:
      tecMode = (tecMode + 1) % 3;
      controlTEC(tecMode, tecSpeed);
      Blynk.virtualWrite(VPIN_TEC_MODE, tecMode);
      break;
    case 1:
      tecSpeed = (tecSpeed >= 255) ? 50 : tecSpeed + 50;
      if (tecMode > 0) controlTEC(tecMode, tecSpeed);
      Blynk.virtualWrite(VPIN_TEC_SPEED, tecSpeed);
      break;
    case 2:
      // Manual pump toggle 0→5→10→15→0
      pumpDuration = (pumpDuration >= 15) ? 0 : pumpDuration + 5;
      if (pumpDuration == 0) {
        digitalWrite(PUMP_PIN, HIGH);
        pumpTimer.active = false;
      } else {
        startTimer(pumpTimer, PUMP_PIN,
                   pumpDuration * 1000UL, VPIN_PUMP);
      }
      Blynk.virtualWrite(VPIN_PUMP, pumpDuration);
      break;
    case 3:
      mistDuration = (mistDuration >= 10) ? 0 : mistDuration + 5;
      if (mistDuration == 0) {
        digitalWrite(MIST_PIN, HIGH);
        mistTimer.active = false;
      } else {
        startTimer(mistTimer, MIST_PIN,
                   mistDuration * 1000UL, VPIN_MIST);
      }
      Blynk.virtualWrite(VPIN_MIST, mistDuration);
      break;
    case 4:
      ledBrightness = (ledBrightness >= 255) ? 0 : ledBrightness + 85;
      ledcWrite(LED_PIN, ledBrightness);
      Blynk.virtualWrite(VPIN_LED, ledBrightness);
      break;
    case 5:
      fertDuration = (fertDuration >= 10) ? 0 : fertDuration + 5;
      if (fertDuration == 0) {
        digitalWrite(FERT_PIN, HIGH);
        fertTimer.active = false;
      } else {
        startTimer(fertTimer, FERT_PIN,
                   fertDuration * 1000UL, VPIN_FERT);
      }
      Blynk.virtualWrite(VPIN_FERT, fertDuration);
      break;
    case 6:
      setExhaustFan(!fanOn);
      break;
    case 7:
      autoMode = true; inMenu = false;
      Blynk.virtualWrite(VPIN_AUTO_MODE, 1);
      allOFF();
      break;
  }
}

void checkButtons() {
  if (millis() - tBtn < DEBOUNCE) return;

  if (digitalRead(MODE_BTN_PIN) == LOW) {
    autoMode = !autoMode;
    inMenu   = !autoMode;
    Blynk.virtualWrite(VPIN_AUTO_MODE, autoMode ? 1 : 0);
    if (autoMode) allOFF();
    tBtn = millis();
    return;
  }
  if (!autoMode) {
    if (digitalRead(UP_BTN_PIN) == LOW) {
      menuIdx = (menuIdx - 1 + MENU_N) % MENU_N;
      tBtn = millis();
    }
    if (digitalRead(DOWN_BTN_PIN) == LOW) {
      menuIdx = (menuIdx + 1) % MENU_N;
      tBtn = millis();
    }
    if (digitalRead(SELECT_BTN_PIN) == LOW) {
      executeMenu();
      tBtn = millis();
    }
  }
}


// ══════════════════════════════════════════════════
//  SETUP
// ══════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("=== Greenhouse S3 v5.3 — HW-103 Digital ===");

  Wire.begin(I2C_SDA, I2C_SCL);

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("OLED FAILED!"); while (1);
  }
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Greenhouse S3 v5.3");
  display.println("Soil: HW-103 Digital");
  display.println("Starting...");
  display.display();

  dht.begin();

  // ── LEDC — TEC + LED only ─────────────────────────
  ledcAttach(TEC_MOSFET_PIN, PWM_FREQ, PWM_BITS);
  ledcAttach(LED_PIN,         PWM_FREQ, PWM_BITS);

  // Output pins
  pinMode(TEC_RELAY1_PIN,  OUTPUT);
  pinMode(TEC_RELAY2_PIN,  OUTPUT);
  pinMode(TEC_FAN_PIN,     OUTPUT);
  pinMode(EXHAUST_FAN_PIN, OUTPUT);
  pinMode(MIST_PIN,        OUTPUT);
  pinMode(PUMP_PIN,        OUTPUT);
  pinMode(FERT_PIN,        OUTPUT);

  // Input pins — all digital now ← v5.3
  pinMode(SOIL_PIN,       INPUT);   // HW-103 DO
  pinMode(LDR_PIN,        INPUT);   // LDR DO
  pinMode(MODE_BTN_PIN,   INPUT_PULLUP);
  pinMode(UP_BTN_PIN,     INPUT_PULLUP);
  pinMode(DOWN_BTN_PIN,   INPUT_PULLUP);
  pinMode(SELECT_BTN_PIN, INPUT_PULLUP);

  // ── Initial states — ALL OFF ──────────────────────
  ledcWrite(TEC_MOSFET_PIN,  0);
  ledcWrite(LED_PIN,          0);
  digitalWrite(TEC_RELAY1_PIN,  HIGH);
  digitalWrite(TEC_RELAY2_PIN,  HIGH);
  digitalWrite(TEC_FAN_PIN,     LOW);
  digitalWrite(EXHAUST_FAN_PIN, HIGH);
  digitalWrite(MIST_PIN,        HIGH);
  digitalWrite(PUMP_PIN,        HIGH);
  digitalWrite(FERT_PIN,        HIGH);

  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);

  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("WiFi  : OK");
  display.println(WiFi.localIP().toString());
  display.println("Blynk : OK");
  display.println("HW103 : Digital");
  display.display();
  delay(1500);

  readSensors();
  Serial.println("=== Ready ===");
}


// ══════════════════════════════════════════════════
//  LOOP
// ══════════════════════════════════════════════════

void loop() {
  Blynk.run();

  unsigned long now = millis();

  if (now - tSensor  >= T_SENSOR)  { readSensors();   tSensor  = now; }
  if (now - tDisplay >= T_DISPLAY)  { updateDisplay(); tDisplay = now; }
  if (now - tBlynk   >= T_BLYNK)   { pushBlynk();     tBlynk   = now; }

  tickTimer(pumpTimer);
  tickTimer(mistTimer);
  tickTimer(fertTimer);

  checkButtons();
  if (autoMode) autoControl();
}
