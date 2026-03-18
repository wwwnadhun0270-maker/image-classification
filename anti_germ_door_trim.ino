/*
  =============================================
   ANTI-GERM DOOR SYSTEM - Arduino Nano
   v2 - Dual UV LEDs + Speed Control
  =============================================

  WIRING:

  - Servo Motor 1  --> Pin D9
  - Servo Motor 2  --> Pin D10
  - UV LED 1       --> Pin D6  (with 100Ω resistor)
  - UV LED 2       --> Pin D5  (with 100Ω resistor)
  - Push Button    --> Pin D2  (other leg to GND)

  =============================================
*/

#include <Servo.h>

// ---- Pin Definitions ----
const int SERVO1_PIN = 9;
const int SERVO2_PIN = 10;
const int UV_LED1    = 6;
const int UV_LED2    = 5;
const int BUTTON_PIN = 2;

// ---- SPEED CONTROL ----
const int SERVO1_SPEED = 15;
const int SERVO2_SPEED = 15;

// ---- Target Angle ----
const int TARGET_ANGLE = 90;

// ---- Servo Trim (adjust 0-point if motor arm is slightly off) ----
// Positive = shift arm forward, Negative = shift arm backward
// Example: if Servo 1 is 5° off, set SERVO1_TRIM = 5 or -5
const int SERVO1_TRIM = 0;
const int SERVO2_TRIM = 0;

// ---- Calculated Start Positions (do not edit) ----
const int SERVO1_START = 180 + SERVO1_TRIM;
const int SERVO2_START = 0   + SERVO2_TRIM;
const int SERVO1_MID   = TARGET_ANGLE + SERVO1_TRIM;
const int SERVO2_MID   = TARGET_ANGLE + SERVO2_TRIM;

// ---- Objects ----
Servo servo1;
Servo servo2;

bool isRunning = false;


// =============================================
// SMOOTH SERVO MOVE FUNCTION
// =============================================
void smoothMove(Servo &servo, int fromAngle, int toAngle, int speedMs) {

  if (fromAngle < toAngle) {

    for (int angle = fromAngle; angle <= toAngle; angle++) {
      servo.write(angle);
      delay(speedMs);
    }

  } else {

    for (int angle = fromAngle; angle >= toAngle; angle--) {
      servo.write(angle);
      delay(speedMs);
    }

  }
}


// =============================================

void setup() {

  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);

  pinMode(UV_LED1, OUTPUT);
  pinMode(UV_LED2, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  // Start positions (with trim applied)
  servo1.write(SERVO1_START);
  servo2.write(SERVO2_START);

  digitalWrite(UV_LED1, LOW);
  digitalWrite(UV_LED2, LOW);

  Serial.begin(9600);
  Serial.println("Anti-Germ System Ready...");
  Serial.print("Servo 1 start: "); Serial.println(SERVO1_START);
  Serial.print("Servo 2 start: "); Serial.println(SERVO2_START);
}


// =============================================

void loop() {

  if (digitalRead(BUTTON_PIN) == LOW && !isRunning) {

    delay(50); // debounce

    if (digitalRead(BUTTON_PIN) == LOW) {

      isRunning = true;
      runSequence();
      isRunning = false;

    }

  }

}


// =============================================

void runSequence() {

  Serial.println(">>> Sequence Started");


  // STEP 1: Servo 1 opens gate
  Serial.println("Servo 1: Opening");
  smoothMove(servo1, SERVO1_START, SERVO1_MID, SERVO1_SPEED);

  delay(1000);


  // STEP 2: Servo 1 closes gate
  Serial.println("Servo 1: Closing");
  smoothMove(servo1, SERVO1_MID, SERVO1_START, SERVO1_SPEED);

  delay(300);


  // STEP 3: UV LEDs ON
  Serial.println("UV LEDs: ON");

  digitalWrite(UV_LED1, HIGH);
  digitalWrite(UV_LED2, HIGH);

  delay(3000);

  digitalWrite(UV_LED1, LOW);
  digitalWrite(UV_LED2, LOW);

  Serial.println("UV LEDs: OFF");

  delay(300);


  // STEP 4: Servo 2 opens
  Serial.println("Servo 2: Opening");
  smoothMove(servo2, SERVO2_START, SERVO2_MID, SERVO2_SPEED);

  delay(1000);


  // STEP 5: Servo 2 closes
  Serial.println("Servo 2: Closing");
  smoothMove(servo2, SERVO2_MID, SERVO2_START, SERVO2_SPEED);

  Serial.println(">>> Sequence Complete");

  delay(500);

}
