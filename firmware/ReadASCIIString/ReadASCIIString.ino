#include <Wire.h>

// pins for the LEDs:
const int redPin = 13;
const int reedPin = 5;
const int trigPin = 8;
int LEDvalue = 0;

void setup() {
  // initialize serial:
  Serial.begin(9600);
  Serial.setTimeout(0);
  // initialize Two wire Interface:
  Wire.begin();
  // make the pins outputs:
  pinMode(redPin, OUTPUT);
  pinMode(reedPin, OUTPUT);
  pinMode(trigPin, OUTPUT);
  }

void loop() {
  
  // if there's any serial available, read it:
  while (Serial.available() > 0) {
    // look for the next valid integer in the incoming serial stream:
    // Potentiometer 0
    int pot0 = Serial.parseInt();
    int R0 = pot0;
    //Potentiometer 1
    int pot1 = Serial.parseInt();
    int R1 = pot1;
    // Reed switch bypass while 1
    int reed = Serial.parseInt();
    // adapt Data for pot0
    pot0 &= 0b00111111;
    // adapt Data for pot1
    pot1 &= 0b00111111;
    pot1 += 0b01000000;
    
    
    // Indicator LEDs
    LEDvalue = ~LEDvalue;
    digitalWrite(redPin,LEDvalue);
    // digitalWrite(reedPin,!reed);
    digitalWrite(reedPin,reed);
    digitalWrite(trigPin, HIGH);  
    // Send Data to DS1808z 
    // use 7 bit Addr: 0101XXX, XXX = set Addr. bits on Board
    Wire.beginTransmission(0b0101000);
    Wire.write(byte(pot0));
    Wire.write(byte(pot1));
    Wire.endTransmission();
   
    // return the three nubers
    
    Serial.print(R0, DEC);
    Serial.print(" ");
    Serial.print(R1, DEC);
    Serial.print(" ");
    Serial.println(reed, HEX);
    digitalWrite(trigPin, LOW);

  }
}
