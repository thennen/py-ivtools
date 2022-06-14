const int pinRelay0 = 18;
const int pinRelay1 = 19;

const int pinLED1 = 12;
const int pinLED2 = 10;
const int pinLED3 = 13;
const int pinLED4 = 9;

void setup() {
  Serial.begin(9600);
  pinMode(pinRelay0, OUTPUT);
  pinMode(pinRelay1, OUTPUT);

  pinMode(pinLED1, OUTPUT);
  pinMode(pinLED2, OUTPUT);
  pinMode(pinLED3, OUTPUT);
  pinMode(pinLED4, OUTPUT);

  // initialize both relays to off/no coil current position
  digitalWrite(pinRelay0, LOW);
  digitalWrite(pinRelay1, LOW);
  digitalWrite(pinLED1, HIGH);
  digitalWrite(pinLED2, LOW);
  digitalWrite(pinLED3, HIGH);
  digitalWrite(pinLED4, LOW);
}

void loop() {

  // Two relais, like so: R2, R1
  //                      0, 1 -> R2 off, R1 on, etc.
  // take this as a number -> 0 ... 3, this is sent via serial for control
  // transmission is as ascii character
  // also a '?' con be sent to inquire about the current relay position
  
  int serialChar = Serial.read(); // returns -1 if nothing is in buffer
  if(serialChar == '0'){
    // off, off
    digitalWrite(pinRelay0, LOW);
    digitalWrite(pinRelay1, LOW);
    digitalWrite(pinLED1, HIGH);
    digitalWrite(pinLED2, LOW);
    digitalWrite(pinLED3, HIGH);
    digitalWrite(pinLED4, LOW);
    Serial.print("0");
  }
  if(serialChar == '1'){
    // off, on
    digitalWrite(pinRelay0, HIGH);
    digitalWrite(pinRelay1, LOW);
    digitalWrite(pinLED1, LOW);
    digitalWrite(pinLED2, HIGH);
    digitalWrite(pinLED3, HIGH);
    digitalWrite(pinLED4, LOW);
       
    Serial.print("1");
  }
  if(serialChar == '2'){
    // on, off
    digitalWrite(pinRelay0, LOW);
    digitalWrite(pinRelay1, HIGH);
    digitalWrite(pinLED1, HIGH);
    digitalWrite(pinLED2, LOW);
    digitalWrite(pinLED3, LOW);
    digitalWrite(pinLED4, HIGH);
    Serial.print("2");
  }
  if(serialChar == '3'){
    // on, on
    digitalWrite(pinRelay0, HIGH);
    digitalWrite(pinRelay1, HIGH);
    digitalWrite(pinLED1, LOW);
    digitalWrite(pinLED2, HIGH);
    digitalWrite(pinLED3, LOW);
    digitalWrite(pinLED4, HIGH);
    Serial.print("3");
  }
  if(serialChar == '?'){
    int state1 = digitalRead(pinRelay0);
    int state2 = digitalRead(pinRelay1);
    Serial.print(state1);
    Serial.print(state2);
  }
 
}
