//bool doBlink = false;
const int pinRelay0 = 18;
const int pinRelay1 = 19;

const int pinLED1 = 13;
const int pinLED2 = 12;
const int pinLED3 = 10;
const int pinLED4 = 9;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(pinRelay0, OUTPUT);
  pinMode(pinRelay1, OUTPUT);

  pinMode(pinLED1, OUTPUT);
  pinMode(pinLED2, OUTPUT);
  pinMode(pinLED3, OUTPUT);
  pinMode(pinLED4, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  // Two relais, like so: R2, R1
  //                      0, 1 -> R2 off, R1 on, etc.
  // take this as a number, this is the byte transmitted for control


  // returns -1 if nothing is in buffer

  int serialChar = Serial.read();
  if(serialChar == '0'){
    // off, off
    digitalWrite(pinRelay0, LOW);
    digitalWrite(pinRelay1, LOW);
    digitalWrite(pinLED1, HIGH);
    digitalWrite(pinLED2, LOW);
    digitalWrite(pinLED3, HIGH);
    digitalWrite(pinLED4, LOW);
    Serial.println("0");
  }
  if(serialChar == '1'){
    // off, on
    digitalWrite(pinRelay0, HIGH);
    digitalWrite(pinRelay1, LOW);
    digitalWrite(pinLED1, LOW);
    digitalWrite(pinLED2, HIGH);
    digitalWrite(pinLED3, HIGH);
    digitalWrite(pinLED4, LOW);
       
    Serial.println("1");
  }
  if(serialChar == '2'){
    // on, off
    digitalWrite(pinRelay0, LOW);
    digitalWrite(pinRelay1, HIGH);
    digitalWrite(pinLED1, HIGH);
    digitalWrite(pinLED2, LOW);
    digitalWrite(pinLED3, LOW);
    digitalWrite(pinLED4, HIGH);
    Serial.println("2");
  }
  if(serialChar == '3'){
    // on, on
    digitalWrite(pinRelay0, HIGH);
    digitalWrite(pinRelay1, HIGH);
    digitalWrite(pinLED1, LOW);
    digitalWrite(pinLED2, HIGH);
    digitalWrite(pinLED3, LOW);
    digitalWrite(pinLED4, HIGH);
    Serial.println("3");
  }
  
//  if(doBlink){
//    digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on (HIGH is the voltage level)
//    delay(300);                       // wait for a second
//    digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
//    delay(300);
//    // wait for a second
//  }
}
