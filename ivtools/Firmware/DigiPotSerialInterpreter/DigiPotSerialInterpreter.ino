  #include "CommandLine.h"
  /* This firmware can control the digital potentiometer DS1808. 
   * The microcontroller can be controlled by sending it a command word with its parameters. 
   * Send it help to get a list of commands 
   */
  void
  setup() {
    // Initialize serial via USB
    Serial.begin(9600);
    // Initialize I2C comm to digital potentiometer
    Wire.begin();
    
    pinMode(redPin, OUTPUT);
    pinMode(reedPin, OUTPUT);
    pinMode(trigPin, OUTPUT);
  }

  void
  loop() {
  bool received = getCommandLineFromSerialPort(CommandLine);      //global CommandLine is defined in CommandLine.h
  if (received) DoMyCommand(CommandLine);
  }
