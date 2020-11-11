//PT1000-Temperature READER with ARDUINO Micro
#include <Wire.h>
#include <math.h>
#include <CmdMessenger.h>
#include <Adafruit_MCP4725.h> // Digital-Analog-Converter
#include <Filter.h> // Exponential-Filter

//LCD-Konfiguration
const byte LCDa = 0x28;         //LCD address on I2C bus
const char unit[] = {223,67,' ',' ',' ',' ' };  //degree Celsius + some spaces to fill display buffer
int count = 0;
// Attach a new CmdMessenger object to the default Serial port
CmdMessenger cmdMessenger = CmdMessenger(Serial);

//Global-Variables
  int analogWriteValue = 655; //Start-Temperature (19*Celsius) / Voltage (0.8V)
  int analogReadValue;

  int analogWriteChannel;
  int analogReadChannel;
  
  //Display backlight RGB pins
  int rPin = 6;
  int gPin = 5;
  int bPin = 4;
  
  //Digital-Analog-Converter
  Adafruit_MCP4725 dac;
  
  
  //Average Measurement for smooth Display-Output
  ExponentialFilter<float> FilteredTemperature(20, 0);
  ExponentialFilter<float> FilteredSP(20, 0);


// This is the list of recognized commands. These can be commands that can either be sent or received.
// For Exmaple in the serial monitor: "0,4,155;" { "Command(0-2), Digital-Pin, Value(0-255);" }
enum
{
  AO                   , // Command to set analog out "0;"
  AI                   , // Command to read analog input "1;"
  Reply                , // Command to report status "2;"
};


// Called when a received command has no attached function
void OnUnknownCommand()
{
  cmdMessenger.sendCmd(Reply,"Command without attached callback");
}


// Callback function that reads the analog input
void AnalogInput()
{
  analogReadChannel = cmdMessenger.readInt16Arg();
  analogReadValue = analogRead(analogReadChannel);
  
  cmdMessenger.sendCmdStart(Reply);
  cmdMessenger.sendCmdArg(analogReadValue);
  cmdMessenger.sendCmdEnd();
}

// Callback function that changes the analog output level
void AnalogOutput()
{
  //analogWriteChannel = cmdMessenger.readInt16Arg(); //reads Pin, not required if DAC is in use
  analogWriteValue = cmdMessenger.readInt16Arg(); //reads Value
 // analogWrite(analogWriteChannel, analogWriteValue);
  
 // cmdMessenger.sendCmdArg(analogWriteValue);
    dac.setVoltage(analogWriteValue, false); //call dac-function "setVoltage"    
 // cmdMessenger.sendCmdEnd();
}


// Callbacks define on which received commands we take action
void attachCommandCallbacks()
{
  // Attach callback methods
  cmdMessenger.attach(OnUnknownCommand);
  cmdMessenger.attach(AO, AnalogOutput);
  cmdMessenger.attach(AI, AnalogInput);
}

//Function to calculate the RTD-Resistance
float pt_resistor(float volt_now, float volt_bridge){
  //Resistor-Values
  float r_1, r_3, r_4;
  r_1 = 9975;
  r_3 = 9976;
  r_4 = 1001;
  
  //Equation for PT-Resistor-Value by splitting in Numerator and Denominator:
  float pt_zaehler = ( (volt_now * r_4) + ( (r_3 + r_4) * volt_bridge ) ) * r_1;  // R1 * ( V_g*(R4+R3)+R4*V_s)
  float pt_nenner = ((r_3 + r_4) * volt_now ) - ( volt_bridge * (r_3 + r_4) + volt_now * r_4);  // V_s*(R3+R4) - ( V_b*(R3+R4) + V_s*R4)

  //Getting RTD-Reistance
  float pt_res = (pt_zaehler / pt_nenner);
  return pt_res; 
}





// LCD functions
void LCDclear() {
  Wire.setClock(50000);
  Wire.beginTransmission(LCDa);
  Wire.write (0xFE);
  Wire.write(0x51);
  Wire.endTransmission();
  delay(2);
}

// Set Cursor to specified position line 0 or 1, pos 0 ... 15;
void LCDprint(char msg[], int line, byte pos) {
  Wire.setClock(50000);
  Wire.beginTransmission(LCDa);
  Wire.write (0xFE);
  Wire.write(0x45);
  pos = pos &0xF; //limit pos to 15 cut upper 4 bits
  line == 0 ? Wire.write(0x00+pos) : Wire.write(0x40+pos);   
  Wire.write(msg);
  Wire.endTransmission();
  delay(2);
}

void LCDprintTemp(float temp) {
  char buff[8];
  dtostrf(temp,6,2,buff);    //convert float to ascii char array
  LCDprint(strcat(buff,unit),0,6);
  delay(100);
  
}

void LCDprintSP(float dacBitValue) {

  float gain = 12.55; //Gain for Bridge-OPAmp
  float volt_now = 10;
  float SPvoltage = dacBitValue*5/4096;

  float temp = (log(pt_resistor(volt_now, SPvoltage/gain) / 1000) / log(1.00385));
  
  char buff[8];
  dtostrf(temp,6,2,buff);    //convert float to ascii char array
  LCDprint(strcat(buff,unit),1,6);
}




void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
  // initialize DAC
  dac.begin(0x62);
  // initialize the LCD's number of columns and rows:
  LCDclear();
  delay(500);     // to show somesthing happend 
  LCDprint("Temp:",0,0);
  LCDprint("Set:",1,0);
  
  pinMode(rPin,OUTPUT);
  pinMode(gPin,OUTPUT);
  pinMode(bPin,OUTPUT);
  
  // Adds newline to every command
  cmdMessenger.printLfCr();

  // Attach my application's user-defined callback methods
  attachCommandCallbacks();
  
  //Start-Temperature (19*Celsius) / Voltage (0.8V)
  dac.setVoltage(analogWriteValue, false); 
}

void loop() {
  float sensorValue1 = analogRead(A1); //Analog-Pin 1, it reads the Output-Voltage from Bridge
  float sensorValue2 = analogRead(A2); //Checking for PowerSupply-Conection

  // Convert the analog reading (which goes from 0 - 1023) to a voltage of (0 - 5V):
  float gain = 12.55; //Gain for Bridge-OPAmp
  float volt_now = 10;
  float voltsensor = (sensorValue1 * 5)  / 1023; //Voltage Output of Bridge-OpAmp and crucial for Temperature-Equation
  float voltbridge = voltsensor / gain;

  //Power-Supply Voltage, should be above 5V
  float volt_powerSupply = (sensorValue2 * 5) / 1023;

  // Temperature Equation
  float temperature = (log(pt_resistor(volt_now, voltbridge) / 1000) / log(1.00385));
  float temp_log = temperature; 
  
  // Average Temperature Values
  FilteredTemperature.Filter(temp_log);
  float SmoothTemperature = FilteredTemperature.Current();
  
 
   
  //Checking if Circuit has enough power
  if (volt_powerSupply < 2){
   
    LCDprint("  Check   ",0,6);
    LCDprint("  Power!  ",1,6);
    delay(10);

    //Violett in case of Power Failture
    analogWrite(rPin,0xFF);
    analogWrite(gPin,0x00);
    analogWrite(bPin,0xFF);
  } 
  else { 
    
    LCDprintTemp(SmoothTemperature);
    LCDprintSP(analogWriteValue); 
    delay(10);

    // Convert Temperature to RGB color 
    if(SmoothTemperature<8){
      analogWrite(rPin,0x00);
      analogWrite(gPin,0x00);
      analogWrite(bPin,0xFF);
    }else if(SmoothTemperature<15){
      analogWrite(rPin,0x00);
      analogWrite(gPin,0xFF);
      analogWrite(bPin,0xFF);
    }else if(SmoothTemperature<40){
      analogWrite(rPin,0x00);
      analogWrite(gPin,0xFF);
      analogWrite(bPin,0x00);
    }else if(SmoothTemperature<60){
      analogWrite(rPin,0xFF);
      analogWrite(gPin,0x8F);
      analogWrite(bPin,0x00);
    }else if(SmoothTemperature<80){
      analogWrite(rPin,0xFF);
      analogWrite(gPin,0x40);
      analogWrite(bPin,0x00);
    }else{
      analogWrite(rPin,0xFF);
      analogWrite(gPin,0x00);
      analogWrite(bPin,0x00);
    }
  }  

  
  // Process incoming serial data, and perform callbacks
  cmdMessenger.feedinSerialData();
}
