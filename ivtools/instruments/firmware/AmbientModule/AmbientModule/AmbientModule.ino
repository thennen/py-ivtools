#include <Arduino_MKRENV.h>
#include <ArduinoJson.h>

// Buffers, what should be the size of these?
StaticJsonDocument<400> requestDoc;
StaticJsonDocument<200> responseDoc;


// Functions to be called remotely
float getTemperature(String unit = "C"){
  float temp;
  if(unit == "C") temp = ENV.readTemperature(CELSIUS);
  else if(unit == "F") temp = ENV.readTemperature(FAHRENHEIT);
  else temp = -1;
  return temp;
}

float getHumidity(){
  // This doesn't take parameters, always returns percent
  float hum = ENV.readHumidity();
  return hum;
}

float getPressure(String unit = "mbar"){
  float pre;
  if(unit == "mbar") pre = ENV.readPressure(MILLIBAR);
  else if(unit == "kpa") pre = ENV.readPressure(KILOPASCAL);
  else if(unit == "psi") pre = ENV.readPressure(PSI);
  else pre = -1;
  return pre;
}

float getIlluminance(String unit = "lux"){
  float ill;
  if(unit == "lux") ill = ENV.readIlluminance(LUX);
  else if (unit == "footcandle") ill = ENV.readIlluminance(FOOTCANDLE);
  else if (unit == "metercandle") ill = ENV.readIlluminance(METERCANDLE);
  else ill = -1;
  return ill;
}




void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  ENV.begin();
}

void loop() {
  // put your main code here, to run repeatedly:
  
  // newline terminated
  if(Serial.available()){
    
    
    String request = Serial.readStringUntil('\n');
    
    deserializeJson(requestDoc, request);

    String reqJsonrpc = requestDoc["jsonrpc"];
    String reqMethod = requestDoc["method"];
    int reqId = requestDoc["id"];

    // Get requested sensor data
    // The requestDoc["params"]["unit"] | "C" syntax specifies a default, because params are optional
    float result;
    String error = "";
    if(reqMethod == "getTemperature") result = getTemperature(requestDoc["params"]["unit"] | "C");
    else if(reqMethod == "getHumidity") result = getHumidity();
    else if(reqMethod == "getPressure") result = getPressure(requestDoc["params"]["unit"] | "mbar");
    else if(reqMethod == "getIlluminance") result = getIlluminance(requestDoc["params"]["unit"] | "lux");
    else error = "method";

    // Build response
    responseDoc.clear();
    responseDoc["jsonrpc"] = "2.0";
    if(error == "method"){
      responseDoc["error"]["code"] = -32601;
      responseDoc["error"]["message"] = "Method doesn't exist!";
    } else if(result == -1){
      responseDoc["error"]["code"] = -32602;
      responseDoc["error"]["message"] = "Wrong parameters!";
    } else {
      responseDoc["result"] = result;
      responseDoc["id"] = reqId;
    }
    // Transmission details?
    serializeJson(responseDoc, Serial);
    Serial.print("\n");
  }
}
