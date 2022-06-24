// Firmware for Arduino MKR ZERO with MKR ENV shield
// This provides a temperature, humidity, pressure and illumination sensor.
// To make the sensor data available, this acts as a JSON-RPC server.
// Communication with this is via the serial interface.


#include <Arduino_MKRENV.h>
#include <ArduinoJson.h>

// determined buffer sizes using the provided calculator
// because the strings are copied for parsing, these mainly determine the required space

// used: {"jsonrpc": "2.0", "method": "getIlluminance", "id": 1000000, "params": {"unit": "metercandle"}}
// to calculate the request string, this should contain the longest parameter/unit names and a big id (the id might not matter though)
// recommended value: 192, double that for good measure: 384

StaticJsonDocument<384> requestDoc;

// TODO
// used: {"jsonrpc": "2.0", "error": {"code": -32601, "method": "Method doesn't exist!"}, "id": 1000000}
StaticJsonDocument<2000> responseDoc;

///////////////////////////////////////////////////////////////////////
///// Functions to be called remotely to return measurement data. /////
///////////////////////////////////////////////////////////////////////

///// Get a single property

// Return a struct with value and unit.
// Value is -1 if a not supported unit is requested.

// This is used so we can return the data with the actual unit used
struct measData {
  float value;
  String unit;
};

struct measData getTemperature(String unit = "C"){
  float temp;
  if(unit == "C") temp = ENV.readTemperature(CELSIUS);
  else if(unit == "F") temp = ENV.readTemperature(FAHRENHEIT);
  else temp = -1;
  
  struct measData ret = {temp, unit};
  return ret;
}

struct measData getHumidity(){
  // This doesn't take parameters, always returns percent
  float hum = ENV.readHumidity();
  
  struct measData ret = {hum, "percent"};
  return ret;
}

struct measData getPressure(String unit = "mbar"){
  float pre;
  if(unit == "mbar") pre = ENV.readPressure(MILLIBAR);
  else if(unit == "kpa") pre = ENV.readPressure(KILOPASCAL);
  else if(unit == "psi") pre = ENV.readPressure(PSI);
  
  struct measData ret = {pre, unit};
  return ret;
}

struct measData getIlluminance(String unit = "lux"){
  float ill;
  if(unit == "lux") ill = ENV.readIlluminance(LUX);
  else if (unit == "footcandle") ill = ENV.readIlluminance(FOOTCANDLE);
  else if (unit == "metercandle") ill = ENV.readIlluminance(METERCANDLE);
  else ill = -1;
  
  struct measData ret = {ill, unit};
  return ret;
}

///// Get all sensors together


// We make a struct to deal with the fact that C can't return arrays
struct allData {
  struct measData temp;
  struct measData hum;
  struct measData pre;
  struct measData ill;
};

struct allData getAll(String tempUnit = "C", String pressUnit = "mbar", String illUnit = "lux"){
  struct measData temp = getTemperature(tempUnit);
  struct measData hum = getHumidity();
  struct measData pre = getPressure(pressUnit);
  struct measData ill = getIlluminance(illUnit);

  struct allData data = {temp, hum, pre, ill};
  return data;
}



/////////////////////////////////////////
///// Handle JSON-RPC communication /////
/////////////////////////////////////////

void setup() {
  Serial.begin(9600);
  ENV.begin();
}

void loop() {
  
  
  if(Serial.available()){

    // Get a request json object from serial, newline terminated
    // and extract data
    String request = Serial.readStringUntil('\n');
    
    deserializeJson(requestDoc, request);

    String reqJsonrpc = requestDoc["jsonrpc"];
    String reqMethod = requestDoc["method"];
    int reqId = requestDoc["id"];



    // Get requested sensor data
    
    // The requestDoc["params"]["unit"] | "C" syntax specifies a default, because params are optional in json-rpc
    // but arduinoJSON defaults to an empty string if the object is not there
    
    struct measData result;
    struct allData bigResult;


    // TODO: handle errors properly in this
    String error = "";
    
    if(reqMethod == "getTemperature") result = getTemperature(requestDoc["params"]["unit"] | "C");
    else if(reqMethod == "getHumidity") result = getHumidity();
    else if(reqMethod == "getPressure") result = getPressure(requestDoc["params"]["unit"] | "mbar");
    else if(reqMethod == "getIlluminance") result = getIlluminance(requestDoc["params"]["unit"] | "lux");
    else if(reqMethod == "getAll") bigResult = getAll(requestDoc["params"]["units"][0],
                                                      requestDoc["params"]["units"][1],
                                                      requestDoc["params"]["units"][2]);
    else error = "method";

    // Build response
    responseDoc.clear();
    responseDoc["jsonrpc"] = "2.0";
    if(error == "method"){
      responseDoc["error"]["code"] = -32601;
      responseDoc["error"]["message"] = "Method doesn't exist!";
    } else if(result.value == -1){
      responseDoc["error"]["code"] = -32602;
      responseDoc["error"]["message"] = "Wrong parameters!";
    } else {
      if (reqMethod == "getAll"){
        responseDoc["result"]["temperature"]["value"] = (bigResult.temp).value; // The brackets are actually needed
        responseDoc["result"]["temperature"]["unit"] = (bigResult.temp).unit;
        responseDoc["result"]["humidity"]["value"] = (bigResult.hum).value;
        responseDoc["result"]["humidity"]["unit"] = (bigResult.hum).unit;
        responseDoc["result"]["pressure"]["value"] = (bigResult.pre).value;
        responseDoc["result"]["pressure"]["unit"] = (bigResult.pre).unit;
        responseDoc["result"]["illuminance"]["value"] = (bigResult.ill).value;
        responseDoc["result"]["illuminance"]["unit"] = (bigResult.ill).unit;
      } else {
        responseDoc["result"]["value"] = result.value;
        responseDoc["result"]["unit"] = result.unit;
      }
    }
    responseDoc["id"] = reqId;
    
    // Send response, also newline terminated
    serializeJson(responseDoc, Serial);
    Serial.print("\n");
  }
}
