/*
  Simple WiFi server for controlling digital output pins.

  programmed by someone who doesn't know anything about protocols.

  Just connect to the TCP socket and send bytes.
  when you send a \n, output bytes to the digital pins.

 Circuit:
 * Board with NINA module (Arduino MKR WiFi 1010, MKR VIDOR 4000 and UNO WiFi Rev.2)

 created 14 Jan 2023
 by Tyler Hennen
 */

#include <SPI.h>
#include <WiFiNINA.h>

#include "arduino_secrets.h" 

#include <CircularBuffer.h>

// we have 18 pins to control, need 3 bytes
#define BUFSIZE 3
///////please enter your sensitive data in the Secret tab/arduino_secrets.h
char ssid[] = SECRET_SSID;        // your network SSID (name)
char pass[] = SECRET_PASS;        // your network password (use for WPA, or use as key for WEP)
int keyIndex = 0;                 // your network key index number (needed only for WEP)

int i = 0;
int j = 0;

CircularBuffer<char,BUFSIZE> buffer;

//char buffer[BUFSIZE];
//int head = 0;
//int tail = 0;

int status = WL_IDLE_STATUS;
WiFiServer server(1337);

// MSB -> LSB
// block1, block2, block3, slset1, slset2, slreset, wlset , wlreset, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9
int pins[] = {3,4,5,6,7,8,9,10,0,21,20,19,18,17,16,15,2,1}; // should not exceed length BUFSIZE*8
int npins = 18;
//int extrabits = 2^BUFSIZE - npins;

void setup() {
  Serial.begin(9600);      // initialize serial communication

  for (i = 0; i<npins; i++) {
    pinMode(pins[i], OUTPUT);
  }

  // check for the WiFi module:
  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    // don't continue
    while (true);
  }

  String fv = WiFi.firmwareVersion();
  if (fv < WIFI_FIRMWARE_LATEST_VERSION) {
    Serial.println("Please upgrade the firmware");
  }

  Serial.print("Creating access point named: ");
  Serial.println(ssid);
  // Create open network. Change this line if you want to create an WEP network:
  status = WiFi.beginAP(ssid, pass);
  if (status != WL_AP_LISTENING) {
    Serial.println("Creating access point failed");
    // don't continue
    while (true);
  }
  
  // wait 10 seconds for connection:
  delay(10000);

  // alternatively ...
  // attempt to connect to WiFi network:
  //while (status != WL_CONNECTED) {
    //Serial.print("Attempting to connect to Network named: ");
    //Serial.println(ssid);                   // print the network name (SSID);

    // Connect to WPA/WPA2 network. Change this line if using open or WEP network:
    //status = WiFi.begin(ssid, pass);
    // wait 10 seconds for connection:
    //delay(10000);
  //}


  server.begin();                           // start the web server on port 80
  printWifiStatus();                        // you're connected now, so print out the status
}

void loop() {
  WiFiClient client = server.available();   // listen for incoming clients

  if (client) {                             // if you get a client,
    Serial.println("new client");           // print a message out the serial port
    while (client.connected()) {            // loop while the client is connected
      if (client.available()) {             // if there's bytes to read from the client,
        char c = client.read();             // read a byte, then
        Serial.write(c);                    // print it out the serial monitor
        if (c == '\n') {                    // if the byte is a newline character
            //Serial.write("Buffer contents: ");
            for (i=0;i<buffer.size();i++){
              Serial.write(buffer[i]);
            }
            Serial.println();
            for (i=0;i<buffer.size();i++){
              //Serial.print(buffer[i], BIN);
              for (j=0; j<8; j++) {
                Serial.print(bitRead(buffer[i], 7-j));
              }
              Serial.print(" ");
            }
            Serial.println();
            
            if (buffer.isFull()) {
              Serial.println("Latching");
              for (i = 0; i < BUFSIZE; i++) {
                for (j = 0; j < 8; j++) {
                  // Output the binary representation of the characters
                  // probably close enough to simultaneously for our purposes
                  if (i*8 + j < npins) {
                    digitalWrite(pins[npins - 1 - i*8 - j], bitRead(buffer[BUFSIZE - 1 - i], j));
                  }
                }
              }
              buffer.clear();
              // send some kind of acknowledgement signal?
            } else {
              // treat this newline as data (0b0001010) since the buffer isn't full yet
              buffer.push(c);
            }
        } else {
          buffer.push(c);
        }
      }
    }
    // close the connection:
    client.stop();
    buffer.clear();
    Serial.println("\nclient disconnected");
  }
}

void printWifiStatus() {
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  long rssi = WiFi.RSSI();
  Serial.print("signal strength (RSSI):");
  Serial.print(rssi);
  Serial.println(" dBm");
  Serial.println(ip);
}
