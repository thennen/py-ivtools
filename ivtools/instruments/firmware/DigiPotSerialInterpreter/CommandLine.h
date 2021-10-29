/*****************************************************************************


*****************************************************************************
  Here's what's going on under the covers
*****************************************************************************
  Simple and Clear Command Line Interpreter

     This file will allow you to type commands into the Serial Window like,
        add 23,599
        blink 5
        playSong Yesterday

     to your sketch running on the Arduino and execute them.

     Implementation note:  This will use C strings as opposed to String Objects based on the assumption that if you need a commandLine interpreter,
     you are probably short on space too and the String object tends to be space inefficient.

   1)  Simple Protocol
         Commands are words and numbers either space or comma spearated
         The first word is the command, each additional word is an argument
         "\n" terminates each command

   2)  Using the C library routine strtok:
       A command is a word separated by spaces or commas.  A word separated by certain characters (like space or comma) is called a token.
       To get tokens one by one, I use the C lib routing strtok (part of C stdlib.h see below how to include it).
           It's part of C language library <string.h> which you can look up online.  Basically you:
              1) pass it a string (and the delimeters you use, i.e. space and comman) and it will return the first token from the string
              2) on subsequent calls, pass it NULL (instead of the string ptr) and it will continue where it left off with the initial string.
        I've written a couple of basic helper routines:
            readNumber: uses strtok and atoi (atoi: ascii to int, again part of C stdlib.h) to return an integer.
              Note that atoi returns an int and if you are using 1 byte ints like uint8_t you'll have to get the lowByte().
            readWord: returns a ptr to a text word

   4)  DoMyCommand: A list of if-then-elses for each command.  You could make this a case statement if all commands were a single char.
      Using a word is more readable.
*/
/******************sample main loop code ************************************

  #include "CommandLine.h"

  void
  setup() {
  Serial.begin(115200);
  }

  void
  loop() {
  bool received = getCommandLineFromSerialPort(CommandLine);      //global CommandLine is defined in CommandLine.h
  if (received) DoMyCommand(CommandLine);
  }

**********************************************************************************/

//Name this tab: CommandLine.h

#include <string.h>
#include <stdlib.h>
#include <Wire.h>
//this following macro is good for debugging, e.g.  print2("myVar= ", myVar);
#define print2(x,y) (Serial.print(x), Serial.println(y))


#define CR '\r'
#define LF '\n'
#define BS '\b'
#define NULLCHAR '\0'
#define SPACE ' '

#define COMMAND_BUFFER_LENGTH        25                        //length of serial buffer for incoming commands
char   CommandLine[COMMAND_BUFFER_LENGTH + 1];                 //Read commands into this buffer from Serial.  +1 in length for a termination char

const char *delimiters            = ", \n";                    //commands can be separated by return, space or comma

// pins for the LEDs:
const int redPin = 13;
const int reedPin = 5;
const int trigPin = 8;
int LEDvalue = 0;


/*************************************************************************************************************
     your Command Names Here
*/
const char *helpCommandToken          = "help";             //help feedback
const char *setupCommandToken         = "setup";            //fastest command no response can configure everything
const char *togglePinCommandToken     = "pin";              //enable and set any pin to high/low
const char *writeBypassCommandToken   = "bypass";           //switches the reed relay bypass on or off
const char *writeDigipotCommandToken  = "wiper";            //sets wiper of the digital potentiometer
const char *readDigipotCommandToken   = "get_wiper";        //reads wiper setting back





/*************************************************************************************************************
    getCommandLineFromSerialPort()
      Return the string of the next command. Commands are delimited by return"
      Handle BackSpace character
      Make all chars lowercase
*************************************************************************************************************/

bool
getCommandLineFromSerialPort(char * commandLine)
{
  static uint8_t charsRead = 0;                      //note: COMAND_BUFFER_LENGTH must be less than 255 chars long
  //read asynchronously until full command input
  while (Serial.available()) {
    char c = Serial.read();
    switch (c) {
      case CR:      //likely have full command in buffer now, commands are terminated by CR and/or LS
      case LF:
        commandLine[charsRead] = NULLCHAR;       //null terminate our command char array
        if (charsRead > 0)  {
          charsRead = 0;                           //charsRead is static, so have to reset
          //Serial.println(commandLine);
          return true;
        }
        break;
      case BS:                                    // handle backspace in input: put a space in last char
        if (charsRead > 0) {                        //and adjust commandLine and charsRead
          commandLine[--charsRead] = NULLCHAR;
          Serial << byte(BS) << byte(SPACE) << byte(BS);  //no idea how this works, found it on the Internet
        }
        break;
      default:
        // c = tolower(c);
        if (charsRead < COMMAND_BUFFER_LENGTH) {
          commandLine[charsRead++] = c;
        }
        commandLine[charsRead] = NULLCHAR;     //just in case
        break;
    }
  }
  return false;
}


/* ****************************
   readNumber: return a 16bit (for Arduino Uno) signed integer from the command line
   readWord: get a text word from the command line

*/
int
readNumber () {
  char * numTextPtr = strtok(NULL, delimiters);         //K&R string.h  pg. 250
  return atoi(numTextPtr);                              //K&R string.h  pg. 251
}

char * readWord() {
  char * word = strtok(NULL, delimiters);               //K&R string.h  pg. 250
  return word;
}

void
nullCommand(char * ptrToCommandName) {
  print2("Command not found: ", ptrToCommandName);      //see above for macro print2
}


/****************************************************
   Add your commands here
*/

void helpCommand(){
  Serial.println("> Available commands are:");
  Serial.println("> help");
  Serial.println("> setup <wiper0> <wiper1> <bypass 0/1>");
  Serial.println("> pin");
  Serial.println("> bypass <0/1 for open/close>");
  Serial.println("> wiper <wiper 0/1> <position>");
  Serial.println("> get_wiper <wiper 0/1>");
}


void setupCommand(){
  int w0 = readNumber();
  int w1 = readNumber();
  int reed = readNumber();
  digitalWrite(reedPin,!reed);

  // trimm setting to 6 bits
  w0 &= 0b00111111;
  w1 &= 0b00111111;
  // configure internal address bits for wiper 0/1
  w1 += 0b01000000;
  // Send Data to DS1808z 
  // use 7 bit Addr: 0101XXX, XXX = set Addr. bits on Board
  Wire.beginTransmission(0b0101000);
  Wire.write(byte(w0));
  Wire.write(byte(w1));
  Wire.endTransmission();
}

void togglePinCommand() {
  int pin = readNumber();
  int val = readNumber();
  pinMode(pin,OUTPUT);
  if(val) digitalWrite(pin,HIGH);
  else digitalWrite(pin,LOW);
}

bool writeBypass(){
  LEDvalue = ~LEDvalue;
  digitalWrite(redPin,LEDvalue);
  
  int reed = readNumber();
  digitalWrite(reedPin,!reed);
  return reed;   
}

void writeDigipot(){
  int pot = readNumber();
  int val = readNumber();
  Serial.println(pot);
  Serial.println(val);
  // trimm setting to 6 bits
  val &= 0b00111111;
  // configure internal address bits for wiper 0/1
  if(pot)
  val += 0b01000000;

  // Send Data to DS1808z 
  // use 7 bit Addr: 0101XXX, XXX = set Addr. bits on Board
  Wire.beginTransmission(0b0101000);
  Wire.write(byte(val));
  Wire.endTransmission();
}

void getWiper(byte *p0, byte *p1){
  Wire.requestFrom(0b0101000,2);
  *p0 = Wire.read();    
  *p1 = Wire.read();
}

/****************************************************
   DoMyCommand
*/
bool
DoMyCommand(char * commandLine) {
  //  print2("\nCommand: ", commandLine);
  int result;
  
  char * ptrToCommandName = strtok(commandLine, delimiters);
  //  print2("commandName= ", ptrToCommandName);

  if (strcmp(ptrToCommandName, helpCommandToken) == 0){
    helpCommand();

  }else if(strcmp(ptrToCommandName, setupCommandToken) == 0){
      setupCommand();
  
  }else if(strcmp(ptrToCommandName, togglePinCommandToken) == 0){
      togglePinCommand();
      print2("Pin set","");

      
  }else if (strcmp(ptrToCommandName, writeBypassCommandToken) == 0){
      
      if(writeBypass())
        print2("Bypass:", " closed");
      else
        print2("Bypass:", " opened");
  
  }else if (strcmp(ptrToCommandName, writeDigipotCommandToken) == 0){
      writeDigipot();
      print2(">    Wiper set.", "");

  }else if (strcmp(ptrToCommandName, readDigipotCommandToken) == 0){
      int pot = readNumber();
      byte p0, p1;
      getWiper(&p0,&p1);
      if(p0 == 255 && p1 == 255)
      Serial.println(-1);
      else{
        //remove don't care bits
        p0 &= 0b00111111;
        p1 &= 0b00111111;
        if(pot)
          Serial.print(p1);
        else
          Serial.print(p0);          
      }
  }else{
      nullCommand(ptrToCommandName);
  }
}
