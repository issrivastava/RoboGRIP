/*
 RoboGrip Arduino: 4-DOF arm + gripper, serial commands from Python.
 Wiring (SG90):
   base     -> D9, shoulder -> D6, elbow -> D5, grip -> D3
   External 5V 2A to servos GND+5V, Arduino GND common. NOT from Arduino 5V.
 Protocol 115200 baud, lines: GRIP, RELEASE, STOP, UP, DOWN, LEFT, RIGHT, MOVE, E:1, E:0
 Workspace limits = safety: BASE 20-160, SHOULDER 20-160, ELBOW 10-150
*/
#include <Servo.h>
Servo base, shoulder, elbow, grip;
int aBase=90, aSh=90, aEl=60, aGr=90; // grip 90=open, 40=closed
bool estop=false;
String buf="";

void applyAll(){
  base.write(constrain(aBase,20,160));
  shoulder.write(constrain(aSh,20,160));
  elbow.write(constrain(aEl,10,150));
  grip.write(constrain(aGr,20,160));
}
void setup(){
  Serial.begin(115200);
  base.attach(9); shoulder.attach(6); elbow.attach(5); grip.attach(3);
  applyAll();
}
void loop(){
  while(Serial.available()){
    char c=Serial.read();
    if(c=='\n'){ handle(buf); buf=""; } else if(c!='\r'){ buf+=c; if(buf.length()>20) buf=""; }
  }
}
void handle(String s){
  s.trim(); s.toUpperCase();
  if(s=="E:1"){ estop=true; Serial.println("OK ESTOP"); return; }
  if(s=="E:0"){ estop=false; Serial.println("OK CLEAR"); return; }
  if(estop){ Serial.println("BLOCKED ESTOP"); return; }
  if(s=="GRIP"){ aGr=40; }
  else if(s=="RELEASE"){ aGr=90; }
  else if(s=="UP"){ aSh=min(160,aSh+5); }
  else if(s=="DOWN"){ aSh=max(20,aSh-5); }
  else if(s=="LEFT"){ aEl=max(10,aEl-5); }
  else if(s=="RIGHT"){ aEl=min(150,aEl+5); }
  else if(s=="MOVE"){ aSh=min(160,aSh+2); aEl=min(150,aEl+2); }
  else if(s=="STOP"){ Serial.println("OK STOP"); return; }
  else { Serial.println("UNKNOWN"); return; }
  applyAll();
  Serial.print("OK "); Serial.print(aBase); Serial.print(" ");
  Serial.print(aSh); Serial.print(" "); Serial.println(aEl);
}
