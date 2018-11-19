import os
import json

Person = ["person06010","person06011","person06012","person06013","person06014",
              "person06015","person06016","person06017","person06018","person06019",
              "person06020","person06021","person06022","person06023"]

TerminalType = ["Samsung;Galaxy Nexus;AndroidOS 4.1;","Logger+Wifi for  Android;1.0","Samsung;NexusS;AndroidOS 4.1;"]

TerminalPosition = ["wear;outer;chest;left","arm;right;hand","wear;pants;waist;fit;right-front",
                    "wear;pants;waist;fit;right-back","bag;position(fixed);shoulderbag","bag;position(fixed);handback"
                    "bag;position(fixed);messengerbag", "bag;position(fixed);backpack"]

Activity = ['jog','skip','stay','stDown','stUp','walk']

persondict = {}
typedict = {}
positiondict = {}
activitydict = {}

for i in range(len(Person)):
    persondict[Person[i]] = i
for i in range(len(TerminalType)):
    typedict[TerminalType[i]] = i
for i in range(len(TerminalPosition)):
    positiondict[TerminalPosition[i]] = i
for i in range(len(Activity)):
    activitydict[Activity[i]] = i

mapdict = {}
mapdict["person"] = persondict
mapdict["type"] = typedict
mapdict["position"] = positiondict
mapdict["activity"] = activitydict

with open("itemdict.json",'w') as fw:
    json.dump(mapdict, fw)

print "Generate itemdict.json successfully"