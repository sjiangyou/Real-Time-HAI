import numpy as np
import pandas as pd
import os

os.chdir("Data/SHIPS")
for file in os.listdir():
    if file.endswith(".dat"):
        print(file)

f = open("23090606AL1323_lsdiag.dat", "r")
text = f.read()
print(text)

lines = text.split("\n")
for i, line in enumerate(lines):
    lines[i] = line[:-10]

new_lines = []
for line in lines:
    new_line = []
    temp_entry = ""
    for j in range(len(line)):
        if line[j] != " ":
            temp_entry += line[j]
        elif line[j] == " " and temp_entry != "":
            new_line.append(temp_entry)
            temp_entry = ""
    new_line.append(temp_entry)
    new_lines.append(new_line)

new_lines[0]
basin = new_lines[0][0][:2]
ident = new_lines[0][0][2:]
latitude = float(new_lines[0][4])
longitude = float(new_lines[0][5])
basin, ident, latitude, longitude

vmax = int(new_lines[0][3])
if vmax >= 135:
    category = "C5"
elif vmax >= 113:
    category = "C4"
elif vmax >= 96:
    category = "C3"
elif vmax >= 83:
    category = "C2"
elif vmax >= 64:
    category = "C1"
elif vmax >= 34:
    category = "TS"
else:
    category = "TD"
category
