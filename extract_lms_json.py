#!/usr/bin/env python3
"""Extract LMS data from CDC CSV files into JSON for JS embedding."""
import csv
import json
from pathlib import Path

BASE = Path(__file__).parent

def read_csv(filename, sex_code):
    rows = []
    with open(BASE / filename) as f:
        for row in csv.DictReader(f):
            if int(row["Sex"]) == sex_code:
                rows.append({
                    "a": round(float(row["Agemos"]), 1),
                    "L": round(float(row["L"]), 6),
                    "M": round(float(row["M"]), 4),
                    "S": round(float(row["S"]), 6),
                })
    return rows

data = {}
for sex_label, sex_code in [("M", 1), ("F", 2)]:
    infant = read_csv("lenageinf.csv", sex_code)
    stature = read_csv("statage.csv", sex_code)
    data[sex_label] = {"infant": infant, "stature": stature}

print(json.dumps(data, separators=(",", ":")))
