#!/usr/bin/env python3
import sys

total = 0
correct = 0

for line in sys.stdin:
    parts = line.strip().split("\t")
    if len(parts) < 3:
        continue
    _, predicted, true = map(int, parts)

    total += 1
    if predicted == true:
        correct += 1

accuracy = (correct / total) * 100
print(f"Accuracy: {accuracy:.2f}%")
