#!/usr/bin/env python3
import sys
from heapq import nsmallest

K = 13  # Change K as needed

current_test_id = None
distances = []

for line in sys.stdin:
    parts = line.strip().split("\t")
    if len(parts) < 4:
        continue

    test_id, distance, train_label, true_label = parts[0], float(parts[1]), int(parts[2]), int(parts[3])

    if test_id != current_test_id and current_test_id is not None:
        # Emit top K nearest neighbors for each test sample
        for dist, label in nsmallest(K, distances):
            print(f"{current_test_id}\t{dist}\t{label}\t{true_label}")
        distances = []

    current_test_id = test_id
    distances.append((distance, train_label))

# Emit the last test sample
if current_test_id is not None:
    for dist, label in nsmallest(K, distances):
        print(f"{current_test_id}\t{dist}\t{label}\t{true_label}")
