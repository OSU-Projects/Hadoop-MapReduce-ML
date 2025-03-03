#!/usr/bin/env python3
import sys
from collections import Counter

K = 9  # Number of nearest neighbors

current_id = None
distances = []

for line in sys.stdin:
    parts = line.strip().split("\t")
    if len(parts) < 4:
        continue  # Skip bad input lines

    try:
        test_id = parts[0]
        distance = float(parts[1])
        train_label = int(parts[2])
        true_label = int(parts[3])

        if test_id != current_id and current_id is not None:
            # Sort distances and get top K neighbors
            distances.sort(key=lambda x: x[0])
            k_nearest = [label for _, label in distances[:K]]

            # Majority voting
            predicted_label = Counter(k_nearest).most_common(1)[0][0]
            
            # Print result (Test ID, Predicted Label, True Label)
            print(f"{current_id}\t{predicted_label}\t{true_label}")

            # Reset for new test instance
            distances = []

        current_id = test_id
        distances.append((distance, train_label))

    except Exception as e:
        print(f"Error processing reducer input: {e}", file=sys.stderr)

# Process last test instance
if current_id is not None and distances:
    distances.sort(key=lambda x: x[0])
    k_nearest = [label for _, label in distances[:K]]
    predicted_label = Counter(k_nearest).most_common(1)[0][0]
    print(f"{current_id}\t{predicted_label}\t{true_label}")
