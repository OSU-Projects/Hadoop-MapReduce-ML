#!/usr/bin/env python3
import sys
import numpy as np

def euclidean_distance_matrix(test_vector, train_matrix):
    """Computes Euclidean distances between test vector and all training vectors."""
    return np.linalg.norm(train_matrix - test_vector, axis=1)

# Load cleaned training data
train_labels = []
train_features = []

try:
    with open("Train_data_cleaned.txt", "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            try:
                train_labels.append(int(parts[0]))  # Extract label
                train_features.append(list(map(float, parts[1].split())))  # Extract feature vector
            except ValueError:
                print(f"Skipping malformed training data line: {line}", file=sys.stderr)

    # Convert to NumPy arrays for optimized operations
    train_labels = np.array(train_labels)
    train_features = np.array(train_features)

except Exception as e:
    print(f"Error reading Train_data_cleaned.txt: {e}", file=sys.stderr)
    sys.exit(1)

# Process test data from stdin
for line in sys.stdin:
    parts = line.strip().split("\t")
    if len(parts) < 2:
        continue
    try:
        test_id = hash(line)
        true_label = int(parts[0])
        test_features = np.array(list(map(float, parts[1].split()))).reshape(1, -1)  # Ensure shape compatibility

        # Ensure dimensions match before distance calculation
        if test_features.shape[1] != train_features.shape[1]:
            print(f"Feature size mismatch: Test({test_features.shape[1]}) vs Train({train_features.shape[1]})",
                  file=sys.stderr)
            continue  # Skip this test sample

        # Compute distances in a single operation
        distances = euclidean_distance_matrix(test_features, train_features)

        # Emit results for all training samples
        for i, dist in enumerate(distances):
            print(f"{test_id}\t{dist:.6f}\t{train_labels[i]}\t{true_label}")

    except ValueError:
        print(f"Skipping malformed test data line: {line}", file=sys.stderr)
    except Exception as e:
        print(f"Error processing test data: {e}", file=sys.stderr)
