"""
Run with:  PYTHONPATH=. python metrics/metrics_test.py
"""
import numpy as np
from metrics.ece import compute_ece

passed = 0
total = 0


def check(name, condition):
    global passed, total
    total += 1
    status = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    print(f"[{status}] {name}")


# 1. Perfect calibration -> ECE ~ 0
# predicted class is always 1; true label is 1 with probability = confidence,
# so in each bin, empirical accuracy should track average confidence closely.
n = 5000
rng = np.random.default_rng(0)
confidences = rng.uniform(0, 1, n)
predictions = np.ones(n, dtype=int)
labels = (rng.uniform(0, 1, n) < confidences).astype(int)
ece, _ = compute_ece(confidences, predictions, labels, n_bins=15)
check("test_perfect_calibration_gives_near_zero_ece", ece < 0.05)

# 2. Overconfident predictions (confidence=1, but only 50% correct) -> large ECE
n2 = 500
confidences2 = np.ones(n2)
predictions2 = np.zeros(n2, dtype=int)
labels2 = np.array([0, 1] * (n2 // 2))  # only 50% match prediction=0
ece2, _ = compute_ece(confidences2, predictions2, labels2, n_bins=15)
check("test_overconfident_predictions_give_large_ece", ece2 > 0.4)

# 3. ECE is bounded in [0, 1]
check("test_ece_bounded", 0.0 <= ece <= 1.0 and 0.0 <= ece2 <= 1.0)

# 4. Empty input doesn't crash and gives ECE = 0
ece_empty, bins_empty = compute_ece(np.array([]), np.array([]), np.array([]), n_bins=10)
check("test_empty_input_handled", ece_empty == 0.0 and bins_empty == [])

# 5. bin_data covers only non-empty bins and fractions sum to ~1
_, bin_data = compute_ece(confidences, predictions, labels, n_bins=10)
total_frac = sum(b[4] for b in bin_data)
check("test_bin_fractions_sum_to_one", abs(total_frac - 1.0) < 1e-6)

# 6. Single perfect bin: all confidence=1.0, all correct -> ECE = 0
n3 = 100
confidences3 = np.ones(n3)
predictions3 = np.zeros(n3, dtype=int)
labels3 = np.zeros(n3, dtype=int)
ece3, _ = compute_ece(confidences3, predictions3, labels3, n_bins=10)
check("test_all_correct_full_confidence_zero_ece", ece3 == 0.0)

# 7. Single fully-wrong bin: confidence=1.0, all wrong -> ECE = 1
n4 = 100
confidences4 = np.ones(n4)
predictions4 = np.zeros(n4, dtype=int)
labels4 = np.ones(n4, dtype=int)
ece4, _ = compute_ece(confidences4, predictions4, labels4, n_bins=10)
check("test_all_wrong_full_confidence_ece_one", abs(ece4 - 1.0) < 1e-6)

# 8. More bins doesn't change ECE drastically for a smooth distribution
ece_10, _ = compute_ece(confidences, predictions, labels, n_bins=10)
ece_20, _ = compute_ece(confidences, predictions, labels, n_bins=20)
check("test_bin_count_stability", abs(ece_10 - ece_20) < 0.1)

# 9. Low confidence, low accuracy, well matched -> low ECE
n5 = 500
confidences5 = np.full(n5, 0.3)
predictions5 = np.zeros(n5, dtype=int)
labels5 = np.array(([0] * 150) + ([1] * 350))  # 30% accuracy matches 0.3 confidence
np.random.default_rng(1).shuffle(labels5)
ece5, _ = compute_ece(confidences5, predictions5, labels5, n_bins=10)
check("test_low_confidence_matched_to_low_accuracy", ece5 < 0.05)

# 10. bin_data tuples have expected structure (5 fields each)
check("test_bin_data_structure", all(len(b) == 5 for b in bin_data))

print(f"\n{passed}/{total} passed")
