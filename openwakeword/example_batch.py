import os
import numpy as np
from openwakeword.model import Model

# Folders
SUCCESS_DIR = "./success"
FAIL_DIR = "./fail"

# Initialize the model
model = Model(
    wakeword_models=["./xiaobai2/xiaobai.tflite"]
)
#    wakeword_models=["./xiaobai2/xiaobai.tflite"]
#    wakeword_models=["./xiaobai4/xiaobai.tflite"]
def get_max_score(filepath):
    """Return the maximum wakeword score across all frames for a file."""
    result = model.predict_clip(filepath)
    max_score = max(max(frame.values()) for frame in result)
    return max_score

def evaluate_threshold(threshold, success_scores, fail_scores):
    tp_files = [f for f, score in success_scores if score >= threshold]
    fn_files = [f for f, score in success_scores if score < threshold]
    fp_files = [f for f, score in fail_scores if score >= threshold]
    tn_files = [f for f, score in fail_scores if score < threshold]

    tp, fn, fp, tn = len(tp_files), len(fn_files), len(fp_files), len(tn_files)

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    return accuracy, precision, recall, tp, fn, fp, tn, fn_files, fp_files

def main():
    # Collect scores (store filename + score)
    success_scores = [(f, get_max_score(os.path.join(SUCCESS_DIR, f)))
                      for f in os.listdir(SUCCESS_DIR) if f.endswith(".wav")]
    fail_scores = [(f, get_max_score(os.path.join(FAIL_DIR, f)))
                   for f in os.listdir(FAIL_DIR) if f.endswith(".wav")]

    thresholds = np.linspace(0.06, 0.9, 85)  # sweep 0.1, 0.15, ..., 0.9
    best_acc = 0
    best_thr = None
    best_stats = None

    print("Threshold tuning results:")
    for thr in thresholds:
        acc, prec, rec, tp, fn, fp, tn, fn_files, fp_files = evaluate_threshold(thr, success_scores, fail_scores)
        print(f"Thr={thr:.2f} | Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | "
              f"TP={tp} FN={fn} FP={fp} TN={tn}")
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
            best_stats = (tp, fn, fp, tn, fn_files, fp_files)

    tp, fn, fp, tn, fn_files, fp_files = best_stats
    print("\n✅ Best threshold:", best_thr, f"(Accuracy={best_acc:.3f})")
    print(f"   - True Positives:  {tp}")
    print(f"   - False Negatives: {fn}  (wakewords missed)")
    print(f"   - False Positives: {fp}  (false alarms)")
    print(f"   - True Negatives:  {tn}")

    if fn_files:
        print("\n📂 False Negative Files (missed wakewords):")
        for f in fn_files:
            print("   -", f)

    if fp_files:
        print("\n📂 False Positive Files (false alarms):")
        for f in fp_files:
            print("   -", f)

if __name__ == "__main__":
    main()

