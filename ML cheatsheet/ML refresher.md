# ML refresher

# Evaluation metrics:

# Precision, Recall and F1

![Untitled](ML%20refresher/Untitled.png)
<!-- C:\Users\amand\Desktop\gitUPLOAD\ML_cheatsheet\ML refresher 99ac69b019064cbea8e538f20b43535d\Untitled 1.png -->

![Untitled](ML%20refresher/Untitled%201.png)

![Untitled](ML%20refresher/Untitled%202.png)

|  | Actual Positive | Actual negative |
| --- | --- | --- |
| Predicted Positive | TP | FP |
| Predicted Negative | FN | TN |

TP = labels that are positive correctly predicted as positive

TN = labels that are negative are correctly predicted as negative

FP = labels that are negative incorrectly predicted as positive

FN = labels that are positive are incorrectly predicted as negative

Q. ***The task that has two labels, NEGATIVE that appears 90% of the
time and POSITIVE that appears 10% of the time.*** 

| Random distribution | Meaning | F1 | Accuracy |
| --- | --- | --- | --- |
| Uniform | Predicting each label with equal
probability (50%) | 0.167 | 0.5 |
| Task’s label Distribution | Predicting NEGATIVE 90% of the
time, and POSITIVE 10% of the time | 0.1  | 0.82 |
1. Precision for Positive (P+) = (True Positives)/(True Positives + False Positives)

Here, True Positives would be when the model correctly identifies a positive label, and this would happen with a 0.5 (chance it predicts positive) * 0.1 (chance it's actually positive) = 0.05 or 5% chance.

False Positives would be when the model incorrectly identifies a negative label as positive. This would happen with a 0.5 (chance it predicts positive) * 0.9 (chance it's actually negative) = 0.45 or 45% chance.

So, Precision for Positive would be 0.05 / (0.05 + 0.45) = 0.05 / 0.5 = 0.1 or 10%.

1. Recall for Positive (R+) = (True Positives)/(True Positives + False Negatives)

False Negatives are when the model incorrectly identifies a positive label as negative. This would happen with a 0.5 (chance it predicts negative) * 0.1 (chance it's actually positive) = 0.05 or 5% chance.

So, Recall for Positive would be 0.05 / (0.05 + 0.05) = 0.05 / 0.1 = 0.5 or 50%.

1. F1 Score for Positive = 2 * (P+ * R+) / (P+ + R+)

F1 Score for Positive = 2 * (0.1 * 0.5) / (0.1 + 0.5) = 0.1 / 0.6 = 1/6 ≈ 0.167 or 16.7%.