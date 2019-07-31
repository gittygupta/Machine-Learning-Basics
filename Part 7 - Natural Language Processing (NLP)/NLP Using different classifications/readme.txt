Evaluating the performance of each of these models. Accuracy is not enough, so we should also look at other performance metrics like 
Precision (measuring exactness), Recall (measuring completeness) and the F1 Score (compromise between Precision and Recall).
Metrics formulas (TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives):

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 Score = 2 * Precision * Recall / (Precision + Recall)


1. Using Logistic Regression Classifier

Accuracy = (66 + 76) / (66 + 76 + 21 + 37) = 0.71
Precision = 66 / (66 + 21) = 0.759
Recall = (66) / (66 + 37) = 0.641
F1 Score = 2 * 0.759 * 0.641 / (0.759 + 0.641) = 0.695

2. Using KNN

Accuracy = (48 + 74) / 200 = 0.61
Precision = 48 / (48 + 23) = 0.676
Recall = 48 / (48 + 55) = 0.466
F1 Score = 2 * 0.676 * 0.466 / (0.676 + 0.466) = 0.552

3. Using SVM

Accuracy = (70 + 74) / 200 = 0.71
Precision = 70 / (70 + 23) = 0.753
Recall = 70 / (70 + 33) = 0.6796
F1 Score = 2 * 0.753 * 0.6796 / (0.753 + 0.6796) = 0.714

4. Using Kernel SVM

Accuracy = (0 + 97) / 200 = 0.485
Precision = 0 / (48 + 23) = 0.0
Recall = 0 / (48 + 55) = 0.0
F1 Score = nan

5. Using Decision Tree

Accuracy = (68 + 74) / 200 = 0.71
Precision = 68 / (68 + 23) = 0.747
Recall = 68 / (68 + 35) = 0.660
F1 Score = 2 * 0.747 * 0.660 / (0.747 + 0.660) = 0.70

6. Using Random Forest

Accuracy = (57 + 87) / 200 = 0.72
Precision = 57 / (57 + 10) = 0.851
Recall = 57 / (57 + 46) = 0.553
F1 Score = 2 * 0.72 * 0.851 / (0.72 + 0.851) = 0.78

7. Using Naive Bayes

Accuracy = (91 + 55) / 200 = 0.73
Precision = 91 / (91 + 42) = 0.684
Recall = 91 / (91 + 12) = 0.884
F1 Score = 2 * 0.684 * 0.884 / (0.684 + 0.884) = 0.771