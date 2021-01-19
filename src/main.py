import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve
from sklearn.svm import SVC

from preprocessing import X_train, X_test, y_train, y_test

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=800, min_samples_split=2, max_features='auto',
                            max_depth=18, criterion='gini', random_state=20)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict_proba(X_test)[:, 1]

# Gradient Boosing Classifier
gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict_proba(X_test)[:, 1]  # Compute probabilities for class 1.

# SVM Classifier
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict_proba(X_test)[:, 1]

# A threshold is used to split the probabilities of the classes.
thresh = .5


# Wrap up the metrics that are used.
def metrics(y_actual, y_pred, thresh):
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    f1 = 2 * (precision * recall) / (precision + recall)

    print('AUC:%.3f' % auc)
    print('accuracy:%.3f' % accuracy)
    print('recall:%.3f' % recall)
    print('precision:%.3f' % precision)
    print('f1:%.3f' % f1)
    print(' ')


# Print results.
print("RF Scores:", metrics(y_test, y_pred_rf, thresh))
print("GB Scores:", metrics(y_test, y_pred_gb, thresh))
print("SVM Scores:", metrics(y_test, y_pred_svm, thresh))

# Plot ROC Curves for all classifiers.
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_pred_rf)

fpr_gb, tpr_gb, thresholds_gb = roc_curve(y_test, y_pred_gb)
auc_gb = roc_auc_score(y_test, y_pred_gb)

fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, y_pred_svm)
auc_svm = roc_auc_score(y_test, y_pred_svm)

plt.plot(fpr_rf, tpr_rf, 'r-', label='Random Forest AUC:%.3f' % auc_rf)
plt.plot(fpr_gb, tpr_gb, 'g-', label='Gradient Boosting AUC:%.3f' % auc_gb)
plt.plot(fpr_svm, tpr_svm, 'b-', label='SVM AUC:%.3f' % auc_svm)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
plt.savefig("auc.png")
