from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

from preprocessing import X_train, y_train

# Classifiers to be optimized.
rf = RandomForestClassifier(random_state=20)
gb = GradientBoostingClassifier()
svm = SVC(probability=True)

auc_score = make_scorer(roc_auc_score)

# Random Forest Hyperparameter Tuning

# number of trees
n_estimators = range(200, 1000, 200)
# maximum number of features to use at each split
max_features = ['auto', 'sqrt']
# maximum depth of the tree
max_depth = range(2, 20, 2)
# minimum number of samples to split a node
min_samples_split = range(2, 10, 2)
# criterion for evaluating a split
criterion = ['gini', 'entropy']

# random grid

random_grid_rf = {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'criterion': criterion}

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid_rf,
                               n_iter=20, cv=2,
                               scoring=auc_score, verbose=1, random_state=42)
rf_random.fit(X_train, y_train)
print("RF best parameters:", rf_random.best_params_)

# Gradient Boosting Classifier Hyperparameter Tuning

# number of trees
n_estimators = range(50, 200, 50)

# maximum depth of the tree
max_depth = range(1, 5, 1)

# learning rate
learning_rate = [0.001, 0.01, 0.1]

# random grid

random_grid_gb = {'n_estimators': n_estimators,
                  'max_depth': max_depth,
                  'learning_rate': learning_rate}

# Create the randomized search cross-validation
gb_random = RandomizedSearchCV(estimator=gb, param_distributions=random_grid_gb, n_iter=20, cv=2, scoring=auc_score,
                               verbose=0, random_state=42)
gb_random.fit(X_train, y_train)
print("GB best parameters:", gb_random.best_params_)

# SVM Classifier Hyperparameter Tuning
# Kernel type to be used in the algorithm
kernel = ['linear', 'poly', 'rbf', 'sigmoid']

random_grid_svm = {'kernel': kernel}

svm_random = RandomizedSearchCV(estimator=svm, param_distributions=random_grid_svm,
                                n_iter=20, cv=2,
                                scoring=auc_score, verbose=1, random_state=42)
svm_random.fit(X_train, y_train)
print("SVM best parameters:", svm_random.best_params_)
