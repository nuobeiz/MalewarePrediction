from sklearn.tree import DecisionTreeClassifier  # for classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve

# Input: X train, y train, X test, y test
# Output: acc, auc, model summary
def logreg(X_train, y_train, X_test, y_test):
    # Train the model
    model = LogisticRegression(penalty='l1', solver='liblinear')
    model.fit(X_train, y_train)

    # Get the testing acc and auc
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    return accuracy, auc