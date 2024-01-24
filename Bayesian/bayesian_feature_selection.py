import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif


path = 'dataset.csv'
load_dataset = pd.read_csv(path)
X, y = load_dataset.drop('Outcome', axis=1), load_dataset['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Feature selection: get most features by number of k
selector = SelectKBest(f_classif, k=6)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)



# Apply Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train_selected, y_train)
y_pred_gnb = gnb.predict(X_test_selected)
accuracy_gnb2 = accuracy_score(y_test, y_pred_gnb)
print(f"Gaussian Naive Bayes Accuracy: {accuracy_gnb2}")


