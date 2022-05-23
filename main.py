import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Dataset
bank_data = pd.read_csv('data/bank.csv', sep=';')
bank_features = bank_data.drop('y', axis=1)
bank_output = bank_data.y
# Let's use pandas to massively convert the values of the features into a numeric set.
bank_features = pd.get_dummies(bank_features)
bank_output = bank_output.replace({
    'no': 0,
    'yes': 1
})
# We split the data set into parts - 75% for training, 25% for verification
X_train, X_test, y_train, y_test = train_test_split(bank_features, bank_output,
                                                    test_size=0.25, random_state=42)
# Creating a model
bank_model = LogisticRegression(C=1e6, solver='liblinear')
bank_model.fit(X_train, y_train)
# Calculate the obtained accuracy
accuracy_score = bank_model.score(X_train, y_train)
print(accuracy_score)
# Demonstration of data problems
plt.bar([0, 1], [len(bank_output[bank_output == 0]), len(bank_output[bank_output ==
1])])
plt.xticks([0, 1])
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()
# Output of the number of moments of "success"
print('Positive cases: {:.3f}% of all'.format(bank_output.sum() / len(bank_output) *
100))
# We carry out forecasting on the test part
predictions = bank_model.predict(X_test)
# We compare forecasts with data and output a report
print(classification_report(y_test,predictions))