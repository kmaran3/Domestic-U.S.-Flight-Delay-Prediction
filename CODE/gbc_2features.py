import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('CX4242_FlightData.csv')

df['Month2'] = pd.to_datetime(df['FLIGHT_DATE']).dt.month

df.drop('FLIGHT_DATE', axis=1, inplace=True)

df.insert(0, 'Month', df['Month2'])

df.drop('Month2', axis=1, inplace=True)

X = df.iloc[:, [1,2]]
y = df["DEP_DELAY"].apply(lambda x: 0 if x <= 15 else 1) 

y = y.fillna(y.median()).astype(float)

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

clf = GradientBoostingClassifier(random_state=42)

clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)

accuracy_train = accuracy_score(y_train, y_pred_train)

print("Training - Accuracy:", accuracy_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Test - Accuracy:", accuracy)




