import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('CX4242_FlightData.csv')

df = df.iloc[:, [0, 1, 2, 6, 10]]

df['FLIGHT_DATE'] = pd.to_datetime(df['FLIGHT_DATE'])
df['Month_of_Travel'] = df['FLIGHT_DATE'].dt.month
X = df[['AIRLINE_CODE', 'ORIGIN_CODE', 'Month_of_Travel']]
y_delay = df['DEP_DELAY']
y_cancel = df['CANCELLED']

X_encoded = pd.get_dummies(X)

X_train, X_test, y_delay_train, y_delay_test, y_cancel_train, y_cancel_test = train_test_split(
    X_encoded, y_delay, y_cancel, test_size=0.2, random_state=42)

delay_model = RandomForestRegressor().fit(X_train, y_delay_train)

df['Predicted_Delay_Time'] = delay_model.predict(X_encoded)

grouped_df = df.groupby(['AIRLINE_CODE', 'ORIGIN_CODE', 'Month_of_Travel']).agg({
    'Predicted_Delay_Time': lambda x: max(x.mean(), 0),
    'CANCELLED': 'mean'
}).reset_index()

grouped_df = grouped_df.rename(columns={'CANCELLED': 'Cancellation_Probability'})

grouped_df.to_csv('flightdata_delay_and_cancellation.csv', index=False)


















