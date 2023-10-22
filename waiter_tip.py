import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('dataset.csv')

X = data[['total_bill', 'day', 'time', 'size']]
y = data['tip']

encoder = OneHotEncoder(sparse=False, drop='first')
X_encoded = encoder.fit_transform(X[['day', 'time']])
feature_names = encoder.get_feature_names_out(['day', 'time'])
X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)

X = pd.concat([X.drop(['day', 'time'], axis=1), X_encoded_df], axis=1)

model = LinearRegression()
model.fit(X, y)

new_data = pd.DataFrame({
    'total_bill': [40.00],
    'day': ['Sun'],
    'time': ['Dinner'],
    'size': [4]
})

new_data_encoded = encoder.transform(new_data[['day', 'time']])
new_data_encoded_df = pd.DataFrame(new_data_encoded, columns=feature_names)
new_data_final = pd.concat([new_data.drop(['day', 'time'], axis=1), new_data_encoded_df], axis=1)

predicted_tip = model.predict(new_data_final)

formatted_tip = "{:.2f}".format(predicted_tip[0])

print('Predicted Tip: ', formatted_tip)
