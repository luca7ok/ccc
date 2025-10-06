import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

y = dataset['OUTPUT']
x = dataset.drop('OUTPUT', axis=1)

categorical_cols = ['MODE', 'POWER', 'UNIT']
numeric_cols = ['AMPS', 'VOLTS', 'TEMP', 'DELTA', 'GAMMA']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
    ]
)

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', RandomForestRegressor(n_estimators=100, random_state=1))])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

model_pipeline.fit(x_train, y_train)

print(round(model_pipeline.score(x_test, y_test), 3))

y_pred = model_pipeline.predict(x_test)
predictions = model_pipeline.predict(test_data)

print(mean_squared_error(y_test, y_pred))


with open('predictions.txt', 'w') as outFile:
    for pred in predictions:
        outFile.write(str(round(pred, 2)) + '\n')
