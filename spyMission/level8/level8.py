import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from scipy.stats import uniform, randint

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

model_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=1))])

param_dist = {
    'regressor__n_estimators': randint(100, 1000),
    'regressor__learning_rate': uniform(0.01, 0.3),
    'regressor__max_depth': randint(3, 10),
    'regressor__subsample': uniform(0.6, 0.4), # 0.6 is start, 0.4 is range width (so 0.6 to 1.0)
    'regressor__colsample_bytree': uniform(0.6, 0.4),
    'regressor__gamma': uniform(0, 0.5)
}
random_search = RandomizedSearchCV(
    estimator=model_pipeline,
    param_distributions=param_dist,
    n_iter=50,  # Increase for a more thorough search, decrease for speed
    cv=5,
    scoring='neg_mean_squared_error', # We want to minimize MSE
    verbose=1,
    n_jobs=-1,  # Use all available CPU cores
    random_state=1
)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

model_pipeline.fit(x_train, y_train)

print(round(model_pipeline.score(x_test, y_test), 3))

y_pred = model_pipeline.predict(x_test)
predictions = model_pipeline.predict(test_data)

print(mean_squared_error(y_test, y_pred))

random_search.fit(x_train, y_train)
best_model = random_search.best_estimator_
y_pred_tuned = best_model.predict(x_test)
print(mean_squared_error(y_test, y_pred_tuned))
final_predictions = best_model.predict(test_data)

with open('predictions.txt', 'w') as outFile:
    for pred in final_predictions:
        outFile.write(str(round(pred, 2)) + '\n')
