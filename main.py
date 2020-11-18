import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump, load

df = pd.read_csv('car data.csv')
print(df.head())

final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset['Current_Year'] = 2020 
final_dataset['no_year'] = final_dataset['Current_Year']-final_dataset['Year']

final_dataset.drop(['Year'], axis=1, inplace=True)
final_dataset.drop(['Current_Year'], axis=1, inplace=True)
final_dataset = pd.get_dummies(final_dataset, drop_first=True)
#print(final_dataset.head(10))

corrmat = final_dataset.corr()
top_corr_fetures = corrmat.index
#plt.figure(figsize=(20,20))
g = snb.heatmap(final_dataset[top_corr_fetures].corr(), annot=True, cmap="RdYlGn")
#plt.show()

X = final_dataset.iloc[:,1:]
Y = final_dataset.iloc[:,0]

model = ExtraTreesRegressor()
model.fit(X,Y)

#print(model.feature_importances_)
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)

rf_random.fit(X_train, Y_train)
predictions = rf_random.predict(X_test)

#print("Test : ",list(Y_test))
#print("Predictions : ",predictions)

#plt.scatter(Y_test,predictions)
#plt.show()

dump(rf_random,'Car_Prediction.joblib')
final_model = load('Car_Prediction.joblib')
print(X_test.head())
print(Y_test.head())
print(final_model.predict([[100000,20000,1,5,0,1,1,1]]))