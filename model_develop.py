import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import randint, expon

X_train = np.genfromtxt("top_env_train.csv", delimiter=',')
y_train = np.genfromtxt("top_resp_train.csv", delimiter=',')
X_test = np.genfromtxt("top_env_test.csv", delimiter=',')
y_test = np.genfromtxt("top_resp_test.csv", delimiter=',')

rng = np.random.RandomState(0)

clf = GradientBoostingRegressor(random_state=rng)

param_dist = {
    "loss": ['squared_error','absolute_error', 'huber'],
    "n_estimators": randint(50,2000),
    "max_depth": randint(5,63),
    "max_features": ['auto', 'sqrt', 'log2', 1,2,3,4,5],
    "min_samples_split": randint(2, 10),
    "criterion": ['friedman_mse', 'squared_error'],
    "learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0],
    "ccp_alpha" : expon(scale=0.1)
}

rsh = RandomizedSearchCV(
    estimator=clf, param_distributions=param_dist,random_state=rng, n_iter=1000,return_train_score=True, 
     cv=5, scoring="neg_root_mean_squared_error", 
)
rsh.fit(X_train, y_train)
print("Cross validation score (RMSE):")
print(-rsh.best_score_)
print("Testing score (RMSE):")
y_hat = rsh.best_estimator_.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_hat)))
print("Testing score (r_square):")
print(rsh.best_estimator_.score(X_test,y_test))