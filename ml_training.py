import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LassoLars
from sklearn import neural_network
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# get resources dir
res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources\\")

# read csv file from resources dir
money = pd.read_csv(res_dir + "money.csv")

past = 7*4  # 4 weeks in the past
future = 7*1  # 1 week in the future

# prepare the dataframe
df = list()
values = money["value"]  # take column 'value'
for i in range(past, len(money)-future):
    part_of_values = values[(i-past):(i+future)]
    df.append(list(part_of_values))
# for el in df:
#    print(el, "\n")

past_columns = [f"past_{i+1}" for i in range(past)]  # columns headers for 'past'
future_columns = [f"future_{i+1}" for i in range(future)]  # columns headers for 'future'

df = pd.DataFrame(df, columns=(past_columns + future_columns))  # create DataFrame with columns headers
print("\n", "DataFrame sample:")
print(df.head(5), "\n")

# Prepare data matrix 'X' and target(s) 'y' to fit the ML model
X = df[past_columns][:-1]
y = df[future_columns][:-1]

# Prepare test data matrix 'X' and real target(s) 'y' to test the ML model
X_test = df[past_columns][-1:]
y_test = df[future_columns][-1:]

# -------------------------
# LinearRegression ML model
# -------------------------
reg = LinearRegression().fit(X, y)  # fitting the ML model
prediction_LinearRegression = reg.predict(X_test)[0]  # predicting values

print("LinearRegression ML model", "\n", "-------------------------", "\n")
print("Real values:", "\n", y_test, "\n")
print("Prediction:", "\n", prediction_LinearRegression, "\n")
print("Difference:", "\n", prediction_LinearRegression - y_test, "\n")
print("Norm Difference: ", np.linalg.norm(prediction_LinearRegression - y_test), "\n", "\n")

# ----------------------------
# KNeighborsRegressor ML model
# ----------------------------
reg = KNeighborsRegressor(n_neighbors=7).fit(X, y)
prediction_KNeighborsRegressor = reg.predict(X_test)[0]

print("KNeighborsRegressor ML model", "\n", "-------------------------", "\n")
print("Real values:", "\n", y_test, "\n")
print("Prediction:", "\n", prediction_KNeighborsRegressor, "\n")
print("Difference:", "\n", prediction_KNeighborsRegressor - y_test, "\n")
print("Norm Difference: ", np.linalg.norm(prediction_KNeighborsRegressor - y_test), "\n", "\n")

# ------------------
# LassoLars ML model
# ------------------
reg = LassoLars(alpha=.11).fit(X, y)
prediction_LassoLars = reg.predict(X_test)[0]

print("KNeighborsRegressor ML model", "\n", "-------------------------", "\n")
print("Real values:", "\n", y_test, "\n")
print("Prediction:", "\n", prediction_LassoLars, "\n")
print("Difference:", "\n", prediction_LassoLars - y_test, "\n")
print("Norm Difference:", np.linalg.norm(prediction_LassoLars - y_test), "\n", "\n")

# ------------------------------------
# Neural network MLPRegressor ML model
# ------------------------------------
reg = neural_network.MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='lbfgs', alpha=0.00001,
                                  batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5,
                                  max_iter=900, shuffle=True, random_state=None, tol=0.00001, verbose=False,
                                  warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                                  validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
reg = reg.fit(X, y)
prediction_MLPRegressor = reg.predict(X_test)[0]

print("Neural net MLPRegressor ML model", "\n", "-------------------------", "\n")
print("Real values:", "\n", y_test, "\n")
print("Prediction:", "\n", prediction_MLPRegressor, "\n")
print("Difference:", "\n", prediction_MLPRegressor - y_test, "\n")
print("Norm Difference:", np.linalg.norm(prediction_MLPRegressor - y_test), "\n", "\n")

plt.title("Prediction ML models for RUR-USD values  ")
plt.plot(prediction_LinearRegression, label="LinearRegression")
plt.plot(prediction_KNeighborsRegressor, label="KNeighborsRegressor")
plt.plot(prediction_LassoLars, label="LassoLars")
plt.plot(prediction_MLPRegressor, label="Neural MLPRegressor")
plt.plot(df[future_columns].iloc[-1], label="Real")  # iloc - select an element from specific position
plt.legend()
plt.show()

# plt.plot(money["date"], money["value"])
# plt.xticks(rotation=45)
