import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LassoLars
from sklearn import neural_network
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# money = pd.read_excel("c:\Temp\RC_F01_01_2019_T14_08_2019.xlsx")
money = pd.read_csv("c:\Temp\money.csv")

# plt.plot(money["date"], money["value"])
# plt.plot(money["value"])
# plt.xticks(rotation=45)
# plt.show()

past = 7*4
future = 7*1

df = list()
values = money["value"]
for i in range(past, len(money)-future):
    part_of_values = values[(i-past):(i+future)]
    df.append(list(part_of_values))

#for el in df:
#    print(el, "\n")

past_columns = [f"past_{i+1}" for i in range(past)]
future_columns = [f"future_{i+1}" for i in range(future)]

df = pd.DataFrame(df, columns=(past_columns + future_columns))
print(df.head(5))

X = df[past_columns][:-1]
y = df[future_columns][:-1]

X_test = df[past_columns][-1:]
y_test = df[future_columns][-1:]

print(y_test, "\n")

reg = LinearRegression().fit(X, y)

prediction = reg.predict(X_test)[0]

print(prediction, "\n")
print(prediction - y_test, "\n")
print(np.linalg.norm(prediction - y_test), "\n")

plt.plot(prediction, label="Prediction")
plt.plot(df[future_columns].iloc[-1], label="real")  # iloc - select an element from specific position
plt.legend()
plt.show()

# ---------------

reg = KNeighborsRegressor(n_neighbors=7).fit(X, y)

prediction = reg.predict(X_test)[0]

print(prediction, "\n")
print(prediction - y_test, "\n")
print(np.linalg.norm(prediction - y_test), "\n")

plt.plot(prediction, label="Prediction")
plt.plot(df[future_columns].iloc[-1], label="real")  # iloc - select an element from specific position
plt.legend()
plt.show()

# ---------------

reg = LassoLars(alpha=.11).fit(X, y)

prediction = reg.predict(X_test)[0]

print(prediction, "\n")
print(prediction - y_test, "\n")
print(np.linalg.norm(prediction - y_test), "\n")

plt.plot(prediction, label="Prediction")
plt.plot(df[future_columns].iloc[-1], label="real")  # iloc - select an element from specific position
plt.legend()
plt.show()

# ----------------
# neural

reg = neural_network.MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
reg = reg.fit(X, y)

prediction = reg.predict(X_test)[0]

print(prediction, "\n")
print(prediction - y_test, "\n")
print(np.linalg.norm(prediction - y_test), "\n")

plt.plot(prediction, label="Prediction")
plt.plot(df[future_columns].iloc[-1], label="real")  # iloc - select an element from specific position
plt.legend()
plt.show()
