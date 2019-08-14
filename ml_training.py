import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
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
