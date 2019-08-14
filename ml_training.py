import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# money = pd.read_excel("c:\Temp\RC_F01_01_2019_T14_08_2019.xlsx")
money = pd.read_csv("c:\Temp\money.csv")

# plt.plot(money["date"], money["value"])
plt.plot(money["value"])
# plt.xticks(rotation=45)
plt.show()


