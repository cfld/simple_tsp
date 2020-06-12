import sys
import pandas as pd
import matplotlib.pyplot as plt
from rsub import *

df = pd.read_csv(sys.stdin, sep=' ', header=None)

_ = plt.plot(df[2], label='best')
_ = plt.plot(df[3], label='curr')
_ = plt.grid('both')
_ = plt.legend()
show_plot()