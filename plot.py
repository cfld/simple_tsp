import sys
import pandas as pd
import matplotlib.pyplot as plt
from rsub import *

for inpath in sys.argv[1:]:
    print(inpath)
    df = pd.read_csv(inpath, sep=' ', header=None)
    # _ = plt.plot(df[2], label='best')
    _ = plt.plot(df[0], df[3], label=f'curr {inpath}')

_ = plt.grid('both')
_ = plt.legend()
show_plot()