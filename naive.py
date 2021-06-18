# %% naive baseline to predict temperature data
import pandas as pd
import numpy as np

dat = pd.read_csv("temperature.csv")
# %%
dat["datetime"] = pd.to_datetime(dat["datetime"])
# %%
dat["Portland_pred"] = dat["Portland"].shift(1)
eval_idx = dat["datetime"]>="2016-11-30"
diffs = (dat.loc[eval_idx,"Portland_pred"] - dat.loc[eval_idx,"Portland"]).values
rmse = np.sqrt((diffs**2).mean())
print(rmse)