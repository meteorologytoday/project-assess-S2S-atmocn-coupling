import pandas as pd
from pandas.tseries.offsets import DateOffset

import numpy as np

@np.vectorize
def toYearFraction(dt):
    
    dt_first_day = pd.Timestamp(year=dt.year, month=1, day=1)
    dt_first_day_next_year = pd.Timestamp(year=dt.year+1, month=1, day=1)
    
    return (dt - dt_first_day) / (dt_first_day_next_year - dt_first_day)



