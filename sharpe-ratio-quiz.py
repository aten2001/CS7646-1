import math

k = math.sqrt(252)
avg_daily_ret = 10
daily_risk_free = 2
std_daily_ret = 10

sr = k * (avg_daily_ret - daily_risk_free) / std_daily_ret
print(sr)
