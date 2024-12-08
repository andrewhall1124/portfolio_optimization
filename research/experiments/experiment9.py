import matplotlib.pyplot as plt
import seaborn as sns

from research.datasets import Historical
from research.utils import table

historical_data = Historical().df

random_date = historical_data["date"].sample(random_state=47).iloc[0]

day = historical_data[historical_data["date"] == random_date].copy().reset_index(drop=True)

prc_data = day["prc"].describe().to_frame().reset_index().rename(columns={"index": "statistic"})

table(prc_data, title="Sample of Random Date Price Data")

lower_bound = day["prc"].quantile(0)
quartile_1 = day["prc"].quantile(0.25)
quartile_2 = day["prc"].quantile(0.50)
quartile_3 = day["prc"].quantile(0.75)
upper_bound = day["prc"].quantile(0.99)

day_subset = day[(day["prc"] >= lower_bound) & (day["prc"] <= upper_bound)]

plt.figure(figsize=(12, 6))
sns.histplot(day_subset, x="prc", bins=100)
plt.suptitle(f"Distribution of Prices for {random_date.strftime("%Y-%m-%d")}")
plt.title("(Excluding top 1% of prices)")
plt.xlabel("Price")
plt.axvline(quartile_1, color="red", label="25%")
plt.axvline(quartile_2, color="red", label="50%")
plt.axvline(quartile_3, color="red", label="75%")
plt.legend()
plt.savefig("research/figures/experiment9-prcies-distribution.png")
