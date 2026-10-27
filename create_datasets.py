import os
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)

# 1. Mall Customers Dataset (Standard 200 rows benchmark)
np.random.seed(42)
n_mall = 200
customer_ids = np.arange(1, n_mall + 1)
genders = np.random.choice(['Male', 'Female'], size=n_mall, p=[0.44, 0.56])
ages = np.random.randint(18, 70, size=n_mall)

# Realistic multi-modal distribution for Mall Customers:
# 5 distinct segments typical of the famous Mall dataset:
# - Low Income, Low Spending (Sensible/Strugglers)
# - Low Income, High Spending (Careless/Youth)
# - High Income, Low Spending (Careful/Savers)
# - High Income, High Spending (Target/VIP)
# - Medium Income, Medium Spending (Standard/Average)
segment_assignments = np.random.choice([0, 1, 2, 3, 4], size=n_mall, p=[0.18, 0.18, 0.18, 0.22, 0.24])

incomes = np.zeros(n_mall)
spending_scores = np.zeros(n_mall)

for i in range(n_mall):
    seg = segment_assignments[i]
    if seg == 0:  # Low income, Low spending
        incomes[i] = np.random.normal(25, 6)
        spending_scores[i] = np.random.normal(20, 8)
    elif seg == 1:  # Low income, High spending
        incomes[i] = np.random.normal(25, 6)
        spending_scores[i] = np.random.normal(78, 10)
    elif seg == 2:  # High income, Low spending
        incomes[i] = np.random.normal(88, 12)
        spending_scores[i] = np.random.normal(18, 8)
    elif seg == 3:  # High income, High spending
        incomes[i] = np.random.normal(86, 14)
        spending_scores[i] = np.random.normal(82, 9)
    else:  # Medium income, Medium spending
        incomes[i] = np.random.normal(55, 10)
        spending_scores[i] = np.random.normal(50, 10)

incomes = np.clip(np.round(incomes), 15, 137).astype(int)
spending_scores = np.clip(np.round(spending_scores), 1, 99).astype(int)

df_mall = pd.DataFrame({
    'CustomerID': customer_ids,
    'Gender': genders,
    'Age': ages,
    'Annual Income (k$)': incomes,
    'Spending Score (1-100)': spending_scores
})
df_mall.to_csv("data/mall_customers.csv", index=False)
print("Saved data/mall_customers.csv:", df_mall.shape)

# 2. Online Retail RFM Dataset (1,000 customers)
# Derived from transaction patterns:
# Recency (days since last purchase: 1 to 365)
# Frequency (number of orders: 1 to 85)
# Monetary (total spend in USD: $30 to $12,000)
# AvgOrderValue (Monetary / Frequency)
n_rfm = 1000
rfm_ids = np.arange(1001, 1001 + n_rfm)

# Segments:
# 1. Champions (Low R, High F, High M)
# 2. Loyal Customers (Low/Mid R, High F, Mid/High M)
# 3. Promising / Potential Loyalist (Low R, Mid F, Mid M)
# 4. At Risk / Inactive High Value (High R, Mid/High F, High M)
# 5. Hibernating / Budget (High R, Low F, Low M)
# 6. Casual / One-Time Shoppers (Mid/High R, Low F, Low M)

rfm_segs = np.random.choice([0, 1, 2, 3, 4], size=n_rfm, p=[0.15, 0.25, 0.25, 0.15, 0.20])
recency = np.zeros(n_rfm)
frequency = np.zeros(n_rfm)
monetary = np.zeros(n_rfm)

for i in range(n_rfm):
    s = rfm_segs[i]
    if s == 0:  # Champions
        recency[i] = np.random.exponential(12) + 1
        frequency[i] = np.random.normal(35, 10)
        monetary[i] = np.random.normal(5200, 1400)
    elif s == 1:  # Loyal
        recency[i] = np.random.exponential(35) + 5
        frequency[i] = np.random.normal(20, 6)
        monetary[i] = np.random.normal(2600, 700)
    elif s == 2:  # Potential Loyalist / Active Growth
        recency[i] = np.random.exponential(25) + 3
        frequency[i] = np.random.normal(9, 3)
        monetary[i] = np.random.normal(950, 300)
    elif s == 3:  # At Risk / Lapsed Spenders
        recency[i] = np.random.normal(220, 45)
        frequency[i] = np.random.normal(16, 5)
        monetary[i] = np.random.normal(3100, 900)
    else:  # Lost / Low Value
        recency[i] = np.random.normal(280, 50)
        frequency[i] = np.random.normal(2, 1)
        monetary[i] = np.random.normal(180, 80)

recency = np.clip(np.round(recency), 1, 365).astype(int)
frequency = np.clip(np.round(frequency), 1, 100).astype(int)
monetary = np.clip(np.round(monetary, 2), 25.0, 15000.0)
avg_order_value = np.round(monetary / frequency, 2)

df_rfm = pd.DataFrame({
    'CustomerID': rfm_ids,
    'Recency': recency,
    'Frequency': frequency,
    'Monetary': monetary,
    'AvgOrderValue': avg_order_value
})
df_rfm.to_csv("data/online_retail_rfm.csv", index=False)
print("Saved data/online_retail_rfm.csv:", df_rfm.shape)
