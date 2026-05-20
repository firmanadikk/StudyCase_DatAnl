import pandas as pd
df = pd.read_csv("status.csv")

df['churn_date'] = pd.to_datetime(df['churn_date'], unit='ns', errors='coerce')
df['churn_date'] = df['churn_date'].dt.strftime('%Y-%m-%d')
df['churn_date'] = df['churn_date'].fillna('')

df.to_csv("status_clean.csv", index=False)
print(df[['user_id', 'is_churn', 'churn_date']].head(10))