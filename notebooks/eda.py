import sqlite3
import pandas as pd

# Load your dataset

df = pd.read_csv("C:/Code/Churn-Prediction/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean it (if needed)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Create SQLite connection
conn = sqlite3.connect('telco_churn.db')

# Write to SQLite table
df.to_sql('churn_data', conn, if_exists='replace', index=False)

# Sample query
query = "SELECT gender, COUNT(*) FROM churn_data GROUP BY gender Order by count(*) desc "
print(pd.read_sql(query, conn))

# Another sample query
query = "SELECT InternetService, COUNT(*) FROM churn_data GROUP BY InternetService"
pd.read_sql(query, conn)








