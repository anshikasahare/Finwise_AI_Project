
import pandas as pd
import sqlite3
import os
import re
from datetime import datetime

df_trans = pd.read_csv("data/transactions.csv")
df_credit = pd.read_csv("data/credit_reports.csv")

def standardize_date(date_str):
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%b %d, %Y'):
        try:
            return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return None

df_trans["date"] = df_trans["date"].apply(standardize_date)

def clean_description(desc):
    if pd.isna(desc):
        return None
    desc = desc.lower()
    desc = re.sub(r'[^a-z0-9\s]', '', desc)
    return desc

df_trans["description"] = df_trans["description"].apply(clean_description)

df_trans["amount"] = df_trans["amount"].fillna(0)
df_trans["description"] = df_trans["description"].fillna("unknown")
df_trans["category"] = df_trans["category"].fillna("uncategorized")

os.makedirs("db", exist_ok=True)
conn = sqlite3.connect("db/finwise.db")

df_trans.to_sql("transactions", conn, if_exists="replace", index=False)
df_credit.to_sql("credit_reports", conn, if_exists="replace", index=False)

conn.close()
print("ETL pipeline completed. Data stored in db/finwise.db")
