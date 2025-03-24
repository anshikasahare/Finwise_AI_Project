
import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker()

categories = {
    "Salary": ["Salary Credit - Infosys Ltd", "Salary from TCS", "HCL Payroll"],
    "Shopping": ["Amazon Purchase", "Flipkart Order", "Myntra"],
    "Food & Dining": ["Swiggy", "Zomato", "Starbucks"],
    "Loan EMI": ["HDFC Loan EMI Payment", "SBI Home Loan", "ICICI EMI"],
    "Utilities": ["Electricity Bill", "Water Bill", "Broadband"],
    "Withdrawals": ["ATM Withdrawal", "Cash at Bank"],
    "Transfers": ["UPI to Raj", "NEFT to Self", "IMPS Transfer"]
}

def generate_transaction(customer_id):
    cat = random.choice(list(categories.keys()))
    desc = random.choice(categories[cat])
    date_formats = [
        fake.date_this_decade().strftime('%Y-%m-%d'),
        fake.date_this_decade().strftime('%d/%m/%Y'),
        fake.date_this_decade().strftime('%b %d, %Y')
    ]
    return {
        "customer_id": customer_id,
        "date": random.choice(date_formats),
        "description": desc,
        "amount": round(random.uniform(100, 50000), 2),
        "category": cat if random.random() > 0.2 else None
    }

def generate_credit_report(customer_id):
    return {
        "customer_id": customer_id,
        "credit_score": random.randint(300, 900),
        "existing_loans": random.randint(0, 5),
        "utilization_ratio": round(random.uniform(0.1, 1.0), 2),
        "missed_payments_12m": random.randint(0, 5),
        "total_outstanding_debt": round(random.uniform(5000, 100000), 2),
        "debt_to_income_ratio": round(random.uniform(0.1, 1.0), 2)
    }

customers = [f"CUST{1000+i}" for i in range(10)]
transactions = []
credit_reports = []

for cust in customers:
    for _ in range(50):
        transactions.append(generate_transaction(cust))
    credit_reports.append(generate_credit_report(cust))

df_trans = pd.DataFrame(transactions)
df_credit = pd.DataFrame(credit_reports)

os.makedirs("data", exist_ok=True)
df_trans.to_csv("data/transactions.csv", index=False)
df_credit.to_csv("data/credit_reports.csv", index=False)

print("Synthetic data generated successfully!")
