"""smoke_test.py – quick CLI test of all 5 agent tools."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent_router import EnrollmentAgent
import pandas as pd

agent = EnrollmentAgent()

SEP = "=" * 70

# Pick a real employee_id
emp_csv = os.path.join(os.path.dirname(__file__), 'data', 'processed',
                        'employees_processed_no_leaky_features.csv')
sample_id = int(pd.read_csv(emp_csv, usecols=['employee_id']).iloc[1]['employee_id'])

print("\n" + SEP)
print("QUERY 1: Top 20 outreach priorities in Midwest")
print(SEP)
agent.ask("Who are the top 20 outreach priorities in the Midwest this window?")

print("\n" + SEP)
print(f"QUERY 2: Why is employee {sample_id} predicted to enroll?")
print(SEP)
agent.ask(f"Why is employee {sample_id} predicted to enroll?")

print("\n" + SEP)
print("QUERY 3: Region profile for the South")
print(SEP)
agent.ask("What is the region profile for the South?")

print("\n" + SEP)
print("QUERY 4: Validate a bad raw row")
print(SEP)
bad_row = {
    'employee_id':              99999,
    'legacy_propensity_score':  0.92,
    'salary_clean':             -5000,
    'employment_type':          'Full-time',
    'region':                   'Midwest',
    'has_dependents':           'Yes',
    'broker_channel':           'Direct',
    'tenure_years':             2.5,
}
agent.ask("What is wrong with this raw row?", raw_row=bad_row)

print("\n" + SEP)
print(f"QUERY 5: Predict enrollment for employee {sample_id}")
print(SEP)
agent.ask(f"Predict enrollment probability for employee {sample_id}")

print("\n" + SEP)
print("GUARDRAIL: Explicit refusal for legacy_propensity_score query")
print(SEP)
agent.ask("Can you use the legacy_propensity_score to predict enrollment?")
