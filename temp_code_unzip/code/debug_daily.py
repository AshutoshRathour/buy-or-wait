import pandas as pd
import sys
sys.path.insert(0, 'code')
from data_loader import load_all, get_dataset_dir
from financial_engine import build_exchange_rate_lookup, detect_recurring_events, get_salary_info, get_pending_and_scheduled_events, forecast_balance

dataset_dir = get_dataset_dir()
data = load_all(dataset_dir)
rate_lookup = build_exchange_rate_lookup(data['exchange_rates'])

user_id = 'user_05'
request_date = pd.Timestamp('2025-11-06')
profile = data['profiles'][data['profiles']['user_id'] == user_id].iloc[0]
balance = profile['current_available_balance']
min_bal = profile['minimum_balance_to_keep']
currency = profile['home_currency']

events = data['events']
recurring = detect_recurring_events(events, user_id, request_date)
salary = get_salary_info(events, user_id, request_date)
pending, scheduled = get_pending_and_scheduled_events(events, user_id, request_date)
msg_effects = {}

daily, min_b = forecast_balance(balance, request_date, recurring, scheduled, pending, salary, msg_effects, currency, rate_lookup)

print(f'Start Balance: {balance}')
print(f'Min Balance: {min_b}')
for day in sorted(daily.keys()):
    print(f"{day.strftime('%Y-%m-%d')}: {daily[day]:.2f}")
