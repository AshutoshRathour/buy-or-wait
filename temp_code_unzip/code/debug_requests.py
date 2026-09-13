"""Debug script for request_05."""
import pandas as pd
import sys
sys.path.insert(0, 'code')

from data_loader import load_all, get_dataset_dir
from financial_engine import (
    build_exchange_rate_lookup,
    detect_recurring_events,
    get_salary_info,
    get_pending_and_scheduled_events,
    compute_amount_safe_to_pay,
    find_earliest_full_payment_date,
    forecast_balance,
)
from message_interpreter import get_user_message_effects

dataset_dir = get_dataset_dir()
data = load_all(dataset_dir)
rate_lookup = build_exchange_rate_lookup(data["exchange_rates"])

user_id = "user_05"
request_date = pd.Timestamp("2025-11-06")
requested_amount = 15488.0

profile = data["profiles"][data["profiles"]["user_id"] == user_id].iloc[0]
balance = profile["current_available_balance"]
min_bal = profile["minimum_balance_to_keep"]
currency = profile["home_currency"]

print(f"Balance: {currency} {balance:,.2f}")
print(f"Min balance: {currency} {min_bal:,.2f}")
print(f"Available: {currency} {balance - min_bal:,.2f}")
print(f"Requested: {currency} {requested_amount:,.2f}")
print(f"Expected safe: {currency} 737.00")

# Show only recent events (last 3 months before request)
events = data["events"]
user_events = events[events["user_id"] == user_id].sort_values("settlement_date")
recent = user_events[user_events["settlement_date"] >= pd.Timestamp("2025-09-01")]
print(f"\nRecent events (since 2025-09-01): {len(recent)}")
for _, e in recent.iterrows():
    amt_str = f"{e['amount']:,.2f}" if pd.notna(e["amount"]) else "NaN"
    print(f"  {e['event_id']:15s} {e['status']:10s} {e['direction']:6s} "
          f"{e['category']:20s} {amt_str:>15s} settle={e['settlement_date']}")

# Pending and scheduled
pending, scheduled = get_pending_and_scheduled_events(events, user_id, request_date)
print(f"\nPending debits: {len(pending)}")
for _, p in pending.iterrows():
    amt_str = f"{p['amount']:,.2f}" if pd.notna(p["amount"]) else "NaN"
    print(f"  {p['event_id']:15s} {p['direction']:6s} {p['category']:20s} {amt_str:>15s}")
print(f"Scheduled: {len(scheduled)}")
for _, s in scheduled.iterrows():
    amt_str = f"{s['amount']:,.2f}" if pd.notna(s["amount"]) else "NaN"
    print(f"  {s['event_id']:15s} {s['direction']:6s} {s['category']:20s} {amt_str:>15s} settle={s['settlement_date']}")

# Messages
user_msgs = data["messages"][data["messages"]["user_id"] == user_id]
print(f"\nMessages ({len(user_msgs)}):")
for _, m in user_msgs.iterrows():
    text = str(m["message_text"])[:200]
    print(f"  {m['message_id']}: {text}")

# Message effects
msg_effects = get_user_message_effects(data["messages"], user_id, request_date)
print(f"\nMessage effects: {msg_effects}")

# Recurring
recurring = detect_recurring_events(events, user_id, request_date)
print(f"\nRecurring ({len(recurring)}):")
for r in recurring:
    print(f"  {r['category']:20s} {r['direction']:6s} amt={r['projected_amount']:,.2f} "
          f"freq={r['frequency_days']}d dom={r['day_of_month']}")

# Salary
salary = get_salary_info(events, user_id, request_date)
print(f"\nSalary: {salary}")

# Safe amount
safe = compute_amount_safe_to_pay(
    balance, request_date, recurring, scheduled, pending,
    salary, msg_effects, currency, rate_lookup,
    min_bal, requested_amount
)
print(f"\nComputed safe: {currency} {safe:,.2f}")
print(f"Expected safe: {currency} 737.00")
print(f"Difference: {safe - 737.0:,.2f}")
