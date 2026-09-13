"""Debug trace for specific sample requests."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datetime import datetime
from data_loader import load_all, get_dataset_dir
from message_interpreter import get_user_message_effects
from image_extractor import resolve_blank_amounts
from financial_engine import (
    build_exchange_rate_lookup, compute_amount_safe_to_pay,
    find_earliest_full_payment_date, get_salary_info,
    get_pending_and_scheduled_events, detect_recurring_events,
    forecast_balance, convert_currency,
)

dataset_dir = get_dataset_dir()
data = load_all(dataset_dir)
data["events"] = resolve_blank_amounts(data["events"], data["images"])
rate_lookup = build_exchange_rate_lookup(data["exchange_rates"])

sample = pd.read_csv(os.path.join(dataset_dir, 'sample_requests.csv'))

for req_id in ['request_10', 'request_15']:
    req = sample[sample['request_id'] == req_id].iloc[0]
    user_id = req['user_id']
    request_date = pd.to_datetime(req['request_date'])
    
    profile = data['profiles'][data['profiles']['user_id'] == user_id].iloc[0]
    events = data['events']
    
    balance = profile['current_available_balance']
    min_balance = profile['minimum_balance_to_keep']
    home_currency = profile['home_currency']
    requested_amount = req['requested_amount']
    
    print(f"\n{'='*60}")
    print(f"DEBUG: {req_id} (user: {user_id})")
    print(f"{'='*60}")
    print(f"Balance: {balance}, Min: {min_balance}, Excess: {balance - min_balance}")
    print(f"Requested: {requested_amount}, Currency: {home_currency}")
    print(f"Request date: {request_date}")
    
    msg_effects = get_user_message_effects(data["messages"], user_id, request_date)
    salary_info = get_salary_info(events, user_id, request_date)
    pending, scheduled = get_pending_and_scheduled_events(events, user_id, request_date)
    recurring = detect_recurring_events(events, user_id, request_date)
    
    print(f"\nSalary info: {salary_info}")
    print(f"Pending debits: {len(pending)}")
    print(f"Scheduled events: {len(scheduled)}")
    print(f"Recurring patterns: {len(recurring)}")
    
    total_monthly_debit = 0
    print("\nRecurring DEBIT categories:")
    for r in recurring:
        if r['direction'] == 'debit':
            monthly_equiv = r['projected_amount'] * 30 / r['frequency_days']
            total_monthly_debit += monthly_equiv
            print(f"  {r['category']}: {r['projected_amount']:.2f} every {r['frequency_days']}d (day={r.get('day_of_month', '?')}) [monthly equiv: {monthly_equiv:.2f}]")
    print(f"  TOTAL monthly debit: {total_monthly_debit:.2f}")
    
    print("\nRecurring CREDIT categories:")
    for r in recurring:
        if r['direction'] == 'credit':
            print(f"  {r['category']}: {r['projected_amount']:.2f} every {r['frequency_days']}d")
    
    # Compute amount safe
    amount_safe = compute_amount_safe_to_pay(
        balance, request_date, recurring, scheduled, pending,
        salary_info, msg_effects, home_currency, rate_lookup,
        min_balance, requested_amount,
    )
    print(f"\nComputed amount_safe: {amount_safe}")
    print(f"Expected amount_safe: {req['amount_safe_to_pay']}")
    
    # Forecast with zero payment to see minimum balance
    daily_bal, min_bal = forecast_balance(
        balance, request_date, recurring, scheduled, pending,
        salary_info, msg_effects, home_currency, rate_lookup,
    )
    print(f"Min balance (no payment): {min_bal:.2f}")
    print(f"Safe to spend: {min_bal - min_balance:.2f}")
    
    # Show daily balance for key dates
    print("\nDaily balance at key dates (first 30 days):")
    sorted_dates = sorted(daily_bal.keys())[:30]
    for d in sorted_dates:
        if daily_bal[d] < balance * 0.8 or d.day in [1, 15]:
            print(f"  {d.strftime('%Y-%m-%d')}: {daily_bal[d]:.2f}")
