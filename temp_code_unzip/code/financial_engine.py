"""
financial_engine.py — Core deterministic financial calculation engine.
Handles balance reconstruction, recurring expense detection, 90-day forecasting,
and amount_safe_to_pay / earliest_date_for_full_payment computation.
"""

import math
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────
# Currency Conversion
# ──────────────────────────────────────────────────────────────

def build_exchange_rate_lookup(exchange_rates_df):
    """Build a lookup for exchange rates: (date, from_currency, to_currency) -> rate."""
    lookup = {}
    for _, row in exchange_rates_df.iterrows():
        key = (row["rate_date"], row["from_currency"], row["to_currency"])
        lookup[key] = row["rate"]
    return lookup


def convert_currency(amount, from_currency, to_currency, settlement_date, rate_lookup):
    """Convert amount from one currency to another using the closest rate."""
    if from_currency == to_currency:
        return amount
    if pd.isna(amount):
        return 0
    
    # Find available rate dates for this pair
    rate_dates = set()
    for (rd, fc, tc), rate in rate_lookup.items():
        if fc == from_currency and tc == to_currency:
            rate_dates.add(rd)
        elif fc == to_currency and tc == from_currency:
            rate_dates.add(rd)
    
    if not rate_dates:
        # Try indirect conversion via USD
        if from_currency != "USD" and to_currency != "USD":
            usd_amount = convert_currency(amount, from_currency, "USD", settlement_date, rate_lookup)
            if usd_amount is not None:
                return convert_currency(usd_amount, "USD", to_currency, settlement_date, rate_lookup)
        return amount
    
    # Find the closest date
    closest_date = min(rate_dates, key=lambda d: abs((d - settlement_date).days))
    
    # Try direct rate
    direct_key = (closest_date, from_currency, to_currency)
    if direct_key in rate_lookup:
        return amount * rate_lookup[direct_key]
    
    # Try inverse rate
    inverse_key = (closest_date, to_currency, from_currency)
    if inverse_key in rate_lookup:
        return amount / rate_lookup[inverse_key]
    
    return amount


# ──────────────────────────────────────────────────────────────
# Recurring Expense Detection
# ──────────────────────────────────────────────────────────────

def detect_recurring_events(events_df, user_id, request_date):
    """
    Detect recurring income/expense patterns for a user.
    
    IMPORTANT: This detects ALL recurring patterns including salary,
    which will be used for forecasting. The separate salary_info is 
    used for schedule timing; salary from recurring detection is excluded
    from forecast to avoid double-counting.
    """
    user_events = events_df[
        (events_df["user_id"] == user_id) &
        (events_df["status"].isin(["settled", "pending", "scheduled"]))
    ].copy()
    
    # Only settled events before request_date for pattern detection
    settled_events = user_events[
        (user_events["status"] == "settled") &
        (user_events["settlement_date"] <= request_date)
    ].sort_values("settlement_date")
    
    if settled_events.empty:
        return []
    
    recurring = []
    
    # Group by (event_type, category, direction) for recurring detection
    groups = settled_events.groupby(["event_type", "category", "direction"])
    
    for (event_type, category, direction), group in groups:
        if len(group) < 2:
            continue
        
        sorted_group = group.sort_values("settlement_date")
        dates = sorted_group["settlement_date"].tolist()
        amounts = sorted_group["amount"].dropna().tolist()
        
        if len(dates) < 2 or not amounts:
            continue
        
        # Calculate intervals between events
        intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        avg_interval = np.mean(intervals) if intervals else 0
        
        # Check for monthly pattern (25-35 day interval)
        is_monthly = 25 <= avg_interval <= 35
        # Check for biweekly pattern (12-16 day interval)
        is_biweekly = 12 <= avg_interval <= 16
        # Check for weekly pattern (6-8 day interval)
        is_weekly = 6 <= avg_interval <= 8
        
        if not (is_monthly or is_biweekly or is_weekly):
            continue
        
        last_row = sorted_group.iloc[-1]
        last_amount = last_row["amount"] if pd.notna(last_row["amount"]) else np.mean(amounts)
        
        # For variable amounts, use conservative estimate
        amount_std = np.std(amounts) if len(amounts) > 1 else 0
        avg_amount = np.mean(amounts)
        
        is_fixed = amount_std / (avg_amount + 1e-9) < 0.05
        # For debits (expenses), use higher estimate (conservative).
        # For credits (income), use lower estimate (conservative).
        if direction == "debit":
            # Use max of last 3 amounts for conservative debit projection
            recent_amounts = sorted_group["amount"].dropna().tail(3).tolist()
            projected_amount = max(recent_amounts) if recent_amounts else avg_amount
        else:
            # For credits, use the last amount
            projected_amount = last_amount if is_fixed else avg_amount
        
        # Day of month for monthly patterns
        day_of_month = None
        if is_monthly:
            days = [d.day for d in dates]
            day_of_month = int(np.median(days))
        
        freq_days = int(round(avg_interval))
        
        flex = last_row.get("flexibility", "fixed")
        min_amount = last_row.get("minimum_allowed_amount", None)
        if pd.isna(min_amount):
            min_amount = None
        
        recurring.append({
            "category": category,
            "description": last_row.get("description", ""),
            "direction": direction,
            "avg_amount": avg_amount,
            "last_amount": last_amount,
            "projected_amount": projected_amount,
            "frequency_days": freq_days,
            "last_date": dates[-1],
            "day_of_month": day_of_month,
            "event_type": event_type,
            "flexibility": flex if pd.notna(flex) else "fixed",
            "minimum_allowed_amount": min_amount,
            "representative_event_id": last_row["event_id"],
            "currency": last_row.get("currency", ""),
            "is_monthly": is_monthly,
            "is_salary": (category == "salary" and direction == "credit"),
        })
    
    return recurring


# ──────────────────────────────────────────────────────────────
# 90-Day Balance Forecast
# ──────────────────────────────────────────────────────────────

def forecast_balance(
    starting_balance,
    request_date,
    recurring_events,
    scheduled_events,
    pending_debits,
    salary_info,
    message_effects,
    home_currency,
    rate_lookup,
    additional_payment_schedule=None,
    spending_changes=None,
    forecast_days=90,
):
    """
    Forecast the user's daily balance for the next `forecast_days` days.
    """
    end_date = request_date + timedelta(days=forecast_days)
    daily_cf = defaultdict(float)
    
    # 1. Pending debits (reserve them — they will be deducted)
    if pending_debits is not None and not pending_debits.empty:
        for _, evt in pending_debits.iterrows():
            if evt["direction"] == "debit" and pd.notna(evt["amount"]):
                settle_date = evt["settlement_date"] if pd.notna(evt["settlement_date"]) else request_date
                if settle_date >= request_date:
                    amt = evt["amount"]
                    if evt.get("currency", home_currency) != home_currency:
                        amt = convert_currency(amt, evt["currency"], home_currency, settle_date, rate_lookup)
                    daily_cf[settle_date] -= amt
    
    # 2. Scheduled events (both credits and debits)
    if scheduled_events is not None and not scheduled_events.empty:
        for _, evt in scheduled_events.iterrows():
            settle_date = evt["settlement_date"] if pd.notna(evt["settlement_date"]) else evt.get("event_date")
            if pd.isna(settle_date) or settle_date < request_date or settle_date > end_date:
                continue
            if pd.isna(evt["amount"]):
                continue
            
            amt = evt["amount"]
            if evt.get("currency", home_currency) != home_currency:
                amt = convert_currency(amt, evt["currency"], home_currency, settle_date, rate_lookup)
            
            # Per rules: don't count pending credits. For scheduled,
            # count debit as reserved. For credits, only count salary.
            if evt["direction"] == "debit":
                daily_cf[settle_date] -= amt
            elif evt["direction"] == "credit":
                # Count confirmed salary credits
                if evt.get("category") == "salary":
                    daily_cf[settle_date] += amt
                # Other scheduled credits: only if income type
                elif evt.get("event_type") == "income":
                    daily_cf[settle_date] += amt
    
    # 3. Build spending change map
    spending_change_map = {}
    if spending_changes:
        for sc in spending_changes:
            if sc.startswith("stop:"):
                event_id = sc.replace("stop:", "")
                spending_change_map[event_id] = ("stop", 0)
            elif sc.startswith("reduce_to:"):
                parts = sc.replace("reduce_to:", "").split(":")
                if len(parts) == 2:
                    spending_change_map[parts[0]] = ("reduce", float(parts[1]))
    
    # 4. Project recurring events (EXPENSES only — salary handled separately)
    # Track which scheduled event dates we already have to avoid double-counting
    scheduled_categories = set()
    if scheduled_events is not None and not scheduled_events.empty:
        for _, evt in scheduled_events.iterrows():
            sd = evt["settlement_date"]
            if pd.notna(sd) and request_date <= sd <= end_date:
                scheduled_categories.add((evt.get("category", ""), sd))
    
    for rec in recurring_events:
        # Skip salary — handled in step 5
        if rec.get("is_salary", False):
            continue
        
        # Skip if stopped
        if rec["representative_event_id"] in spending_change_map:
            action, _ = spending_change_map[rec["representative_event_id"]]
            if action == "stop":
                continue
        
        # Project next occurrences
        last_date = rec["last_date"]
        freq = rec["frequency_days"]
        
        if rec["is_monthly"] and rec["day_of_month"]:
            current = last_date
            for _ in range(6):
                if current.month == 12:
                    next_date = current.replace(year=current.year + 1, month=1, day=min(rec["day_of_month"], 28))
                else:
                    next_day = min(rec["day_of_month"], 28)
                    try:
                        next_date = current.replace(month=current.month + 1, day=next_day)
                    except ValueError:
                        next_date = current.replace(month=current.month + 1, day=28)
                
                if next_date <= request_date:
                    current = next_date
                    continue
                if next_date > end_date:
                    break
                
                # Check if this is already covered by a scheduled event
                if (rec["category"], next_date) in scheduled_categories:
                    current = next_date
                    continue
                
                amt = rec["projected_amount"]
                
                # Apply spending change
                if rec["representative_event_id"] in spending_change_map:
                    action, new_amt = spending_change_map[rec["representative_event_id"]]
                    if action == "reduce":
                        amt = new_amt
                
                # Apply rent increase
                if rec["category"] == "rent" and message_effects.get("rent_increase_pct"):
                    amt = amt * (1 + message_effects["rent_increase_pct"] / 100.0)
                
                if rec.get("currency", home_currency) != home_currency:
                    amt = convert_currency(amt, rec["currency"], home_currency, next_date, rate_lookup)
                
                if rec["direction"] == "debit":
                    daily_cf[next_date] -= amt
                elif rec["direction"] == "credit":
                    daily_cf[next_date] += amt
                
                current = next_date
        else:
            # Project by fixed interval
            next_date = last_date + timedelta(days=freq)
            for _ in range(20):
                if next_date > end_date:
                    break
                if next_date > request_date:
                    # Check if covered by scheduled event
                    if (rec["category"], next_date) not in scheduled_categories:
                        amt = rec["projected_amount"]
                        if rec["representative_event_id"] in spending_change_map:
                            action, new_amt = spending_change_map[rec["representative_event_id"]]
                            if action == "reduce":
                                amt = new_amt
                        if rec.get("currency", home_currency) != home_currency:
                            amt = convert_currency(amt, rec["currency"], home_currency, next_date, rate_lookup)
                        if rec["direction"] == "debit":
                            daily_cf[next_date] -= amt
                        elif rec["direction"] == "credit":
                            daily_cf[next_date] += amt
                next_date = next_date + timedelta(days=freq)
    
    # 5. Project salary (the primary, controlled salary projection)
    if salary_info and not message_effects.get("salary_ended", False):
        salary_amount = salary_info.get("amount", 0)
        salary_day = salary_info.get("day_of_month", 15)
        salary_currency = salary_info.get("currency", home_currency)
        
        # Apply message overrides
        if message_effects.get("salary_override"):
            salary_amount = message_effects["salary_override"]
        
        # Check for scheduled salary already counted in step 2
        scheduled_salary_dates = set()
        if scheduled_events is not None and not scheduled_events.empty:
            sal_scheduled = scheduled_events[
                (scheduled_events.get("category", pd.Series(dtype=str)) == "salary") &
                (scheduled_events["direction"] == "credit")
            ] if "category" in scheduled_events.columns else pd.DataFrame()
            for _, se in sal_scheduled.iterrows():
                sd = se["settlement_date"]
                if pd.notna(sd):
                    scheduled_salary_dates.add(sd)
        
        # Project salary for 4 months
        for month_offset in range(0, 4):
            year = request_date.year
            month = request_date.month + month_offset
            if month > 12:
                year += (month - 1) // 12
                month = ((month - 1) % 12) + 1
            
            sal_date_override = message_effects.get("salary_date_override")
            if sal_date_override and month_offset == 0:
                sal_date = sal_date_override
            else:
                try:
                    sal_date = datetime(year, month, min(salary_day, 28))
                except ValueError:
                    sal_date = datetime(year, month, 28)
            
            if sal_date <= request_date or sal_date > end_date:
                continue
            
            # Skip if already in scheduled events
            if sal_date in scheduled_salary_dates:
                continue
            
            amt = salary_amount
            if salary_currency != home_currency:
                amt = convert_currency(amt, salary_currency, home_currency, sal_date, rate_lookup)
            
            daily_cf[sal_date] += amt
    
    # 6. Confirmed income from messages
    for income in message_effects.get("confirmed_income", []):
        inc_date = income["date"]
        if request_date < inc_date <= end_date:
            amt = income["amount"]
            inc_currency = income.get("currency", home_currency)
            if inc_currency and inc_currency != home_currency:
                amt = convert_currency(amt, inc_currency, home_currency, inc_date, rate_lookup)
            daily_cf[inc_date] += amt
    
    # 7. Salary arrears (one-time)
    if message_effects.get("salary_arrears"):
        arrears_date = message_effects.get("salary_date_override")
        if arrears_date and request_date < arrears_date <= end_date:
            daily_cf[arrears_date] += message_effects["salary_arrears"]
    
    # 8. Proposed payment schedule
    if additional_payment_schedule:
        for pay_date, pay_amount in additional_payment_schedule:
            if isinstance(pay_date, str):
                pay_date = datetime.strptime(pay_date, "%Y-%m-%d")
            if pay_date >= request_date and pay_date <= end_date:
                daily_cf[pay_date] -= pay_amount
    
    # 9. Compute daily balance
    daily_balance = {}
    balance = starting_balance
    min_balance = balance
    
    current_date = request_date
    while current_date <= end_date:
        if current_date in daily_cf:
            balance += daily_cf[current_date]
        daily_balance[current_date] = balance
        min_balance = min(min_balance, balance)
        current_date += timedelta(days=1)
    
    return daily_balance, min_balance


# ──────────────────────────────────────────────────────────────
# Safety Analysis
# ──────────────────────────────────────────────────────────────

def compute_amount_safe_to_pay(
    starting_balance,
    request_date,
    recurring_events,
    scheduled_events,
    pending_debits,
    salary_info,
    message_effects,
    home_currency,
    rate_lookup,
    minimum_balance,
    requested_amount,
    forecast_days=90,
):
    """
    Compute the maximum amount the user can safely pay on request_date
    while maintaining minimum_balance throughout the 90-day forecast.
    Uses binary search for efficiency.
    """
    def is_safe(amount):
        if amount < 0:
            return True
        schedule = [(request_date, amount)] if amount > 0 else []
        _, min_bal = forecast_balance(
            starting_balance, request_date, recurring_events,
            scheduled_events, pending_debits, salary_info,
            message_effects, home_currency, rate_lookup,
            additional_payment_schedule=schedule,
            forecast_days=forecast_days,
        )
        return min_bal >= minimum_balance
    
    # Check if zero payment is safe
    if not is_safe(0):
        return 0
    
    # Check if full amount is safe
    if is_safe(requested_amount):
        return requested_amount
    
    # Binary search
    low, high = 0.0, requested_amount
    for _ in range(50):
        mid = (low + high) / 2
        if is_safe(mid):
            low = mid
        else:
            high = mid
        if high - low < 0.01:
            break
    
    # Round down to be safe
    return math.floor(low * 100) / 100


def find_earliest_full_payment_date(
    starting_balance,
    request_date,
    recurring_events,
    scheduled_events,
    pending_debits,
    salary_info,
    message_effects,
    home_currency,
    rate_lookup,
    minimum_balance,
    requested_amount,
    forecast_days=90,
):
    """
    Find earliest date when full requested_amount can be safely paid.
    Returns the date, or None if not possible within forecast period.
    """
    end_date = request_date + timedelta(days=forecast_days)
    
    current_date = request_date
    while current_date <= end_date:
        schedule = [(current_date, requested_amount)]
        _, min_bal = forecast_balance(
            starting_balance, request_date, recurring_events,
            scheduled_events, pending_debits, salary_info,
            message_effects, home_currency, rate_lookup,
            additional_payment_schedule=schedule,
            forecast_days=forecast_days,
        )
        if min_bal >= minimum_balance:
            return current_date
        
        current_date += timedelta(days=1)
    
    return None


def get_salary_info(events_df, user_id, request_date):
    """
    Extract salary information for a user from their financial events.
    Returns a dict with salary amount, day of month, and currency.
    """
    user_events = events_df[events_df["user_id"] == user_id].copy()
    
    salary_events = user_events[
        (user_events["event_type"] == "income") &
        (user_events["category"] == "salary") &
        (user_events["direction"] == "credit") &
        (user_events["status"].isin(["settled", "scheduled"]))
    ].sort_values("settlement_date")
    
    if salary_events.empty:
        return None
    
    past_salaries = salary_events[salary_events["settlement_date"] <= request_date]
    future_salaries = salary_events[salary_events["settlement_date"] > request_date]
    
    if not past_salaries.empty:
        last_salary = past_salaries.iloc[-1]
        
        if not future_salaries.empty:
            next_salary = future_salaries.iloc[0]
            return {
                "amount": last_salary["amount"] if pd.notna(last_salary["amount"]) else 0,
                "day_of_month": last_salary["settlement_date"].day,
                "currency": last_salary["currency"],
                "next_date": next_salary["settlement_date"],
                "next_amount": next_salary["amount"] if pd.notna(next_salary["amount"]) else None,
            }
        
        return {
            "amount": last_salary["amount"] if pd.notna(last_salary["amount"]) else 0,
            "day_of_month": last_salary["settlement_date"].day,
            "currency": last_salary["currency"],
            "next_date": None,
            "next_amount": None,
        }
    elif not future_salaries.empty:
        next_salary = future_salaries.iloc[0]
        return {
            "amount": next_salary["amount"] if pd.notna(next_salary["amount"]) else 0,
            "day_of_month": next_salary["settlement_date"].day,
            "currency": next_salary["currency"],
            "next_date": next_salary["settlement_date"],
            "next_amount": next_salary["amount"] if pd.notna(next_salary["amount"]) else None,
        }
    
    return None


def get_pending_and_scheduled_events(events_df, user_id, request_date):
    """
    Get pending and scheduled events for a user.
    Returns (pending_debits, scheduled_events) DataFrames.
    
    Also includes failed debit events that represent outstanding bills.
    """
    user_events = events_df[events_df["user_id"] == user_id].copy()
    
    # Pending debits
    pending = user_events[
        (user_events["status"] == "pending") &
        (user_events["direction"] == "debit")
    ]
    
    # Scheduled events  
    scheduled = user_events[
        (user_events["status"] == "scheduled")
    ]
    
    # Failed debits (bills still outstanding — treat as pending)
    failed_debits = user_events[
        (user_events["status"] == "failed") &
        (user_events["direction"] == "debit")
    ].copy()
    if not failed_debits.empty:
        # Treat failed debits as pending (the bill is still due)
        failed_debits.loc[:, "status"] = "pending"
        pending = pd.concat([pending, failed_debits], ignore_index=True)
    
    return pending, scheduled


def identify_flexible_recurring(recurring_events, profile, events_df, user_id):
    """
    Identify recurring expenses that can be stopped or reduced.
    """
    changes = []
    
    reduce_categories = profile.get("expense_categories_user_is_willing_to_reduce", [])
    if isinstance(reduce_categories, str):
        reduce_categories = [c.strip() for c in reduce_categories.split("|") if c.strip()]
    
    stop_categories = profile.get("expense_categories_user_is_willing_to_stop", [])
    if isinstance(stop_categories, str):
        stop_categories = [c.strip() for c in stop_categories.split("|") if c.strip()]
    
    protected = profile.get("expense_categories_to_protect", [])
    if isinstance(protected, str):
        protected = [c.strip() for c in protected.split("|") if c.strip()]
    
    for rec in recurring_events:
        if rec["direction"] != "debit":
            continue
        
        category = rec["category"]
        flexibility = rec["flexibility"]
        
        if category in protected:
            continue
        
        if flexibility in ("stoppable", "reducible_or_stoppable") and category in stop_categories:
            changes.append({
                "action": "stop",
                "event_id": rec["representative_event_id"],
                "category": category,
                "monthly_savings": rec["projected_amount"],
                "description": rec["description"],
            })
        
        if flexibility in ("reducible", "reducible_or_stoppable") and category in reduce_categories:
            min_amount = rec.get("minimum_allowed_amount", 0) or 0
            savings = rec["projected_amount"] - min_amount
            if savings > 0:
                changes.append({
                    "action": "reduce",
                    "event_id": rec["representative_event_id"],
                    "category": category,
                    "new_amount": min_amount,
                    "monthly_savings": savings,
                    "description": rec["description"],
                })
    
    changes.sort(key=lambda x: x["monthly_savings"], reverse=True)
    return changes
