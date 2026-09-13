"""
plan_evaluator.py — Evaluate and rank payment plans for the Buy or Wait? challenge.
Implements the 6-criteria ranking system and generates the final recommendation.
"""

import math
from datetime import datetime, timedelta
import pandas as pd

from financial_engine import (
    forecast_balance, compute_amount_safe_to_pay,
    find_earliest_full_payment_date, identify_flexible_recurring,
)


def generate_installment_schedule(option):
    """
    Generate the payment schedule for an installment option.
    Returns list of (date, amount) tuples.
    """
    first_date = option["first_payment_date"]
    if isinstance(first_date, str):
        first_date = datetime.strptime(first_date, "%Y-%m-%d")
    
    n_payments = int(option["number_of_payments"])
    payment_amount = option["payment_amount"]
    freq_days = option["payment_frequency_days"]
    
    schedule = []
    current_date = first_date
    for i in range(n_payments):
        schedule.append((current_date, payment_amount))
        if pd.notna(freq_days) and freq_days > 0:
            current_date = current_date + timedelta(days=int(freq_days))
    
    return schedule


def is_plan_safe(
    schedule,
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
    spending_changes=None,
):
    """
    Check if a payment schedule keeps the balance above minimum throughout 90 days.
    """
    _, min_bal = forecast_balance(
        starting_balance, request_date, recurring_events,
        scheduled_events, pending_debits, salary_info,
        message_effects, home_currency, rate_lookup,
        additional_payment_schedule=schedule,
        spending_changes=spending_changes,
    )
    return min_bal >= minimum_balance


def evaluate_payment_options(
    request,
    profile,
    payment_options_df,
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
    amount_safe_to_pay,
    earliest_full_payment,
    flexible_changes,
):
    """
    Evaluate all payment options and spending change combinations.
    Returns the best recommendation.
    """
    requested_amount = request["requested_amount"]
    desired_completion = request["desired_completion_date"]
    allows_partial = request["allows_partial_payment"]
    
    # Get user's acceptable payment methods
    methods = profile["payment_methods_user_will_consider"]
    max_installment_months = profile.get("max_installment_months", None)
    if pd.isna(max_installment_months):
        max_installment_months = None
    
    # Get payment options for this request
    request_id = request["request_id"]
    options = payment_options_df[payment_options_df["request_id"] == request_id].copy()
    
    # ── Collect all candidate plans ──
    candidates = []
    
    # ── Plan 1: Full payment today (no spending changes) ──
    if "full_payment" in methods and amount_safe_to_pay >= requested_amount:
        plan_schedule = [(request_date, requested_amount)]
        candidates.append({
            "status": "affordable_now",
            "method": "full_payment",
            "plan_str": f"{request_date.strftime('%Y-%m-%d')}:{requested_amount}",
            "earliest_date": request_date,
            "spending_changes": "none",
            "total_paid": requested_amount,
            "start_date": request_date,
            "num_payments": 1,
            "option_id": None,
            "completes_by_deadline": True,
            "needs_spending_changes": False,
        })
    
    # ── Plan 2: Full payment today WITH spending changes ──
    if "full_payment" in methods and amount_safe_to_pay < requested_amount:
        # Try spending changes to make full payment work
        for changes_combo in _generate_spending_change_combos(flexible_changes, max_changes=3):
            change_strs = _format_spending_changes(changes_combo)
            schedule = [(request_date, requested_amount)]
            
            safe = is_plan_safe(
                schedule, starting_balance, request_date,
                recurring_events, scheduled_events, pending_debits,
                salary_info, message_effects, home_currency, rate_lookup,
                minimum_balance, spending_changes=change_strs,
            )
            
            if safe:
                completes = True
                candidates.append({
                    "status": "affordable_with_plan",
                    "method": "full_payment",
                    "plan_str": f"{request_date.strftime('%Y-%m-%d')}:{requested_amount}",
                    "earliest_date": earliest_full_payment if earliest_full_payment else request_date,
                    "spending_changes": "|".join(change_strs),
                    "total_paid": requested_amount,
                    "start_date": request_date,
                    "num_payments": 1,
                    "option_id": None,
                    "completes_by_deadline": completes,
                    "needs_spending_changes": True,
                })
                break  # Use the first valid combo (already sorted by savings)
    
    # ── Plan 3: Installments ──
    if "installments" in methods:
        installment_options = options[options["payment_method"] == "installments"]
        
        for _, opt in installment_options.iterrows():
            # Check if number of payments within user's max installment months
            if max_installment_months is not None:
                n_payments = int(opt["number_of_payments"])
                freq_days = opt["payment_frequency_days"] if pd.notna(opt["payment_frequency_days"]) else 30
                total_months = (n_payments * freq_days) / 30
                if total_months > max_installment_months + 0.5:  # Small tolerance
                    continue
            
            schedule = generate_installment_schedule(opt)
            
            if not schedule:
                continue
            
            # Check if plan completes by deadline
            last_payment_date = schedule[-1][0]
            completes = last_payment_date <= desired_completion
            
            # Check safety without spending changes
            safe = is_plan_safe(
                schedule, starting_balance, request_date,
                recurring_events, scheduled_events, pending_debits,
                salary_info, message_effects, home_currency, rate_lookup,
                minimum_balance,
            )
            
            if safe:
                plan_parts = [f"{d.strftime('%Y-%m-%d')}:{a}" for d, a in schedule]
                candidates.append({
                    "status": "affordable_with_plan",
                    "method": "installments",
                    "plan_str": "|".join(plan_parts),
                    "earliest_date": earliest_full_payment,
                    "spending_changes": "none",
                    "total_paid": opt["total_payable_amount"],
                    "start_date": schedule[0][0],
                    "num_payments": len(schedule),
                    "option_id": opt["payment_option_id"],
                    "completes_by_deadline": completes,
                    "needs_spending_changes": False,
                })
            else:
                # Try with spending changes
                for changes_combo in _generate_spending_change_combos(flexible_changes, max_changes=3):
                    change_strs = _format_spending_changes(changes_combo)
                    safe = is_plan_safe(
                        schedule, starting_balance, request_date,
                        recurring_events, scheduled_events, pending_debits,
                        salary_info, message_effects, home_currency, rate_lookup,
                        minimum_balance, spending_changes=change_strs,
                    )
                    if safe:
                        plan_parts = [f"{d.strftime('%Y-%m-%d')}:{a}" for d, a in schedule]
                        candidates.append({
                            "status": "affordable_with_plan",
                            "method": "installments",
                            "plan_str": "|".join(plan_parts),
                            "earliest_date": earliest_full_payment,
                            "spending_changes": "|".join(change_strs),
                            "total_paid": opt["total_payable_amount"],
                            "start_date": schedule[0][0],
                            "num_payments": len(schedule),
                            "option_id": opt["payment_option_id"],
                            "completes_by_deadline": completes,
                            "needs_spending_changes": True,
                        })
                        break
    
    # ── Plan 4: Partial payment ──
    if (allows_partial and "partial_payment" in methods and
        0 < amount_safe_to_pay < requested_amount and
        earliest_full_payment is not None and
        earliest_full_payment <= desired_completion):
        
        remaining = round(requested_amount - amount_safe_to_pay, 2)
        schedule = [
            (request_date, amount_safe_to_pay),
            (earliest_full_payment, remaining),
        ]
        
        safe = is_plan_safe(
            schedule, starting_balance, request_date,
            recurring_events, scheduled_events, pending_debits,
            salary_info, message_effects, home_currency, rate_lookup,
            minimum_balance,
        )
        
        if safe:
            plan_parts = [f"{d.strftime('%Y-%m-%d')}:{a}" for d, a in schedule]
            candidates.append({
                "status": "affordable_with_plan",
                "method": "partial_payment",
                "plan_str": "|".join(plan_parts),
                "earliest_date": earliest_full_payment,
                "spending_changes": "none",
                "total_paid": requested_amount,
                "start_date": request_date,
                "num_payments": 2,
                "option_id": None,
                "completes_by_deadline": True,
                "needs_spending_changes": False,
            })
    
    # ── Plan 5: Wait ──
    if ("full_payment" in methods and 
        earliest_full_payment is not None and
        earliest_full_payment > request_date):
        
        schedule = [(earliest_full_payment, requested_amount)]
        safe = is_plan_safe(
            schedule, starting_balance, request_date,
            recurring_events, scheduled_events, pending_debits,
            salary_info, message_effects, home_currency, rate_lookup,
            minimum_balance,
        )
        
        if safe:
            completes = earliest_full_payment <= desired_completion
            candidates.append({
                "status": "affordable_later" if completes else "affordable_later",
                "method": "wait",
                "plan_str": f"{earliest_full_payment.strftime('%Y-%m-%d')}:{requested_amount}",
                "earliest_date": earliest_full_payment,
                "spending_changes": "none",
                "total_paid": requested_amount,
                "start_date": earliest_full_payment,
                "num_payments": 1,
                "option_id": None,
                "completes_by_deadline": completes,
                "needs_spending_changes": False,
            })
    
    # ── Rank candidates using the 6-criteria system ──
    if not candidates:
        return {
            "status": "not_affordable",
            "method": "not_recommended",
            "plan_str": "none",
            "earliest_date": earliest_full_payment,
            "spending_changes": "none",
        }
    
    def plan_sort_key(plan):
        return (
            0 if plan["completes_by_deadline"] else 1,
            0 if not plan["needs_spending_changes"] else 1,
            plan["total_paid"],
            plan["start_date"],
            plan["num_payments"],
            plan.get("option_id", "") or "zzz",
        )
    
    candidates.sort(key=plan_sort_key)
    
    best = candidates[0]
    return best


def _generate_spending_change_combos(flexible_changes, max_changes=3):
    """
    Generate combinations of spending changes, up to max_changes.
    Each combo is a list of change dicts. Returns combos sorted by total savings.
    """
    if not flexible_changes:
        return []
    
    # Generate single changes
    combos = [[c] for c in flexible_changes[:max_changes]]
    
    # Generate pairs (if different event_ids)
    for i in range(len(flexible_changes)):
        for j in range(i+1, len(flexible_changes)):
            if flexible_changes[i]["event_id"] != flexible_changes[j]["event_id"]:
                combos.append([flexible_changes[i], flexible_changes[j]])
    
    # Generate triples
    for i in range(len(flexible_changes)):
        for j in range(i+1, len(flexible_changes)):
            for k in range(j+1, len(flexible_changes)):
                event_ids = {flexible_changes[i]["event_id"], flexible_changes[j]["event_id"], flexible_changes[k]["event_id"]}
                if len(event_ids) == 3:
                    combos.append([flexible_changes[i], flexible_changes[j], flexible_changes[k]])
    
    # Filter: no stop and reduce on same event
    valid = []
    for combo in combos:
        if len(combo) <= max_changes:
            event_ids = [c["event_id"] for c in combo]
            if len(event_ids) == len(set(event_ids)):
                valid.append(combo)
    
    # Sort by total savings descending
    valid.sort(key=lambda c: sum(x["monthly_savings"] for x in c), reverse=True)
    
    return valid


def _format_spending_changes(changes_combo):
    """Format a list of spending change dicts into string actions."""
    result = []
    for change in changes_combo:
        if change["action"] == "stop":
            result.append(f"stop:{change['event_id']}")
        elif change["action"] == "reduce":
            result.append(f"reduce_to:{change['event_id']}:{change['new_amount']}")
    return result


def generate_explanation(
    request, profile, best_plan, amount_safe_to_pay,
    earliest_full_payment, minimum_balance,
):
    """Generate a concise decision explanation."""
    requested_amount = request["requested_amount"]
    currency = profile["home_currency"]
    method = best_plan["method"]
    status = best_plan["status"]
    
    # Format amounts nicely
    def fmt_amt(a):
        if a is None:
            return "N/A"
        if a == int(a):
            return f"{currency} {int(a):,}"
        return f"{currency} {a:,.2f}"
    
    if method == "full_payment" and status == "affordable_now":
        return (
            f"Pay {fmt_amt(requested_amount)} today. "
            f"This leaves at least {fmt_amt(minimum_balance)} available over the next 90 days."
        )
    
    if method == "full_payment" and status == "affordable_with_plan":
        sc = best_plan.get("spending_changes", "none")
        if sc != "none":
            changes_desc = _describe_spending_changes(sc)
            return (
                f"{changes_desc}, then pay {fmt_amt(requested_amount)} today. "
                f"This leaves at least {fmt_amt(minimum_balance)} available."
            )
        return (
            f"Pay {fmt_amt(requested_amount)} today. "
            f"This leaves at least {fmt_amt(minimum_balance)} available."
        )
    
    if method == "installments":
        parts = best_plan["plan_str"].split("|")
        n = len(parts)
        first_part = parts[0].split(":")
        first_date = first_part[0]
        payment_amt = first_part[1]
        return (
            f"Use {n} installments of {currency} {payment_amt}, starting {first_date}. "
            f"This leaves at least {fmt_amt(minimum_balance)} available."
        )
    
    if method == "partial_payment":
        parts = best_plan["plan_str"].split("|")
        first = parts[0].split(":")
        second = parts[1].split(":")
        return (
            f"Pay {currency} {first[1]} today and the remaining {currency} {second[1]} "
            f"on {second[0]}. This completes the full request and keeps the "
            f"{fmt_amt(minimum_balance)} minimum protected."
        )
    
    if method == "wait":
        if earliest_full_payment:
            efp_str = earliest_full_payment.strftime("%Y-%m-%d")
            return (
                f"Pay {fmt_amt(requested_amount)} in full on {efp_str}. "
                f"Paying earlier would take the balance below the "
                f"{fmt_amt(minimum_balance)} minimum."
            )
        return (
            f"Wait for more funds before making this payment. "
            f"Paying now would take the balance below the "
            f"{fmt_amt(minimum_balance)} minimum."
        )
    
    if method == "not_recommended":
        if earliest_full_payment and earliest_full_payment <= request["desired_completion_date"]:
            return (
                f"Do not proceed with the {fmt_amt(requested_amount)} request. "
                f"Although {fmt_amt(amount_safe_to_pay)} is available today, "
                f"the full amount cannot be completed safely within 90 days."
            )
        return (
            f"Do not make this payment by {request['desired_completion_date'].strftime('%d %B %Y')}. "
            f"None of the available options keeps the {fmt_amt(minimum_balance)} minimum protected."
        )
    
    return f"Financial analysis completed for {fmt_amt(requested_amount)} request."


def _describe_spending_changes(changes_str):
    """Convert spending change string to human-readable description."""
    parts = changes_str.split("|")
    descriptions = []
    for p in parts:
        if p.startswith("stop:"):
            event_id = p.replace("stop:", "")
            descriptions.append(f"Stop {event_id}")
        elif p.startswith("reduce_to:"):
            components = p.replace("reduce_to:", "").split(":")
            if len(components) == 2:
                descriptions.append(f"Reduce {components[0]} to {components[1]}")
    
    if len(descriptions) == 1:
        return descriptions[0]
    elif len(descriptions) == 2:
        return f"{descriptions[0]} and {descriptions[1].lower()}"
    else:
        return ", ".join(descriptions[:-1]) + f", and {descriptions[-1].lower()}"
