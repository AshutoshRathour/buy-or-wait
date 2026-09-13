"""
validate.py — Run the financial agent on sample_requests.csv and compare with ground truth.
"""

import os
import sys
import csv
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_all, get_dataset_dir, load_sample_requests
from message_interpreter import get_user_message_effects, interpret_all_messages
from financial_engine import (
    build_exchange_rate_lookup,
    compute_amount_safe_to_pay,
    find_earliest_full_payment_date,
    get_salary_info,
    get_pending_and_scheduled_events,
    detect_recurring_events,
    identify_flexible_recurring,
)
from plan_evaluator import evaluate_payment_options, generate_explanation
from main import process_request


def validate():
    print("=" * 70)
    print("VALIDATION: Running on sample_requests.csv")
    print("=" * 70)

    dataset_dir = get_dataset_dir()
    data = load_all(dataset_dir)
    rate_lookup = build_exchange_rate_lookup(data["exchange_rates"])

    sample_df = load_sample_requests(dataset_dir)
    print(f"Loaded {len(sample_df)} sample requests\n")

    results = []
    for idx, (_, request) in enumerate(sample_df.iterrows()):
        try:
            result = process_request(request, data, rate_lookup)
            results.append(result)
        except Exception as e:
            print(f"ERROR {request['request_id']}: {e}")
            results.append({
                "request_id": request["request_id"],
                "amount_safe_to_pay": 0,
                "affordability_status": "not_affordable",
                "recommended_payment_method": "not_recommended",
                "payment_plan": "none",
                "earliest_date_for_full_payment": "",
                "spending_changes_needed": "none",
                "decision_explanation": "Error",
            })

    # Compare with ground truth
    output_df = pd.DataFrame(results)

    print(f"\n{'Req ID':<14} {'Status Match':<16} {'Method Match':<16} "
          f"{'Safe Amt':>12} {'Expected':>12} {'Err%':>8}")
    print("-" * 80)

    correct_status = 0
    correct_method = 0
    total = 0
    total_amt_error = 0

    for _, sample_row in sample_df.iterrows():
        req_id = sample_row["request_id"]
        pred_rows = output_df[output_df["request_id"] == req_id]
        if pred_rows.empty:
            continue

        pred = pred_rows.iloc[0]
        total += 1

        # Status
        exp_status = sample_row["affordability_status"]
        pred_status = pred["affordability_status"]
        status_ok = exp_status == pred_status
        if status_ok:
            correct_status += 1

        # Method
        exp_method = sample_row["recommended_payment_method"]
        pred_method = pred["recommended_payment_method"]
        method_ok = exp_method == pred_method
        if method_ok:
            correct_method += 1

        # Amount
        exp_amt = sample_row["amount_safe_to_pay"]
        pred_amt = pred["amount_safe_to_pay"]
        if exp_amt > 0:
            err_pct = abs(pred_amt - exp_amt) / exp_amt * 100
        elif pred_amt == 0:
            err_pct = 0
        else:
            err_pct = 100

        total_amt_error += err_pct

        s_mark = "OK" if status_ok else f"X ({pred_status})"
        m_mark = "OK" if method_ok else f"X ({pred_method})"
        print(f"  {req_id:<12} {s_mark:<16} {m_mark:<16} {pred_amt:>12.2f} {exp_amt:>12.2f} {err_pct:>7.1f}%")

    print("\n" + "-" * 80)
    print(f"Status accuracy:  {correct_status}/{total} ({100*correct_status/max(total,1):.0f}%)")
    print(f"Method accuracy:  {correct_method}/{total} ({100*correct_method/max(total,1):.0f}%)")
    print(f"Avg amount error: {total_amt_error/max(total,1):.1f}%")


if __name__ == "__main__":
    validate()
