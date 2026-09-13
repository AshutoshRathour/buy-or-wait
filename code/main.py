"""
main.py — Entry point for the Buy or Wait? financial decision agent.
Orchestrates data loading, financial analysis, and output generation.
"""

import os
import sys
import csv
import time
import argparse
import traceback
from datetime import datetime

import pandas as pd

# Add the code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_all, get_dataset_dir
from message_interpreter import get_user_message_effects, interpret_all_messages
from image_extractor import resolve_blank_amounts
from financial_engine import (
    build_exchange_rate_lookup,
    compute_amount_safe_to_pay,
    find_earliest_full_payment_date,
    get_salary_info,
    get_pending_and_scheduled_events,
    detect_recurring_events,
    identify_flexible_recurring,
    convert_currency,
)
from plan_evaluator import (
    evaluate_payment_options,
    generate_explanation,
)


def process_request(request, data, rate_lookup):
    """
    Process a single financial request and return the output row.
    """
    request_id = request["request_id"]
    user_id = request["user_id"]
    request_date = request["request_date"]
    requested_amount = request["requested_amount"]
    
    # Get user profile
    profiles = data["profiles"]
    user_profile = profiles[profiles["user_id"] == user_id]
    if user_profile.empty:
        return _default_output(request_id, requested_amount)
    profile = user_profile.iloc[0].to_dict()
    
    home_currency = profile["home_currency"]
    minimum_balance = profile["minimum_balance_to_keep"]
    starting_balance = profile["current_available_balance"]
    
    # Get message effects for this user
    user_messages = data["messages"][data["messages"]["user_id"] == user_id]
    msg_effects = get_user_message_effects(data["messages"], user_id, request_date)
    
    # Get financial events
    events = data["events"]
    
    # Detect recurring events
    recurring = detect_recurring_events(events, user_id, request_date)
    
    # Get pending and scheduled events
    pending_debits, scheduled_events = get_pending_and_scheduled_events(
        events, user_id, request_date
    )
    
    # Get salary info
    salary_info = get_salary_info(events, user_id, request_date)
    
    # Compute amount_safe_to_pay
    amount_safe = compute_amount_safe_to_pay(
        starting_balance, request_date, recurring, scheduled_events,
        pending_debits, salary_info, msg_effects, home_currency,
        rate_lookup, minimum_balance, requested_amount,
    )
    
    # Find earliest full payment date
    earliest_full = find_earliest_full_payment_date(
        starting_balance, request_date, recurring, scheduled_events,
        pending_debits, salary_info, msg_effects, home_currency,
        rate_lookup, minimum_balance, requested_amount,
    )
    
    # Identify flexible recurring expenses that could be changed
    flexible_changes = identify_flexible_recurring(
        recurring, profile, events, user_id
    )
    
    # Evaluate payment options and find best plan
    best_plan = evaluate_payment_options(
        request, profile, data["payment_options"],
        starting_balance, request_date, recurring,
        scheduled_events, pending_debits, salary_info,
        msg_effects, home_currency, rate_lookup,
        minimum_balance, amount_safe, earliest_full,
        flexible_changes,
    )
    
    # Generate explanation
    explanation = generate_explanation(
        request, profile, best_plan, amount_safe,
        earliest_full, minimum_balance,
    )
    
    # Format output
    status = best_plan.get("status", "not_affordable")
    method = best_plan.get("method", "not_recommended")
    plan_str = best_plan.get("plan_str", "none")
    spending_changes = best_plan.get("spending_changes", "none")
    
    # Format earliest_date
    earliest_date_str = ""
    if status == "affordable_now":
        earliest_date_str = request_date.strftime("%Y-%m-%d")
    elif earliest_full is not None:
        earliest_date_str = earliest_full.strftime("%Y-%m-%d")
    
    # Ensure amount_safe_to_pay is within bounds
    amount_safe = max(0, min(amount_safe, requested_amount))
    
    # Round amount to reasonable precision
    if home_currency in ("IDR",):
        amount_safe = round(amount_safe, 1)
    else:
        amount_safe = round(amount_safe, 2)
    
    return {
        "request_id": request_id,
        "amount_safe_to_pay": amount_safe,
        "affordability_status": status,
        "recommended_payment_method": method,
        "payment_plan": plan_str,
        "earliest_date_for_full_payment": earliest_date_str,
        "spending_changes_needed": spending_changes,
        "decision_explanation": explanation,
    }


def _default_output(request_id, requested_amount):
    """Return a safe default output for edge cases."""
    return {
        "request_id": request_id,
        "amount_safe_to_pay": 0,
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "payment_plan": "none",
        "earliest_date_for_full_payment": "",
        "spending_changes_needed": "none",
        "decision_explanation": "Unable to evaluate this request due to insufficient data.",
    }



def main():
    """Main entry point."""
    start_time = time.time()
    
    print("=" * 60)
    print("Buy or Wait? — Financial Decision Agent")
    print("=" * 60)
    
    # Load all data
    print("\n[1/5] Loading datasets...")
    dataset_dir = get_dataset_dir()
    
    # Override dataset_dir from command line if provided
    parser = argparse.ArgumentParser(description="Buy or Wait? Financial Decision Agent")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to the dataset/ directory")
    args, _ = parser.parse_known_args()
    if args.data_dir:
        dataset_dir = os.path.abspath(args.data_dir)
    
    data = load_all(dataset_dir)
    
    # Resolve blank amounts from images BEFORE processing requests
    print("  Resolving blank amounts from images...")
    data["events"] = resolve_blank_amounts(data["events"], data["images"])
    remaining_blanks = data["events"]["amount"].isna().sum()
    print(f"  Remaining events with blank amounts: {remaining_blanks}")
    
    requests = data["requests"]
    print(f"  Loaded {len(requests)} requests to evaluate")
    print(f"  Loaded {len(data['profiles'])} user profiles")
    print(f"  Loaded {len(data['events'])} financial events")
    print(f"  Loaded {len(data['messages'])} messages")
    print(f"  Loaded {len(data['images'])} images")
    print(f"  Loaded {len(data['payment_options'])} payment options")
    print(f"  Loaded {len(data['exchange_rates'])} exchange rates")
    
    # Build exchange rate lookup
    print("\n[2/5] Building exchange rate lookup...")
    rate_lookup = build_exchange_rate_lookup(data["exchange_rates"])
    
    # Interpret all messages
    print("\n[3/5] Interpreting messages...")
    message_interpretations = interpret_all_messages(data["messages"])
    interpretation_types = {}
    for interp in message_interpretations:
        t = interp["type"]
        interpretation_types[t] = interpretation_types.get(t, 0) + 1
    print(f"  Message types found: {dict(sorted(interpretation_types.items()))}")
    
    # Process each request
    print(f"\n[4/5] Processing {len(requests)} requests...")
    results = []
    errors = []
    
    for idx, (_, request) in enumerate(requests.iterrows()):
        request_id = request["request_id"]
        try:
            result = process_request(request, data, rate_lookup)
            results.append(result)
            
            if (idx + 1) % 25 == 0 or idx == 0:
                print(f"  [{idx+1}/{len(requests)}] {request_id}: "
                      f"{result['affordability_status']} / {result['recommended_payment_method']}")
        except Exception as e:
            print(f"  ERROR processing {request_id}: {e}")
            traceback.print_exc()
            errors.append(request_id)
            results.append(_default_output(request_id, request["requested_amount"]))
    
    # Write output CSV
    print(f"\n[5/5] Writing output.csv...")
    output_path = os.path.join(os.path.dirname(dataset_dir), "output.csv")
    
    columns = [
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    ]
    
    output_df = pd.DataFrame(results, columns=columns)
    output_df.to_csv(output_path, index=False)
    
    # Also write to dataset/ directory as the template expects
    dataset_output_path = os.path.join(dataset_dir, "output.csv")
    output_df.to_csv(dataset_output_path, index=False)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print(f"COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Output written to: {output_path}")
    print(f"  Total requests: {len(results)}")
    print(f"  Errors: {len(errors)}")
    if errors:
        print(f"  Error request IDs: {errors}")
    print(f"  Elapsed time: {elapsed:.1f}s")
    
    # Summary statistics
    statuses = {}
    methods = {}
    for r in results:
        s = r["affordability_status"]
        m = r["recommended_payment_method"]
        statuses[s] = statuses.get(s, 0) + 1
        methods[m] = methods.get(m, 0) + 1
    
    print(f"\n  Status distribution:")
    for s, count in sorted(statuses.items()):
        print(f"    {s}: {count}")
    print(f"\n  Method distribution:")
    for m, count in sorted(methods.items()):
        print(f"    {m}: {count}")
    
    # Write usage report
    _write_usage_report(elapsed, len(results))
    
    return output_path


def _write_usage_report(elapsed_seconds, num_requests):
    """Write the required evaluation/usage_report.md."""
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "evaluation", "usage_report.md"
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# Token Usage and Cost Report\n\n")
        f.write("## Final Full-Dataset Run\n\n")
        f.write(f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Total requests processed**: {num_requests}\n")
        f.write(f"- **Elapsed time**: {elapsed_seconds:.1f}s\n\n")
        f.write("## Model Usage\n\n")
        f.write("| Provider | Model | Calls | Input Tokens | Output Tokens | Total Tokens | Est. Cost |\n")
        f.write("|----------|-------|-------|-------------|---------------|--------------|----------|\n")
        f.write("| Local | Deterministic Engine | N/A | N/A | N/A | N/A | $0.00 |\n\n")
        f.write("## Notes\n\n")
        f.write("This solution uses a deterministic financial calculation engine.\n")
        f.write("Message interpretation uses rule-based pattern matching.\n")
        f.write("No external LLM API calls are made in this configuration.\n")
        f.write(f"Average processing time per request: {elapsed_seconds/max(num_requests,1):.2f}s\n")


if __name__ == "__main__":
    main()
