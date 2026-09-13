"""
evaluation/main.py - Evaluate the output.csv against sample_requests.csv ground truth.
Reports accuracy metrics for each output field.
"""

import os
import sys
import csv
import math

import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate():
    """Compare output.csv against sample_requests.csv and report metrics."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_dir = os.path.join(project_root, "dataset")
    
    # Load sample requests (ground truth)
    sample_path = os.path.join(dataset_dir, "sample_requests.csv")
    sample_df = pd.read_csv(sample_path)
    
    # Load generated output
    output_path = os.path.join(project_root, "output.csv")
    if not os.path.exists(output_path):
        print("ERROR: output.csv not found at", output_path)
        print("Run 'python code/main.py' first to generate the output.")
        return
    
    output_df = pd.read_csv(output_path)
    
    print("=" * 70)
    print("Evaluation Report")
    print("=" * 70)
    
    # Basic checks
    print("\n" + "-" * 40)
    print("FORMAT VALIDATION")
    print("-" * 40)
    
    expected_cols = [
        "request_id", "amount_safe_to_pay", "affordability_status",
        "recommended_payment_method", "payment_plan",
        "earliest_date_for_full_payment", "spending_changes_needed",
        "decision_explanation"
    ]
    
    missing_cols = [c for c in expected_cols if c not in output_df.columns]
    if missing_cols:
        print(f"  MISSING COLUMNS: {missing_cols}")
    else:
        print(f"  [OK] All {len(expected_cols)} required columns present")
    
    # Check row count
    requests_path = os.path.join(dataset_dir, "requests.csv")
    requests_df = pd.read_csv(requests_path)
    expected_rows = len(requests_df)
    actual_rows = len(output_df)
    if actual_rows == expected_rows:
        print(f"  [OK] Row count matches: {actual_rows}")
    else:
        print(f"  [FAIL] Row count mismatch: expected {expected_rows}, got {actual_rows}")
    
    # Check amount_safe_to_pay bounds
    merged = requests_df.merge(output_df, on="request_id", how="left")
    violations = merged[
        (merged["amount_safe_to_pay"] < 0) |
        (merged["amount_safe_to_pay"] > merged["requested_amount"])
    ]
    if len(violations) == 0:
        print(f"  [OK] All amount_safe_to_pay values within bounds")
    else:
        print(f"  [FAIL] {len(violations)} bound violations in amount_safe_to_pay")
    
    # Check valid affordability_status
    valid_statuses = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
    invalid_status = output_df[~output_df["affordability_status"].isin(valid_statuses)]
    if len(invalid_status) == 0:
        print(f"  [OK] All affordability_status values are valid")
    else:
        print(f"  [FAIL] {len(invalid_status)} invalid affordability_status values")
    
    # Check valid recommended_payment_method
    valid_methods = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
    invalid_method = output_df[~output_df["recommended_payment_method"].isin(valid_methods)]
    if len(invalid_method) == 0:
        print(f"  [OK] All recommended_payment_method values are valid")
    else:
        print(f"  [FAIL] {len(invalid_method)} invalid recommended_payment_method values")
    
    # Compare with sample requests
    print("\n" + "-" * 40)
    print("SAMPLE REQUEST ACCURACY")
    print("-" * 40)
    
    sample_ids = set(sample_df["request_id"].tolist())
    output_sample = output_df[output_df["request_id"].isin(sample_ids)]
    
    if output_sample.empty:
        print("  No sample requests found in output. Skipping accuracy check.")
        print("  (Sample requests are request_01 through request_25)")
        return
    
    # Merge for comparison
    comparison = sample_df.merge(output_sample, on="request_id", suffixes=("_expected", "_predicted"))
    
    # Accuracy metrics
    metrics = {
        "affordability_status": {"correct": 0, "total": 0},
        "recommended_payment_method": {"correct": 0, "total": 0},
        "amount_safe_to_pay": {"total_error": 0, "total": 0, "within_5pct": 0},
        "earliest_date_for_full_payment": {"correct": 0, "total": 0},
        "spending_changes_needed": {"correct": 0, "total": 0},
    }
    
    print(f"\n  {'Request':<12} {'Status':<10} {'Method':<10} {'Amt Safe':>14} {'Amt Expected':>14} {'Error%':>8}")
    print("  " + "-" * 70)
    
    for _, row in comparison.iterrows():
        req_id = row["request_id"]
        
        # Affordability status
        status_ok = row["affordability_status_expected"] == row["affordability_status_predicted"]
        metrics["affordability_status"]["correct"] += int(status_ok)
        metrics["affordability_status"]["total"] += 1
        
        # Payment method
        method_ok = row["recommended_payment_method_expected"] == row["recommended_payment_method_predicted"]
        metrics["recommended_payment_method"]["correct"] += int(method_ok)
        metrics["recommended_payment_method"]["total"] += 1
        
        # Amount safe to pay
        expected_amt = row.get("amount_safe_to_pay_expected", 0)
        predicted_amt = row.get("amount_safe_to_pay_predicted", 0)
        if pd.notna(expected_amt) and pd.notna(predicted_amt) and expected_amt > 0:
            error_pct = abs(predicted_amt - expected_amt) / expected_amt * 100
            metrics["amount_safe_to_pay"]["total_error"] += error_pct
            metrics["amount_safe_to_pay"]["total"] += 1
            if error_pct <= 5:
                metrics["amount_safe_to_pay"]["within_5pct"] += 1
        elif pd.notna(expected_amt) and pd.notna(predicted_amt):
            metrics["amount_safe_to_pay"]["total"] += 1
            error_pct = abs(predicted_amt - expected_amt) if expected_amt == 0 else 0
            if predicted_amt == expected_amt:
                metrics["amount_safe_to_pay"]["within_5pct"] += 1
        else:
            error_pct = float('nan')
        
        # Earliest full payment date
        efp_exp = str(row.get("earliest_date_for_full_payment_expected", ""))
        efp_pred = str(row.get("earliest_date_for_full_payment_predicted", ""))
        efp_ok = efp_exp.strip() == efp_pred.strip()
        metrics["earliest_date_for_full_payment"]["correct"] += int(efp_ok)
        metrics["earliest_date_for_full_payment"]["total"] += 1
        
        # Spending changes
        sc_exp = str(row.get("spending_changes_needed_expected", "none"))
        sc_pred = str(row.get("spending_changes_needed_predicted", "none"))
        sc_ok = sc_exp.strip() == sc_pred.strip()
        metrics["spending_changes_needed"]["correct"] += int(sc_ok)
        metrics["spending_changes_needed"]["total"] += 1
        
        status_mark = "OK" if status_ok else "X"
        method_mark = "OK" if method_ok else "X"
        
        print(f"  {req_id:<12} {status_mark:<10} {method_mark:<10} "
              f"{predicted_amt:>14.2f} {expected_amt:>14.2f} {error_pct:>7.1f}%")
    
    # Summary
    print("\n" + "-" * 40)
    print("ACCURACY SUMMARY")
    print("-" * 40)
    
    for field, m in metrics.items():
        if field == "amount_safe_to_pay":
            total = m["total"]
            if total > 0:
                avg_error = m["total_error"] / total
                within_5 = m["within_5pct"]
                print(f"  {field}: avg error {avg_error:.1f}%, "
                      f"within 5%: {within_5}/{total} ({100*within_5/total:.0f}%)")
        else:
            total = m["total"]
            correct = m["correct"]
            if total > 0:
                print(f"  {field}: {correct}/{total} ({100*correct/total:.0f}%)")


if __name__ == "__main__":
    evaluate()
