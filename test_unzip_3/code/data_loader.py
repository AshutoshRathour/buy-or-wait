"""
data_loader.py — Load and parse all dataset CSV files for the Buy or Wait? challenge.
"""

import os
import pandas as pd
from datetime import datetime


def get_dataset_dir():
    """Return the absolute path to the dataset/ directory."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")


def load_requests(dataset_dir=None):
    """Load dataset/requests.csv."""
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    df = pd.read_csv(os.path.join(dataset_dir, "requests.csv"))
    df["request_date"] = pd.to_datetime(df["request_date"])
    df["desired_completion_date"] = pd.to_datetime(df["desired_completion_date"])
    df["requested_amount"] = pd.to_numeric(df["requested_amount"], errors="coerce")
    df["allows_partial_payment"] = df["allows_partial_payment"].astype(str).str.lower() == "true"
    return df


def load_sample_requests(dataset_dir=None):
    """Load dataset/sample_requests.csv with ground-truth output columns."""
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    df = pd.read_csv(os.path.join(dataset_dir, "sample_requests.csv"))
    df["request_date"] = pd.to_datetime(df["request_date"])
    df["desired_completion_date"] = pd.to_datetime(df["desired_completion_date"])
    df["requested_amount"] = pd.to_numeric(df["requested_amount"], errors="coerce")
    df["allows_partial_payment"] = df["allows_partial_payment"].astype(str).str.lower() == "true"
    df["amount_safe_to_pay"] = pd.to_numeric(df["amount_safe_to_pay"], errors="coerce")
    return df


def load_financial_profiles(dataset_dir=None):
    """Load dataset/financial_profiles.csv."""
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    df = pd.read_csv(os.path.join(dataset_dir, "financial_profiles.csv"))
    df["current_available_balance"] = pd.to_numeric(df["current_available_balance"], errors="coerce")
    df["minimum_balance_to_keep"] = pd.to_numeric(df["minimum_balance_to_keep"], errors="coerce")
    df["max_installment_months"] = pd.to_numeric(df["max_installment_months"], errors="coerce")
    # Parse pipe-delimited fields into lists
    for col in [
        "financial_priorities",
        "expense_categories_to_protect",
        "expense_categories_user_is_willing_to_reduce",
        "expense_categories_user_is_willing_to_stop",
        "payment_methods_user_will_consider",
    ]:
        df[col] = df[col].fillna("").astype(str).apply(lambda x: [s.strip() for s in x.split("|") if s.strip()])
    return df


def load_financial_events(dataset_dir=None):
    """Load dataset/financial_events.csv."""
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    df = pd.read_csv(os.path.join(dataset_dir, "financial_events.csv"))
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["minimum_allowed_amount"] = pd.to_numeric(df["minimum_allowed_amount"], errors="coerce")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["settlement_date"] = pd.to_datetime(df["settlement_date"], errors="coerce")
    return df


def load_exchange_rates(dataset_dir=None):
    """Load dataset/exchange_rates.csv."""
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    df = pd.read_csv(os.path.join(dataset_dir, "exchange_rates.csv"))
    df["rate_date"] = pd.to_datetime(df["rate_date"])
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    return df


def load_payment_options(dataset_dir=None):
    """Load dataset/request_payment_options.csv."""
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    df = pd.read_csv(os.path.join(dataset_dir, "request_payment_options.csv"))
    df["payment_amount"] = pd.to_numeric(df["payment_amount"], errors="coerce")
    df["number_of_payments"] = pd.to_numeric(df["number_of_payments"], errors="coerce")
    df["first_payment_date"] = pd.to_datetime(df["first_payment_date"], errors="coerce")
    df["payment_frequency_days"] = pd.to_numeric(df["payment_frequency_days"], errors="coerce")
    df["financing_fee"] = pd.to_numeric(df["financing_fee"], errors="coerce")
    df["total_payable_amount"] = pd.to_numeric(df["total_payable_amount"], errors="coerce")
    return df


def load_messages(dataset_dir=None):
    """Load dataset/messages.csv."""
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    df = pd.read_csv(os.path.join(dataset_dir, "messages.csv"))
    df["sent_at"] = pd.to_datetime(df["sent_at"], errors="coerce")
    return df


def load_images(dataset_dir=None):
    """Load dataset/images.csv."""
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    df = pd.read_csv(os.path.join(dataset_dir, "images.csv"))
    return df


def load_all(dataset_dir=None):
    """Load all datasets and return as a dictionary."""
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    return {
        "requests": load_requests(dataset_dir),
        "sample_requests": load_sample_requests(dataset_dir),
        "profiles": load_financial_profiles(dataset_dir),
        "events": load_financial_events(dataset_dir),
        "exchange_rates": load_exchange_rates(dataset_dir),
        "payment_options": load_payment_options(dataset_dir),
        "messages": load_messages(dataset_dir),
        "images": load_images(dataset_dir),
    }
