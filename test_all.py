import sys
import pandas as pd
import numpy as np
sys.path.insert(0, 'code')
from data_loader import load_all, get_dataset_dir
from main import process_request
from financial_engine import build_exchange_rate_lookup

data = load_all(get_dataset_dir())
rate_lookup = build_exchange_rate_lookup(data['exchange_rates'])
sample_reqs = data['sample_requests'].copy()
# Ensure request_date is datetime
sample_reqs['request_date'] = pd.to_datetime(sample_reqs['request_date'])

matches = 0
total = len(sample_reqs)

for idx, req in sample_reqs.iterrows():
    result = process_request(req, data, rate_lookup)
    expected = req['amount_safe_to_pay']
    pred = result['amount_safe_to_pay']
    
    if abs(expected - pred) > 0.01:
        print(f"FAILED {req['request_id']}: Expected: {expected} Predicted: {pred}")
    else:
        matches += 1

print(f"Matches: {matches} / {total}")
