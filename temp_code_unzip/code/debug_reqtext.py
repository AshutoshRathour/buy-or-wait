import pandas as pd
requests = pd.read_csv('dataset/sample_requests.csv')
for _, r in requests.iterrows():
    if r['request_id'] in ['request_05', 'request_10', 'request_13', 'request_20', 'request_25']:
        print(f"{r['request_id']}: {r['request_text']}")
