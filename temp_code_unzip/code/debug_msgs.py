import pandas as pd

events = pd.read_csv('dataset/financial_events.csv')
profiles = pd.read_csv('dataset/financial_profiles.csv')
msgs = pd.read_csv('dataset/messages.csv')
imgs = pd.read_csv('dataset/images.csv')

user_id = "user_05"
profile = profiles[profiles['user_id'] == user_id].iloc[0]
print(f"Balance: {profile['home_currency']} {profile['current_available_balance']:,.2f}")
print(f"Min balance: {profile['minimum_balance_to_keep']:,.2f}")
print(f"Max installments: {profile['max_installment_months']}")
print(f"Protected: {profile['expense_categories_to_protect']}")
print(f"Reduce: {profile['expense_categories_user_is_willing_to_reduce']}")
print(f"Stop: {profile['expense_categories_user_is_willing_to_stop']}")
print(f"Methods: {profile['payment_methods_user_will_consider']}")
print(f"Priorities: {profile['financial_priorities']}")

user_events = events[events['user_id'] == user_id].sort_values('settlement_date')

# Count events by status
print(f"\nEvent status counts:")
print(user_events['status'].value_counts().to_string())

# Count events by category
print(f"\nEvent categories:")
print(user_events['category'].value_counts().to_string())

# Show ALL events - check for linked events, non-cash, etc.
print(f"\nAll user_05 events ({len(user_events)}):")
for _, e in user_events.iterrows():
    amt_str = f"{e['amount']:,.2f}" if pd.notna(e['amount']) else "NaN"
    linked = e.get('linked_event_id', '')
    linked_str = f" linked={linked}" if pd.notna(linked) and linked else ""
    print(f"  {e['event_id']:15s} {e['status']:10s} {e['direction']:6s} "
          f"{e['category']:25s} {amt_str:>15s} settle={str(e['settlement_date'])[:10]}{linked_str}")

# Check images
u_imgs = imgs[imgs['user_id'] == user_id]
print(f"\nImages: {len(u_imgs)}")
for _, i in u_imgs.iterrows():
    print(f"  {i['image_id']}: related={i.get('related_event_id','')} desc={i.get('image_description','')}")

# Check request_payment_options
options = pd.read_csv('dataset/request_payment_options.csv')
req_opts = options[options['request_id'] == 'request_05']
print(f"\nPayment options for request_05:")
for _, o in req_opts.iterrows():
    print(f"  {o['payment_option_id']}: method={o['payment_method']} amount={o['payment_amount']} "
          f"n={o['number_of_payments']} first={o['first_payment_date']} freq={o['payment_frequency_days']} "
          f"fee={o['financing_fee']} total={o['total_payable_amount']}")
