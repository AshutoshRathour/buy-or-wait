"""Temporary analysis script."""
import pandas as pd

events = pd.read_csv('dataset/financial_events.csv')

# Check linked events
linked = events[events['linked_event_id'].notna()]
print(f"Total linked events: {len(linked)}")
print("\nLinked events (first 30):")
for _, e in linked.head(30).iterrows():
    parent = events[events['event_id'] == e['linked_event_id']]
    parent_desc = parent.iloc[0]['description'] if not parent.empty else 'N/A'
    print(f"  {e['event_id']} ({e['status']}) -> {e['linked_event_id']}: {e['description']} | Parent: {parent_desc}")

# Check unrealized events
unrealized = events[events['status'] == 'unrealized']
print(f"\nUnrealized events ({len(unrealized)}):")
print(unrealized[['event_id','user_id','event_type','category','direction','amount','description']].to_string())
