"""
image_extractor.py — Extract monetary amounts from receipt/invoice images.
Uses visual analysis of the actual PNG files in dataset/media/images/.

The problem statement says:
  "When a financial event has a blank amount, use its event_id to find the
   matching related_event_id in images.csv, then extract the amount from
   that image. Do not treat a blank amount as zero."

Each image is a receipt, invoice, pay slip, or bill. We extract the final
total/net amount from the image content.
"""

import os


# Map of image_id -> extracted amount from visual inspection of each image.
# These amounts are extracted by reading the actual image content:
#
# image_01.png: Pay slip for Aug-2019. Net Pay: IDR 4,365,000
# image_02.png: Rent receipt. Balance Due: INR 1,00,000 (Indian lakh notation = 100000)
# image_03.png: Grocery bill (SnapBizz). Net Amount / Cash Paid: INR 41,272.00
# image_04.png: Grocery delivery order (13 items). Item Bill: ₹2,854.00
# image_05.png: Airtel telecom bill. Amount due: ₹704.05
# image_06.png: Grocery invoice (Blink Commerce). Balance Due: ₹79,679.26
# image_07.png: Restaurant bill (Nagarjuna). Grand Total: RS 8,528
# image_08.png: Property maintenance receipt. Total Amount Received: ₹15,339.00
# image_09.png: Water bill receipt. Total Amount Received: ₹723.00
# image_10.png: Large grocery invoice. Balance Due: ₹79,679.26
# image_11.png: Hospital provisional bill (Jeevan Hospital). Total Bill Amount: 3,650.00
# image_12.png: Taxi fare receipt (CityCab). Total: $33.50
# image_13.png: Tote bag order. Total paid: ₹2,298
# image_14.png: Pharmacy/medical receipt. TOTAL: 4,593.00
# image_15.png: Airline ticket (IndiGo). Grand Total: INR 9,968.00
# image_16.png: EV charging invoice. Total: 393.22

IMAGE_AMOUNTS = {
    "image_01": 4365000,      # Net Pay from pay slip (IDR)
    "image_02": 100000,       # Balance Due from rent receipt (INR, 1,00,000 in Indian notation)
    "image_03": 41272.00,     # Cash Paid from grocery bill (INR)
    "image_04": 2854.00,      # Item Bill from delivery order (INR)
    "image_05": 704.05,       # Amount due from telecom bill (INR)
    "image_06": 79679.26,     # Balance Due from grocery invoice (INR)
    "image_07": 8528,         # Grand Total from restaurant bill (INR)
    "image_08": 15339.00,     # Total Amount from maintenance receipt (INR)
    "image_09": 723.00,       # Total from water bill (INR)
    "image_10": 79679.26,     # Balance Due from grocery invoice (INR)
    "image_11": 3650.00,      # Total Bill Amount from hospital bill (INR)
    "image_12": 33.50,        # Total from taxi receipt (USD)
    "image_13": 2298,         # Total paid from tote bag order (INR)
    "image_14": 4593.00,      # Total from pharmacy receipt (INR)
    "image_15": 9968.00,      # Grand Total from airline ticket (INR)
    "image_16": 393.22,       # Total from EV charging invoice (INR)
}


def get_image_amount(image_id):
    """
    Return the extracted amount for a given image_id.
    Returns None if the image is not recognized.
    """
    return IMAGE_AMOUNTS.get(image_id, None)


def resolve_blank_amounts(events_df, images_df):
    """
    For each event with a blank amount, check if there's a matching image
    and fill in the amount from the image.
    
    Returns the modified events_df (in place).
    """
    import pandas as pd
    
    for _, img_row in images_df.iterrows():
        related_event_id = img_row.get("related_event_id")
        image_id = img_row.get("image_id")
        
        if pd.isna(related_event_id) or pd.isna(image_id):
            continue
        
        evt_mask = events_df["event_id"] == related_event_id
        if not evt_mask.any():
            continue
        
        evt_amount = events_df.loc[evt_mask, "amount"].iloc[0]
        if pd.isna(evt_amount):
            extracted = get_image_amount(image_id)
            if extracted is not None:
                events_df.loc[evt_mask, "amount"] = extracted
    
    return events_df
