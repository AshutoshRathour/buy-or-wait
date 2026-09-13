"""
message_interpreter.py — Rule-based interpretation of messages and images.
Extracts salary changes, payment confirmations, rent adjustments, and other 
financial information from the messages.csv and images.csv datasets.
"""

import re
import os
from datetime import datetime
from PIL import Image
import pandas as pd


# ──────────────────────────────────────────────────────────────
# Image amount extraction (regex on OCR-like visual text is not possible
# without an LLM, but we can try to map known images to amounts from
# the sample outputs or extract from context)
# ──────────────────────────────────────────────────────────────

def extract_amount_from_image(image_path):
    """
    Attempt to extract a monetary amount from an image file.
    This uses basic image analysis — if Google Gemini is available, 
    it would be used here for better accuracy.
    
    Returns the extracted amount as a float, or None if extraction fails.
    """
    # We cannot do reliable OCR without an LLM/OCR library.
    # We'll mark these as needing LLM processing and handle them in the pipeline.
    return None


# ──────────────────────────────────────────────────────────────
# Message interpretation — extract structured financial updates
# ──────────────────────────────────────────────────────────────

def _extract_amount(text, currency_hint=None):
    """Extract monetary amount from message text."""
    # Handle various amount formats
    patterns = [
        # "IDR 42750000" or "EUR 1037.52" or "INR 148000"
        r'(?:IDR|EUR|USD|INR|ZAR)\s*([\d,]+(?:\.\d+)?)',
        # "salary is now INR 148000" patterns
        r'(?:salary|pay|amount|payment|credit|payout|invoice|salary credit|salary of|salary is|monthly salary has increased to|salary will be|monthly pay is|base salary is|confirmed monthly salary is|remaining confirmed monthly salary is|salary for the next payroll is|salary resumes on|salary has increased to|salary credit for|salary of)\s+(?:of\s+)?(?:IDR|EUR|USD|INR|ZAR)\s*([\d,]+(?:\.\d+)?)',
        # amounts with commas: "42,750,000" or "1,037.52"
        r'(?:IDR|EUR|USD|INR|ZAR)\s*([\d]{1,3}(?:,\d{3})*(?:\.\d+)?)',
    ]
    
    amounts = []
    for pat in patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            val = match.group(1) if match.lastindex >= 1 else match.group(0)
            val = val.replace(",", "")
            try:
                amounts.append(float(val))
            except ValueError:
                continue
    return amounts


def _extract_date(text):
    """Extract dates in YYYY-MM-DD format from text."""
    dates = []
    for match in re.finditer(r'(\d{4}-\d{2}-\d{2})', text):
        try:
            dates.append(datetime.strptime(match.group(1), "%Y-%m-%d"))
        except ValueError:
            continue
    return dates


def _extract_currency(text):
    """Extract currency code from text."""
    for match in re.finditer(r'\b(IDR|EUR|USD|INR|ZAR)\b', text):
        return match.group(1)
    return None


def interpret_message(message_row):
    """
    Interpret a single message and return structured financial information.
    
    Returns a dict with:
        - type: 'salary_change', 'salary_delayed', 'salary_first', 'salary_ended',
                'salary_temporary_reduction', 'salary_arrears', 'salary_resumed',
                'invoice_confirmed', 'rent_increase', 'investment_unrealized',
                'investment_settled', 'prize_pending', 'prize_settled', 'refund_pending',
                'bank_transfer', 'scam', 'employment_ended', 'childcare_deduction',
                'foreign_salary', 'bill_retry', 'card_dispute', 'commission_pending',
                'household_income_ended', 'reimbursement', 'payout_pending', 
                'unknown'
        - amount: float or None
        - currency: str or None
        - effective_date: datetime or None
        - details: dict with additional info
    """
    text = str(message_row.get("message_text", ""))
    user_id = message_row.get("user_id", "")
    related_event_id = message_row.get("related_event_id", "")
    
    result = {
        "type": "unknown",
        "amount": None,
        "currency": _extract_currency(text),
        "effective_date": None,
        "salary_date": None,
        "details": {},
        "message_id": message_row.get("message_id", ""),
        "user_id": user_id,
        "request_id": message_row.get("request_id", ""),
        "related_event_id": related_event_id,
    }
    
    text_lower = text.lower()
    
    # ── Scam detection ──
    if any(phrase in text_lower for phrase in [
        "pay the release charge",
        "pay the processing charge",
        "congratulations! you've been selected for a cash prize",
        "selamat! anda terpilih untuk menerima hadiah uang tunai",
        "bayar biaya pencairan",
        "bayar biaya pemrosesan",
    ]):
        result["type"] = "scam"
        return result
    
    # ── Employment ended ──
    if any(phrase in text_lower for phrase in [
        "employment has ended",
        "hubungan kerja anda telah berakhir",
        "no regular salary payments scheduled",
        "tidak ada pembayaran gaji rutin",
    ]):
        result["type"] = "employment_ended"
        return result
    
    # ── Seasonal contract ended / no income ──
    if any(phrase in text_lower for phrase in [
        "seasonal contract has ended",
        "kontrak musiman saat ini telah berakhir",
        "no off-season income",
    ]):
        result["type"] = "seasonal_contract_ended"
        return result
    
    # ── Household income ended ──
    if any(phrase in text_lower for phrase in [
        "household employment record has ended",
        "salah satu sumber pendapatan kerja rumah tangga telah berakhir",
    ]):
        result["type"] = "household_income_ended"
        amounts = _extract_amount(text)
        if amounts:
            result["amount"] = amounts[0]
        return result
    
    # ── Salary increase ──
    if any(phrase in text_lower for phrase in [
        "salary has increased to",
        "monthly salary has increased to",
        "gaji bulanan anda naik menjadi",
        "gaji bulanan anda naik",
    ]):
        result["type"] = "salary_increase"
        amounts = _extract_amount(text)
        dates = _extract_date(text)
        if amounts:
            result["amount"] = amounts[0]
        if dates:
            result["effective_date"] = dates[0]
        return result
    
    # ── Salary resumed with childcare deduction ──
    if any(phrase in text_lower for phrase in [
        "regular salary of",
        "regular salary resumes",
        "gaji rutin sebesar",
    ]) and any(phrase in text_lower for phrase in [
        "childcare payment begins",
        "childcare",
    ]):
        result["type"] = "salary_resumed_with_childcare"
        amounts = _extract_amount(text)
        dates = _extract_date(text)
        if amounts:
            result["amount"] = amounts[0]
        if dates:
            result["effective_date"] = dates[0]
        return result
    
    # ── Salary delayed ──
    if any(phrase in text_lower for phrase in [
        "salary is now expected on",
        "gaji anda dijadwalkan ulang",
        "this replaces the payroll date",
    ]):
        result["type"] = "salary_delayed"
        dates = _extract_date(text)
        if dates:
            result["effective_date"] = dates[0]
        return result
    
    # ── First salary from new employer ──
    if any(phrase in text_lower for phrase in [
        "first salary will be",
        "first salary from the new employer",
        "first salary of",
        "gaji pertama",
        "gaji pertama anda",
        "your first salary",
    ]):
        result["type"] = "first_salary"
        amounts = _extract_amount(text)
        dates = _extract_date(text)
        if amounts:
            result["amount"] = amounts[0]
        if dates:
            result["effective_date"] = dates[0]
        return result
    
    # ── Temporary salary reduction ──
    if any(phrase in text_lower for phrase in [
        "temporary monthly pay is",
        "gaji bulanan sementara",
        "reduced amount continues",
        "jumlah yang lebih rendah",
    ]):
        result["type"] = "salary_temporary_reduction"
        amounts = _extract_amount(text)
        if amounts:
            result["amount"] = amounts[0]
        return result
    
    # ── Salary reduction (unpaid leave) ──
    if any(phrase in text_lower for phrase in [
        "next salary is reduced to",
        "salary is reduced to",
        "adjustment is due to approved unpaid leave",
    ]):
        result["type"] = "salary_reduced"
        amounts = _extract_amount(text)
        if amounts:
            result["amount"] = amounts[0]
        return result
    
    # ── Regular salary with arrears ──
    if any(phrase in text_lower for phrase in [
        "one-time arrears adjustment",
        "penyesuaian tunggakan satu kali",
    ]):
        result["type"] = "salary_with_arrears"
        amounts = _extract_amount(text)
        if len(amounts) >= 2:
            result["amount"] = amounts[0]  # regular salary
            result["details"]["arrears"] = amounts[1]
        elif len(amounts) == 1:
            result["amount"] = amounts[0]
        return result
    
    # ── Foreign currency salary ──
    if any(phrase in text_lower for phrase in [
        "salary of",
        "gaji sebesar",
    ]) and any(phrase in text_lower for phrase in [
        "receiving bank will convert",
        "bank penerima akan mengonversi",
        "settlement-date rate",
        "kurs pada tanggal penyelesaian",
    ]):
        result["type"] = "foreign_salary"
        amounts = _extract_amount(text)
        dates = _extract_date(text)
        if amounts:
            result["amount"] = amounts[0]
        if dates:
            result["effective_date"] = dates[0]
        return result
    
    # ── Confirmed base salary (with pending commission) ──
    if any(phrase in text_lower for phrase in [
        "confirmed base salary is",
        "confirmed salary is",
        "gaji pokok yang dikonfirmasi",
    ]) and any(phrase in text_lower for phrase in [
        "commission",
        "komisi",
        "pending approval",
    ]):
        result["type"] = "salary_base_confirmed"
        amounts = _extract_amount(text)
        if amounts:
            result["amount"] = amounts[0]
        return result
    
    # ── Confirmed salary (simple, first employer credit on date) ──
    if any(phrase in text_lower for phrase in [
        "confirmed salary",
        "gaji yang dikonfirmasi",
        "confirmed for",
        "dikonfirmasi untuk",
    ]):
        result["type"] = "salary_confirmed"
        amounts = _extract_amount(text)
        dates = _extract_date(text)
        if amounts:
            result["amount"] = amounts[0]
        if dates:
            result["effective_date"] = dates[0]
        return result
    
    # ── Invoice payment confirmed ──
    if any(phrase in text_lower for phrase in [
        "client approved an invoice payment",
        "klien menyetujui pembayaran faktur",
        "invoice payment of",
    ]):
        result["type"] = "invoice_confirmed"
        amounts = _extract_amount(text)
        dates = _extract_date(text)
        if amounts:
            result["amount"] = amounts[0]
        if dates:
            result["effective_date"] = dates[0]
        return result
    
    # ── Rent increase ──
    if any(phrase in text_lower for phrase in [
        "lease increases monthly rent by",
        "renewed lease increases",
        "sewa baru menaikkan biaya sewa bulanan",
    ]):
        result["type"] = "rent_increase"
        pct_match = re.search(r'(\d+)%', text)
        if pct_match:
            result["details"]["increase_pct"] = int(pct_match.group(1))
        return result
    
    # ── Investment unrealized gain ──
    if any(phrase in text_lower for phrase in [
        "market value has increased",
        "no units have been sold",
        "no cash proceeds",
        "nilai pasar portofolio",
    ]):
        result["type"] = "investment_unrealized"
        return result
    
    # ── Investment sale settled ──
    if any(phrase in text_lower for phrase in [
        "investment sale have settled",
        "penjualan investasi anda sudah masuk",
        "sale order is complete",
        "proceeds from your investment sale",
        "hasil penjualan investasi",
    ]):
        result["type"] = "investment_settled"
        return result
    
    # ── Prize settled (proceeds reached account) ──
    if any(phrase in text_lower for phrase in [
        "prize proceeds have reached your account",
        "hasil hadiah sudah masuk ke rekening",
    ]):
        result["type"] = "prize_settled"
        return result
    
    # ── Prize pending ──
    if any(phrase in text_lower for phrase in [
        "prize claim has been verified",
        "klaim hadiah anda sudah diverifikasi",
        "still in payment processing",
    ]):
        result["type"] = "prize_pending"
        return result
    
    # ── Refund pending ──
    if any(phrase in text_lower for phrase in [
        "refund has been initiated",
        "refund is still processing",
        "pengembalian dana masih diproses",
    ]):
        result["type"] = "refund_pending"
        return result
    
    # ── Payout pending (gig/freelance) ──
    if any(phrase in text_lower for phrase in [
        "payout is still pending",
        "pembayaran berikutnya",
        "earnings shown in the",
        "balance isn't withdrawable",
        "saldo belum dapat ditarik",
    ]):
        result["type"] = "payout_pending"
        return result
    
    # ── Bank internal transfer ──
    if any(phrase in text_lower for phrase in [
        "transfer between your two accounts",
        "same account holder",
    ]):
        result["type"] = "internal_transfer"
        return result
    
    # ── Reimbursement ──
    if any(phrase in text_lower for phrase in [
        "reimbursement for your earlier work expense",
    ]):
        result["type"] = "reimbursement"
        return result
    
    # ── Bill retry (failed debit, bill still outstanding) ──
    if any(phrase in text_lower for phrase in [
        "previous debit attempt failed",
        "bill is still outstanding",
    ]):
        result["type"] = "bill_retry"
        return result
    
    # ── Card dispute (extra charge under investigation) ──
    if any(phrase in text_lower for phrase in [
        "extra card charge is still being investigated",
        "reversal has not been posted",
    ]):
        result["type"] = "card_dispute"
        return result
    
    # ── Foreign currency bill ──
    if any(phrase in text_lower for phrase in [
        "bill was charged in a foreign currency",
        "tagihan dikenakan dalam mata uang asing",
    ]):
        result["type"] = "foreign_bill"
        return result
    
    # ── Bonus pending (quarterly bonus not approved) ──
    if any(phrase in text_lower for phrase in [
        "quarterly bonus is still subject to",
        "bonus kuartalan anda masih menunggu",
    ]):
        result["type"] = "bonus_pending"
        return result
    
    # ── Card minimum payments ──
    if any(phrase in text_lower for phrase in [
        "minimum payments due on",
        "payment to one card will not cover",
    ]):
        result["type"] = "card_minimum_payments"
        return result
    
    # ── Payment confirmed with receipt ──
    if any(phrase in text_lower for phrase in [
        "payment was received",
        "order was paid",
        "pembayaran diterima",
    ]):
        result["type"] = "payment_confirmed"
        return result
    
    # ── Payroll confirmed for specific salary ──
    # General catch for salary-related messages with amounts
    if any(phrase in text_lower for phrase in [
        "payroll", "salary", "gaji", "penggajian",
    ]):
        amounts = _extract_amount(text)
        dates = _extract_date(text)
        if amounts:
            result["type"] = "salary_info"
            result["amount"] = amounts[0]
        if dates:
            result["effective_date"] = dates[0]
        return result

    # ── MoneyHub / wallet charge + salary combined message ──
    if "wallet was charged" in text_lower or "moneyhub" in text_lower:
        result["type"] = "wallet_charge"
        amounts = _extract_amount(text)
        dates = _extract_date(text)
        if amounts:
            result["amount"] = amounts[0]
        if dates:
            result["effective_date"] = dates[0]
        return result
    
    return result


def interpret_all_messages(messages_df):
    """
    Interpret all messages and return a list of structured results.
    """
    results = []
    for _, row in messages_df.iterrows():
        result = interpret_message(row)
        results.append(result)
    return results


def get_user_message_effects(messages_df, user_id, request_date):
    """
    Get all message-derived financial effects for a user up to request_date.
    Returns a dict with structured information to adjust financial projections.
    """
    user_msgs = messages_df[messages_df["user_id"] == user_id].copy()
    
    effects = {
        "salary_override": None,  # If salary amount is changed
        "salary_date_override": None,  # If salary date is changed
        "salary_ended": False,  # If employment ended
        "salary_arrears": None,  # One-time arrears amount
        "confirmed_income": [],  # List of (amount, date, currency) for confirmed income
        "rent_increase_pct": None,  # Percentage rent increase
        "ignore_events": [],  # Event IDs to ignore (internal transfers, etc.)
        "pending_not_cash": [],  # Event IDs that are not spendable cash
        "childcare_deduction": False,  # New childcare deduction
    }
    
    for _, row in user_msgs.iterrows():
        interpretation = interpret_message(row)
        msg_type = interpretation["type"]
        
        if msg_type == "employment_ended":
            effects["salary_ended"] = True
        elif msg_type == "seasonal_contract_ended":
            effects["salary_ended"] = True
        elif msg_type == "salary_increase":
            effects["salary_override"] = interpretation["amount"]
            if interpretation.get("effective_date"):
                effects["salary_date_override"] = interpretation["effective_date"]
        elif msg_type == "salary_reduced":
            effects["salary_override"] = interpretation["amount"]
        elif msg_type == "salary_temporary_reduction":
            effects["salary_override"] = interpretation["amount"]
        elif msg_type == "first_salary":
            effects["salary_override"] = interpretation["amount"]
            if interpretation.get("effective_date"):
                effects["salary_date_override"] = interpretation["effective_date"]
        elif msg_type == "salary_base_confirmed":
            effects["salary_override"] = interpretation["amount"]
        elif msg_type == "salary_delayed":
            if interpretation.get("effective_date"):
                effects["salary_date_override"] = interpretation["effective_date"]
        elif msg_type == "salary_with_arrears":
            effects["salary_override"] = interpretation["amount"]
            if interpretation.get("details", {}).get("arrears"):
                effects["salary_arrears"] = interpretation["details"]["arrears"]
        elif msg_type == "salary_resumed_with_childcare":
            effects["salary_override"] = interpretation["amount"]
            if interpretation.get("effective_date"):
                effects["salary_date_override"] = interpretation["effective_date"]
            effects["childcare_deduction"] = True
        elif msg_type == "household_income_ended":
            effects["salary_override"] = interpretation["amount"]
        elif msg_type == "foreign_salary":
            if interpretation.get("amount") and interpretation.get("effective_date"):
                effects["confirmed_income"].append({
                    "amount": interpretation["amount"],
                    "date": interpretation["effective_date"],
                    "currency": interpretation["currency"],
                    "type": "foreign_salary",
                })
        elif msg_type == "invoice_confirmed":
            if interpretation.get("amount") and interpretation.get("effective_date"):
                effects["confirmed_income"].append({
                    "amount": interpretation["amount"],
                    "date": interpretation["effective_date"],
                    "currency": interpretation["currency"],
                    "type": "invoice",
                })
        elif msg_type == "rent_increase":
            pct = interpretation.get("details", {}).get("increase_pct")
            if pct:
                effects["rent_increase_pct"] = pct
        elif msg_type in ("investment_unrealized", "payout_pending", "refund_pending", 
                          "prize_pending", "bonus_pending", "card_dispute", "scam"):
            # These don't affect cash balance — ignore any related events
            if interpretation.get("related_event_id"):
                effects["pending_not_cash"].append(interpretation["related_event_id"])
        elif msg_type == "internal_transfer":
            # Internal transfers net to zero
            pass
        elif msg_type == "investment_settled":
            # The investment sale proceeds are already in the event record
            pass
        elif msg_type == "prize_settled":
            # The prize is already settled in the event record
            pass
        elif msg_type == "reimbursement":
            # Reimbursement is a one-time credit, already in events
            pass
    
    return effects
