"""Human-readable product labels for reports and charts.

Keyed by (product_code, bs_side) because some codes appear on both sides
(7900 = interbank placement on A / interbank deposit on L; 0000 = the two IRS
legs), so a code-only lookup can't tell them apart.
"""

PRODUCT_LABELS = {
    ("2200", "A"): "Credit Cards — Revolvers",
    ("2210", "A"): "Credit Cards — Transactors",
    ("0000", "A"): "IRS receive leg",
    ("0000", "L"): "IRS pay leg",
    ("0100", "A"): "FX deals — currency received",
    ("0100", "L"): "FX deals — currency paid",
    ("1000", "A"): "Mortgage Fixed",
    ("1100", "A"): "Mortgage Float",
    ("2000", "A"): "Consumer Loan Fixed",
    ("2100", "A"): "Consumer Loan Float",
    ("3000", "A"): "Bond Fixed",
    ("3100", "A"): "Bond Float",
    ("3200", "A"): "T-Bill",
    ("3300", "A"): "Corporate Bond AA Fixed",
    ("3310", "A"): "Corporate Bond BBB Fixed",
    ("3320", "A"): "Corporate Bond BB Fixed",
    ("3330", "A"): "Corporate FRN A",
    ("3500", "A"): "Cash (vault + min. reserves)",
    ("3510", "A"): "Central Bank Reserves (DFR)",
    ("4000", "A"): "SME Investment Loan Fixed",
    ("4100", "A"): "SME Investment Loan",
    ("4200", "A"): "Corporate Investment Loan",
    ("4300", "A"): "Corporate Investment Loan Fixed",
    ("7900", "A"): "Interbank Placement",
    ("1050", "A"): "Mortgage Fixed GBP",
    ("3050", "A"): "US Treasury Bond",
    ("3060", "A"): "UK Gilt",
    ("4150", "A"): "Corporate Loan Float USD",
    ("4250", "A"): "Corporate Loan Float GBP",
    ("4350", "A"): "Corporate Loan Fixed USD",
    ("6360", "L"): "Current Account (Corporate) USD",
    ("6370", "L"): "Current Account (Corporate) GBP",
    ("7080", "L"): "Term Deposit (Corporate) USD",
    ("7085", "L"): "Term Deposit (Corporate) GBP",
    ("7910", "L"): "Interbank Deposit USD",
    ("5000", "L"): "Issued Bond",
    ("6000", "L"): "Current Account (Retail)",
    ("6300", "L"): "Current Account (SME)",
    ("6350", "L"): "Current Account (Corporate)",
    ("7060", "L"): "Term Deposit",
    ("7070", "L"): "Term Deposit (SME)",
    ("7900", "L"): "Interbank Deposit",
    ("8000", "L"): "Savings Account",
    ("5100", "E"): "Common Shares",
    ("5300", "E"): "Retained Earnings",
    ("5400", "E"): "Risk Reserves",
}


def product_label(product_code, bs_side):
    """Readable label; falls back to the raw code so a new product never crashes a report."""
    return PRODUCT_LABELS.get((str(product_code), str(bs_side)), str(product_code))
