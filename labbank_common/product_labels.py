"""Human-readable product labels for reports and charts.

Keyed by (product_code, bs_side) because some codes appear on both sides
(7900 = interbank placement on A / interbank deposit on L; 0000 = the two IRS
legs), so a code-only lookup can't tell them apart.
"""

PRODUCT_LABELS = {
    ("0000", "A"): "IRS receive leg",
    ("0000", "L"): "IRS pay leg",
    ("1000", "A"): "Mortgage Fixed",
    ("1100", "A"): "Mortgage Float",
    ("2000", "A"): "Consumer Loan Fixed",
    ("2100", "A"): "Consumer Loan Float",
    ("3000", "A"): "Bond Fixed",
    ("3100", "A"): "Bond Float",
    ("3200", "A"): "T-Bill",
    ("3500", "A"): "Cash / Central Bank",
    ("4100", "A"): "SME Investment Loan",
    ("7900", "A"): "Interbank Placement",
    ("5000", "L"): "Issued Bond",
    ("6000", "L"): "Current Account (Retail)",
    ("6300", "L"): "Current Account (SME)",
    ("7060", "L"): "Term Deposit",
    ("7900", "L"): "Interbank Deposit",
    ("8000", "L"): "Savings Account",
    ("5100", "E"): "Common Shares",
    ("5300", "E"): "Retained Earnings",
    ("5400", "E"): "Risk Reserves",
}


def product_label(product_code, bs_side):
    """Readable label; falls back to the raw code so a new product never crashes a report."""
    return PRODUCT_LABELS.get((str(product_code), str(bs_side)), str(product_code))
