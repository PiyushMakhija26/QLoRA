import hashlib
import json
import random
from datetime import datetime, timedelta
from typing import Any

from invoice_extractor.data.schema import InvoiceSchema, LineItem

VENDORS = [
    "Apex Global Solutions",
    "QuickPrint Inc.",
    "Corner Cafe",
    "Initech Office Supplies",
    "Prime Web Services",
    "Alpha Construction",
    "Vortex Media",
    "Blue Ribbon Services",
    "NextGen Consulting",
    "Stellar Retailers",
    "Green Valley Farms",
    "Metro Tech Repair",
    "Deli & Co",
    "Urban Fashion House",
    "Horizon Software",
    "Silver Star Logistics",
]

ITEMS_POOL = [
    ("Cloud Server Hosting", 49.99),
    ("Database Subscription", 120.0),
    ("Python Development Support", 85.0),
    ("Office Desk Chair", 150.0),
    ("Coffee Beans 1kg", 24.50),
    ("Paper Shredder Service", 35.0),
    ("Web Domain Registry", 12.0),
    ("High-speed Ethernet Cable", 18.99),
    ("SaaS Enterprise License", 299.99),
    ("Corporate Training Session", 500.0),
    ("Wireless Mouse & Keyboard", 45.50),
    ("USB-C Docking Station", 89.99),
    ("A4 Printer Paper Ream", 8.50),
    ("Office Coffee Machine", 349.99),
    ("Data Storage 1TB SSD", 110.0),
    ("Network Security Audit", 750.0),
]

CURRENCIES = ["USD", "EUR", "GBP"]
TAX_RATES = [0.0, 0.05, 0.10, 0.15, 0.20]


def get_random_date(rng: random.Random) -> tuple[str, str]:
    """Generates a random date and returns (standardized_str, display_str)."""
    start_date = datetime(2020, 1, 1)
    days_to_add = rng.randint(0, 2400)
    date_obj = start_date + timedelta(days=days_to_add)

    standardized = date_obj.strftime("%Y-%m-%d")

    # Random display format
    fmt = rng.choice(["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY", "Month DD, YYYY"])
    if fmt == "YYYY-MM-DD":
        display = standardized
    elif fmt == "DD/MM/YYYY":
        display = date_obj.strftime("%d/%m/%Y")
    elif fmt == "MM/DD/YYYY":
        display = date_obj.strftime("%m/%d/%Y")
    else:
        display = date_obj.strftime("%B %d, %Y")

    return standardized, display


def get_random_invoice_number(rng: random.Random) -> str:
    """Generates various invoice number formats."""
    prefix = rng.choice(["INV", "TX", "REC", ""])
    year = rng.randint(2020, 2026)
    num = rng.randint(1000, 99999)
    if prefix:
        return f"{prefix}-{year}-{num}"
    return f"{year}{num}"


def format_price(val: float, currency: str, rng: random.Random) -> str:
    """Formats price in various currency styles."""
    style = rng.choice(["symbol_first", "code_first", "code_last"])
    symbol = {"USD": "$", "EUR": "€", "GBP": "£"}[currency]

    if style == "symbol_first":
        return f"{symbol}{val:.2f}"
    if style == "code_first":
        return f"{currency} {val:.2f}"
    return f"{val:.2f} {currency}"


def inject_ocr_noise(text: str, rng: random.Random, probability: float = 0.02) -> str:
    """Injects OCR substitutions and messy spacing into the text."""
    substitutions = {"0": "O", "1": "l", "2": "Z", "5": "S", "8": "B", "S": "5", "o": "0"}

    chars = list(text)
    for i in range(len(chars)):
        # OCR character replacement
        if chars[i] in substitutions and rng.random() < probability:
            chars[i] = substitutions[chars[i]]

        # Messy spacing
        if chars[i] == " " and rng.random() < probability:
            chars[i] = "  "

    return "".join(chars)


def generate_invoice(seed: int) -> dict[str, Any]:
    """Generates a single synthetic invoice data point (noisy text + target json schema)."""
    rng = random.Random(seed)

    vendor = rng.choice(VENDORS)
    invoice_number = get_random_invoice_number(rng)
    std_date, display_date = get_random_date(rng)
    currency = rng.choice(CURRENCIES)
    tax_rate = rng.choice(TAX_RATES)

    # Sample items
    num_items = rng.randint(1, 5)
    selected_items = rng.sample(ITEMS_POOL, num_items)

    line_items = []
    subtotal = 0.0
    for desc, unit_p in selected_items:
        # Slightly perturb unit price
        perturbed_price = round(unit_p * rng.uniform(0.9, 1.1), 2)
        qty = rng.randint(1, 6)
        total_p = round(qty * perturbed_price, 2)
        subtotal += total_p

        line_items.append(
            LineItem(
                description=desc, quantity=qty, unit_price=perturbed_price, total_price=total_p
            )
        )

    tax_amount = round(subtotal * tax_rate, 2)
    total_amount = round(subtotal + tax_amount, 2)

    target_obj = InvoiceSchema(
        vendor=vendor,
        invoice_date=std_date,
        invoice_number=invoice_number,
        currency=currency,
        line_items=line_items,
        tax_amount=tax_amount,
        total_amount=total_amount,
    )

    # Create the raw text using one of 4 templates
    template_type = rng.choice(["formal", "email", "thermal", "messy_scan"])

    if template_type == "formal":
        table_rows = []
        for item in line_items:
            price_str = format_price(item.unit_price, currency, rng)
            total_str = format_price(item.total_price, currency, rng)
            table_rows.append(
                f"{item.description:<30} {item.quantity:<5} {price_str:<12} {total_str:<12}"
            )
        items_table = "\n".join(table_rows)

        raw_text = f"""
============================================================
                      I N V O I C E
============================================================
Vendor: {vendor}
Invoice No: #{invoice_number}
Date: {display_date}
Currency: {currency}
------------------------------------------------------------
Description                    Qty   Unit Price   Total Price
------------------------------------------------------------
{items_table}
------------------------------------------------------------
Subtotal: {format_price(subtotal, currency, rng)}
Tax ({tax_rate * 100:.0f}%): {format_price(tax_amount, currency, rng)}
Total Due: {format_price(total_amount, currency, rng)}
============================================================
"""
    elif template_type == "email":
        items_list = []
        for item in line_items:
            items_list.append(
                f"- {item.quantity}x {item.description} @ {format_price(item.unit_price, currency, rng)} each (Total: {format_price(item.total_price, currency, rng)})"
            )
        items_list_str = "\n".join(items_list)

        raw_text = f"""
From: billing@{vendor.lower().replace(" ", "")}.com
To: accounts-payable@clientcorp.com
Date: {display_date}
Subject: Invoice {invoice_number} from {vendor}

Dear Customer,

Please find attached invoice #{invoice_number} for your recent purchase.

Details:
{items_list_str}

Summary:
Subtotal: {format_price(subtotal, currency, rng)}
Tax: {format_price(tax_amount, currency, rng)}
Amount Due: {format_price(total_amount, currency, rng)}

Please remit payment at your earliest convenience.

Best regards,
{vendor} Billing Dept.
"""
    elif template_type == "thermal":
        items_receipt = []
        for item in line_items:
            items_receipt.append(
                f"{item.description}\n  {item.quantity} * {format_price(item.unit_price, currency, rng)} = {format_price(item.total_price, currency, rng)}"
            )
        items_receipt_str = "\n".join(items_receipt)

        raw_text = f"""
----------------------------------------
              {vendor.upper()}
----------------------------------------
REC NO: {invoice_number}
DATE: {display_date}
----------------------------------------
{items_receipt_str}
----------------------------------------
SUBTOTAL: {format_price(subtotal, currency, rng)}
TAX: {format_price(tax_amount, currency, rng)}
TOTAL: {format_price(total_amount, currency, rng)}
----------------------------------------
           THANK YOU FOR SHOPPING
----------------------------------------
"""
    else:  # messy scan
        table_rows = []
        for item in line_items:
            price_str = format_price(item.unit_price, currency, rng)
            total_str = format_price(item.total_price, currency, rng)
            table_rows.append(f"{item.description}   {item.quantity}   {price_str}   {total_str}")
        items_list_str = " | ".join(table_rows)

        raw_text = f"""
OCR SCAN START >>>
{vendor} - INVOICE_DOC
Num: {invoice_number} Date: {display_date} Cur: {currency}
ITEMS DETAIL: {items_list_str}
TAX_VAL: {format_price(tax_amount, currency, rng)} TOTAL_SUM: {format_price(total_amount, currency, rng)}
<<< OCR SCAN END
"""
        raw_text = inject_ocr_noise(raw_text, rng, probability=0.04)

    if template_type != "messy_scan":
        raw_text = inject_ocr_noise(raw_text, rng, probability=0.01)

    return {"text": raw_text.strip(), "target": target_obj.model_dump()}


def generate_dataset(num_samples: int, seed: int) -> list[dict[str, Any]]:
    """Generates multiple synthetic invoice examples."""
    dataset = []
    # Seed the RNG sequence so each sample has a unique but reproducible seed
    rng = random.Random(seed)
    seeds = [rng.randint(0, 1000000) for _ in range(num_samples)]
    for s in seeds:
        dataset.append(generate_invoice(s))
    return dataset


def save_dataset_and_calculate_checksum(dataset: list[dict[str, Any]], filepath: str) -> str:
    """Saves the dataset to a JSONL file and returns the SHA256 checksum."""
    import os

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Calculate SHA256
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()
