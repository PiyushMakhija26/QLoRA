from invoice_extractor.evaluation.metrics import (
    categorize_error,
    compute_normalized_edit_distance,
    evaluate_line_items,
    evaluate_prediction,
    extract_json_block,
)


def test_extract_json_block() -> None:
    # 1. Backticks formatting
    txt1 = 'Here is the output:\n```json\n{"vendor": "Apex"}\n```\nHope it helps.'
    assert extract_json_block(txt1) == '{"vendor": "Apex"}'

    # 2. Backticks format without json label
    txt2 = '```\n{"vendor": "Apex"}\n```'
    assert extract_json_block(txt2) == '{"vendor": "Apex"}'

    # 3. Plain text format with loose brackets
    txt3 = 'Vendor is Apex, invoice details are: {"vendor": "Apex", "items": []}.'
    assert extract_json_block(txt3) == '{"vendor": "Apex", "items": []}'


def test_compute_normalized_edit_distance() -> None:
    # 1. Exact match
    assert compute_normalized_edit_distance("Apex Corp", "Apex Corp") == 1.0
    # 2. Minor differences
    assert compute_normalized_edit_distance("Apex Corp", "Apex Clrp") > 0.8
    # 3. Completely different
    assert compute_normalized_edit_distance("Apex Corp", "Google Inc") < 0.3
    # 4. Empty strings
    assert compute_normalized_edit_distance("", "") == 1.0


def test_evaluate_line_items() -> None:
    targets = [
        {"description": "Server Hosting", "quantity": 1, "unit_price": 50.0, "total_price": 50.0},
        {"description": "Support Hours", "quantity": 5, "unit_price": 80.0, "total_price": 400.0},
    ]

    # 1. Perfect prediction (order shuffled)
    preds_perfect = [
        {"description": "Support Hours", "quantity": 5, "unit_price": 80.0, "total_price": 400.0},
        {"description": "Server Hosting", "quantity": 1, "unit_price": 50.0, "total_price": 50.0},
    ]
    prec, rec, f1 = evaluate_line_items(preds_perfect, targets)
    assert prec == 1.0
    assert rec == 1.0
    assert f1 == 1.0

    # 2. Partial prediction (one missing, one correct)
    preds_partial = [
        {"description": "Server Hosting", "quantity": 1, "unit_price": 50.0, "total_price": 50.0}
    ]
    prec, rec, f1 = evaluate_line_items(preds_partial, targets)
    assert prec == 1.0
    assert rec == 0.5
    assert f1 == 2 / 3

    # 3. Messy description typo and close price
    preds_typo = [
        {"description": "Server H0sting", "quantity": 1, "unit_price": 50.0, "total_price": 50.0},
        {"description": "Support Hours", "quantity": 5, "unit_price": 80.0, "total_price": 400.0},
    ]
    prec, rec, f1 = evaluate_line_items(preds_typo, targets)
    # Both should match as the similarity is above threshold (0.7)
    assert f1 == 1.0


def test_categorize_error() -> None:
    target = {
        "vendor": "Apex Global Solutions",
        "invoice_date": "2026-08-29",
        "invoice_number": "INV-2026-01",
        "currency": "USD",
        "line_items": [
            {"description": "Hosting", "quantity": 1, "unit_price": 100.0, "total_price": 100.0}
        ],
        "tax_amount": 10.0,
        "total_amount": 110.0,
    }

    # 1. Malformed JSON
    assert "MalformedJSON" in categorize_error("Not a json at all", target)

    # 2. Schema Mismatch (missing date)
    bad_pred1 = '{"vendor": "Apex Global Solutions", "invoice_number": "INV-2026-01", "currency": "USD", "line_items": [], "tax_amount": 0.0, "total_amount": 0.0}'
    errs1 = categorize_error(bad_pred1, target)
    assert "SchemaMismatch" in errs1 or "MissingField" in errs1

    # 3. Value Mismatch
    bad_pred2 = """
    {
        "vendor": "Apex Global Sol",
        "invoice_date": "2026-08-29",
        "invoice_number": "INV-2026-01",
        "currency": "USD",
        "line_items": [
            {"description": "Hosting", "quantity": 1, "unit_price": 100.0, "total_price": 100.0}
        ],
        "tax_amount": 10.0,
        "total_amount": 110.0
    }
    """
    assert "ValueMismatch" in categorize_error(bad_pred2, target)


def test_evaluate_prediction() -> None:
    target = {
        "vendor": "Apex Global Solutions",
        "invoice_date": "2026-08-29",
        "invoice_number": "INV-2026-01",
        "currency": "USD",
        "line_items": [
            {"description": "Hosting", "quantity": 1, "unit_price": 100.0, "total_price": 100.0}
        ],
        "tax_amount": 10.0,
        "total_amount": 110.0,
    }

    raw_pred = f"```json\n{import_json_str(target)}\n```"
    m = evaluate_prediction(raw_pred, target)
    assert m["json_valid"] == 1.0
    assert m["schema_compliant"] == 1.0
    assert m["exact_match"] == 1.0
    assert m["vendor_acc"] == 1.0
    assert m["date_acc"] == 1.0
    assert len(m["errors"]) == 0


def import_json_str(obj: dict) -> str:
    import json

    return json.dumps(obj)
