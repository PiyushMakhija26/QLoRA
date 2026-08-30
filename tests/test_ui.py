import json

from invoice_extractor.ui.app import parse_and_validate, rule_based_mock_extractor


def test_parse_and_validate_success() -> None:
    # 1. Perfect compliance
    valid_json = """
    {
      "vendor": "Apex",
      "invoice_date": "2026-08-30",
      "invoice_number": "INV-101",
      "currency": "USD",
      "line_items": [
        {"description": "Item A", "quantity": 1, "unit_price": 10.0, "total_price": 10.0}
      ],
      "tax_amount": 1.0,
      "total_amount": 11.0
    }
    """
    res = parse_and_validate(valid_json)
    assert res["is_valid_json"] is True
    assert res["is_schema_compliant"] is True
    assert res["math_match"] is True
    assert res["error_msg"] == ""


def test_parse_and_validate_failures() -> None:
    # 1. Invalid JSON string
    res1 = parse_and_validate("{bad_json: missing_quotes}")
    assert res1["is_valid_json"] is False
    assert res1["is_schema_compliant"] is False
    assert "JSON Decode Error" in res1["error_msg"]

    # 2. Schema Mismatch (missing vendor)
    invalid_schema = """
    {
      "invoice_date": "2026-08-30",
      "invoice_number": "INV-101",
      "currency": "USD",
      "line_items": [],
      "tax_amount": 0.0,
      "total_amount": 0.0
    }
    """
    res2 = parse_and_validate(invalid_schema)
    assert res2["is_valid_json"] is True
    assert res2["is_schema_compliant"] is False

    # 3. Mathematical Summation Mismatch
    mismatch_math = """
    {
      "vendor": "Apex",
      "invoice_date": "2026-08-30",
      "invoice_number": "INV-101",
      "currency": "USD",
      "line_items": [
        {"description": "Item A", "quantity": 1, "unit_price": 10.0, "total_price": 10.0}
      ],
      "tax_amount": 1.0,
      "total_amount": 99.0
    }
    """
    res3 = parse_and_validate(mismatch_math)
    assert res3["is_valid_json"] is True
    assert res3["is_schema_compliant"] is True
    assert res3["math_match"] is False


def test_rule_based_mock_extractor() -> None:
    text = "From: Corner Cafe Cafe Supplies  Total: $122.50  Tax: $10.00"
    extracted_str = rule_based_mock_extractor(text)

    # Verify valid JSON
    extracted = json.loads(extracted_str)
    assert extracted["vendor"] == "Corner Cafe Cafe Supplies"
    assert extracted["currency"] == "USD"
    assert extracted["tax_amount"] == 10.0
    assert extracted["total_amount"] == 122.50
    assert isinstance(extracted["line_items"], list)
    assert len(extracted["line_items"]) == 1
