import json
import re
from typing import Any

from pydantic import ValidationError
from scipy.optimize import linear_sum_assignment

from invoice_extractor.data.schema import InvoiceSchema


def extract_json_block(text: str) -> str:
    """Extracts the JSON substring from markdown code blocks or generic text."""
    # Find code block first
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Fallback: search for first '{' and last '}'
    first_bracket = text.find("{")
    last_bracket = text.rfind("}")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        return text[first_bracket : last_bracket + 1].strip()

    return text.strip()


def compute_normalized_edit_distance(s1: str, s2: str) -> float:
    """Computes Normalized Levenshtein Edit Distance similarity score in range [0, 1]."""
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)

    dist = dp[m][n]
    max_len = max(m, n)
    return 1.0 - (dist / max_len)


def evaluate_line_items(
    pred_items: list[dict[str, Any]],
    target_items: list[dict[str, Any]],
    similarity_threshold: float = 0.7,
) -> tuple[float, float, float]:
    """Evaluates line item lists using Bipartite Matching (Hungarian algorithm)."""
    if not pred_items and not target_items:
        return 1.0, 1.0, 1.0
    if not pred_items or not target_items:
        return 0.0, 0.0, 0.0

    num_pred = len(pred_items)
    num_target = len(target_items)
    cost_matrix = []

    for p_item in pred_items:
        row = []
        for t_item in target_items:
            # Field similarities
            desc_sim = compute_normalized_edit_distance(
                str(p_item.get("description", "")), str(t_item.get("description", ""))
            )
            qty_sim = (
                1.0 if int(p_item.get("quantity", 0)) == int(t_item.get("quantity", 0)) else 0.0
            )

            try:
                p_price = float(p_item.get("unit_price", 0.0))
                t_price = float(t_item.get("unit_price", 0.0))
                price_sim = 1.0 if abs(p_price - t_price) < 0.01 else 0.0
            except (ValueError, TypeError):
                price_sim = 0.0

            try:
                p_total = float(p_item.get("total_price", 0.0))
                t_total = float(t_item.get("total_price", 0.0))
                total_sim = 1.0 if abs(p_total - t_total) < 0.01 else 0.0
            except (ValueError, TypeError):
                total_sim = 0.0

            # Weighted average similarity
            sim = 0.4 * desc_sim + 0.2 * qty_sim + 0.2 * price_sim + 0.2 * total_sim
            row.append(1.0 - sim)
        cost_matrix.append(row)

    # Solve linear assignment
    pred_ind, target_ind = linear_sum_assignment(cost_matrix)

    tp = 0
    for p, t in zip(pred_ind, target_ind):
        cost = cost_matrix[p][t]
        sim = 1.0 - cost
        if sim >= similarity_threshold:
            tp += 1

    fp = num_pred - tp
    fn = num_target - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def categorize_error(raw_pred: str, target: dict[str, Any]) -> list[str]:
    """Classifies extraction failures into standard error categories."""
    errors = []

    json_str = extract_json_block(raw_pred)
    try:
        pred_dict = json.loads(json_str)
    except json.JSONDecodeError:
        return ["MalformedJSON"]

    try:
        InvoiceSchema.model_validate(pred_dict)
    except ValidationError as ve:
        errors.append("SchemaMismatch")
        for error in ve.errors():
            if error["type"] == "missing":
                errors.append("MissingField")
            elif error["type"] in ("type_error", "int_parsing", "float_parsing"):
                errors.append("TypeError")
        return list(set(errors))

    # Check scalars
    scalars = ["vendor", "invoice_date", "invoice_number", "currency", "tax_amount", "total_amount"]
    for key in scalars:
        if key not in pred_dict:
            errors.append("MissingField")
            continue

        t_val = target.get(key)
        p_val = pred_dict.get(key)

        if type(t_val) is not type(p_val):
            errors.append("TypeError")
        elif isinstance(t_val, float) and isinstance(p_val, float):
            if abs(t_val - p_val) >= 0.01:
                errors.append("ValueMismatch")
        elif str(t_val).strip().lower() != str(p_val).strip().lower():
            errors.append("ValueMismatch")

    # Check line items
    t_items = target.get("line_items", [])
    p_items = pred_dict.get("line_items", [])

    if len(t_items) != len(p_items):
        errors.append("LineItemMismatch")

    for p_item in p_items:
        if not isinstance(p_item, dict):
            errors.append("TypeError")
            continue
        for item_key in ["description", "quantity", "unit_price", "total_price"]:
            if item_key not in p_item:
                errors.append("SchemaMismatch")

    return list(set(errors))


def evaluate_prediction(raw_pred: str, target: dict[str, Any]) -> dict[str, Any]:
    """Computes detailed metrics for a prediction vs target."""
    json_str = extract_json_block(raw_pred)

    metrics: dict[str, Any] = {
        "json_valid": 0.0,
        "schema_compliant": 0.0,
        "exact_match": 0.0,
        "vendor_acc": 0.0,
        "vendor_sim": 0.0,
        "date_acc": 0.0,
        "invoice_number_acc": 0.0,
        "currency_acc": 0.0,
        "tax_amount_err": 1.0,
        "total_amount_err": 1.0,
        "line_items_precision": 0.0,
        "line_items_recall": 0.0,
        "line_items_f1": 0.0,
        "errors": [],
    }

    metrics["errors"] = categorize_error(raw_pred, target)

    try:
        pred_dict = json.loads(json_str)
        metrics["json_valid"] = 1.0
    except json.JSONDecodeError:
        return metrics

    try:
        InvoiceSchema.model_validate(pred_dict)
        metrics["schema_compliant"] = 1.0
    except ValidationError:
        pass

    target_canon = json.dumps(target, sort_keys=True)
    pred_canon = json.dumps(pred_dict, sort_keys=True)
    if target_canon == pred_canon:
        metrics["exact_match"] = 1.0

    metrics["vendor_sim"] = compute_normalized_edit_distance(
        str(pred_dict.get("vendor", "")), str(target.get("vendor", ""))
    )
    metrics["vendor_acc"] = 1.0 if metrics["vendor_sim"] > 0.9 else 0.0

    metrics["date_acc"] = (
        1.0
        if str(pred_dict.get("invoice_date", "")).strip()
        == str(target.get("invoice_date", "")).strip()
        else 0.0
    )
    metrics["invoice_number_acc"] = (
        1.0
        if str(pred_dict.get("invoice_number", "")).strip().lower()
        == str(target.get("invoice_number", "")).strip().lower()
        else 0.0
    )
    metrics["currency_acc"] = (
        1.0
        if str(pred_dict.get("currency", "")).strip().upper()
        == str(target.get("currency", "")).strip().upper()
        else 0.0
    )

    try:
        p_tax = float(pred_dict.get("tax_amount", 0.0))
        t_tax = float(target.get("tax_amount", 0.0))
        metrics["tax_amount_err"] = abs(p_tax - t_tax)
    except (ValueError, TypeError):
        metrics["tax_amount_err"] = 999.0

    try:
        p_tot = float(pred_dict.get("total_amount", 0.0))
        t_tot = float(target.get("total_amount", 0.0))
        metrics["total_amount_err"] = abs(p_tot - t_tot)
    except (ValueError, TypeError):
        metrics["total_amount_err"] = 999.0

    p_items = pred_dict.get("line_items", [])
    t_items = target.get("line_items", [])
    if isinstance(p_items, list) and isinstance(t_items, list):
        p_items_list = [item for item in p_items if isinstance(item, dict)]
        prec, rec, f1 = evaluate_line_items(p_items_list, t_items)
        metrics["line_items_precision"] = prec
        metrics["line_items_recall"] = rec
        metrics["line_items_f1"] = f1

    return metrics
