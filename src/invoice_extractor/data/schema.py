from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str = Field(description="Description of the item or service.")
    quantity: int = Field(description="Quantity of the item or service.")
    unit_price: float = Field(description="Price per unit of the item.")
    total_price: float = Field(description="Total price of the item (quantity * unit_price).")


class InvoiceSchema(BaseModel):
    vendor: str = Field(description="Name of the vendor/merchant issuing the invoice.")
    invoice_date: str = Field(description="Date the invoice was issued (in YYYY-MM-DD format).")
    invoice_number: str = Field(description="Unique identifier/number for the invoice.")
    currency: str = Field(description="Three-letter currency code (e.g. USD, EUR, GBP).")
    line_items: list[LineItem] = Field(description="List of line items in the invoice.")
    tax_amount: float = Field(description="Calculated tax amount.")
    total_amount: float = Field(description="Grand total amount of the invoice.")
