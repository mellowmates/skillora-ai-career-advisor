"""
Utils Package - Common utility functions for Skillora AI
"""

def sanitize_text(text: str) -> str:
    """Simple text sanitizer to prevent XSS or injection"""
    import html
    return html.escape(text)

def format_currency(amount: float, currency_symbol: str = '₹') -> str:
    """Format number as currency string"""
    return f"{currency_symbol}{amount:,.2f}"

# Add more shared helpers as needed
