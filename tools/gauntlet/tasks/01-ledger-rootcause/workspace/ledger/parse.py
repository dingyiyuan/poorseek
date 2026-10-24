"""Amount parsing for bank-statement exports."""
from decimal import Decimal, InvalidOperation


def parse_amount(text):
    """Parse one amount cell from a bank export into an exact Decimal.

    Cents matter, so the value is exact: this function never rounds.  A cell written with
    three decimals ('1.005') comes back as Decimal('1.005') - quantization belongs to the
    callers that compute totals, not to this function.

    Accepted forms (surrounding whitespace is ignored):
      '$1,234.50'    -> Decimal('1234.50')  optional '$'; thousands separated by commas
      '(12.50)'      -> Decimal('-12.50')   accounting parentheses mean negative
      '($1,000.00)'  -> Decimal('-1000.00') the accounting form may wrap the '$' and commas
      '-$3.10'       -> Decimal('-3.10')    a leading minus before the '$' also works
      '3.10-'        -> Decimal('-3.10')    trailing-minus form (bank export style)
      '1234.50 USD'  -> Decimal('1234.50')  a trailing currency code, separated by a space
      '1234.50 EUR'  -> Decimal('1234.50')  ... exactly three UPPERCASE letters, ignored
      '0.10'         -> Decimal('0.10')     a plain cell keeps the scale it was written with

    Anything else raises ValueError, in particular:
      - a separator that is not a group of exactly three ('1,23.00', '12,3456.00')
      - no digits, or more than one '.' ('', '$', '1.2.3')
      - a sign carrying two markers at once, which is ambiguous: '(3.10-)', '(-3.10)'
      - unbalanced parentheses ('(1.00')
      - a currency code that is not three uppercase letters ('12.00 usd', '12.00 US')
    """
    t = text.strip().replace("$", "").replace(",", "")
    try:
        return Decimal(t)
    except InvalidOperation:
        raise ValueError(f"not an amount: {text!r}") from None
