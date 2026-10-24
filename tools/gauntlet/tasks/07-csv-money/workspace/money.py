"""Consolidate a directory of bank exports into one money report.

Public API: build_report(data_dir, out_path) -> None.  It reads every export in
data_dir, writes the report to out_path (UTF-8, '\n' line ends) and returns None.

data_dir holds exactly one non-export file, 'rates.yaml', and any number of bank
export files; every other immediate file is an export.  Exports are processed in
ascending filename order.  (Grading feeds data_dir more exports than data/ holds.)

RATE FILE
  rates.yaml is a flat mapping, one "key: value" per line; blank lines and lines
  starting with '#' are ignored:
      base: USD      -> the base currency; all report amounts are in it
      EUR: 1.0855    -> each other key is a 3-letter currency code whose value is
      ...               the units of the base per ONE unit of that currency
  An amount in currency C is normalised to the base as amount * rate[C].

DIALECT DETECTION - by header signature, never by the delimiter
  A file's dialect is decided ONLY by the field names on its first line.  Strip a
  leading BOM, then split that line on a single delimiter from ',', ';', '\t'.
  The dialect is the one whose field names (below, compared stripped and
  lower-cased) equal the resulting cells for SOME delimiter, and that delimiter is
  the file's delimiter.  So a file whose field names are dialect A's but which is
  ';'-separated is dialect A, not B.  No dialect matches -> ValueError.
    A: date, account, description, amount     (US numbers)
    B: datum, konto, beschreibung, betrag     (German numbers; UTF-8 BOM, CRLF)
    C: posted, account, memo, amount          (trailing minus; a summary row)

AMOUNT CELLS
  Every amount cell is "<number> <CODE>": a number in the dialect's format, one
  space, then a 3-letter UPPERCASE currency code.  The number's format:
    A  1,234.56 / -12.00    optional '-' sign, ',' thousands, '.' decimal
    B  1.234,56 / -12,00    optional '-' sign, '.' thousands, ',' decimal
    C  1234.56 / 12.50-     no thousands separators; a TRAILING '-' is negative
  A malformed number, a bad code, or a code missing from rates.yaml -> ValueError.

ROWS
  Fields are separated by the dialect's delimiter; text fields may be quoted and
  may contain the delimiter, quotes ("" escaped) or newlines.  A blank row is
  skipped.  In dialect C a row whose account cell is 'TOTAL' is the trailing
  summary row and is skipped (it is not an account).

NORMALISATION AND ROUNDING - per row, then sum
  Each row's amount is normalised to the base currency and then quantized to 2
  decimal places with ROUND_HALF_EVEN, PER ROW.  Kept rows are summed afterwards,
  so a group's net is the sum of its rows' already-quantized values, never a
  quantization of the group's raw sum.  The two differ: e.g. two 0.01 CHF rows at
  rate 1.4 are 0.01 + 0.01 = 0.02, not quantize(0.028) = 0.03.

DUPLICATES - within one file
  Within a single file, rows identical on (date, account, normalised amount) are
  duplicates: the first occurrence in input order is kept and every later one is
  dropped.  Dropped duplicates are listed in the report (see below) in a stable
  order - files by ascending name, then rows in input order.

OUTPUT - UTF-8, '\n' line ends, exactly these lines
  [account-month] then 'account,month,net,count' then one row per (account,
  month) with 'net' a 2-decimal string (e.g. -12.30) and 'month' 'YYYY-MM',
  sorted by account then month ascending;
  [account] then 'account,net,count' then one row per account, sorted by account
  ascending;
  [ledger] then 'account,date,net,balance' then one line per KEPT row (a dropped
  duplicate never appears), grouped by account ascending; within an account the
  rows are in date-ascending order, and rows sharing a date keep INPUT ORDER
  (files by ascending name, then rows in the order they appear in the file).
  'balance' is the running sum, within that account, of these rows' quantized
  nets up to and including that row - the SAME per-row quantization as above, so
  a dropped duplicate never moves a balance, every account in [account] has at
  least one [ledger] row, and an account's last balance equals its [account] net;
  [total] then '<base>,<net>' - the grand total, the sum of every kept row;
  [duplicates] then 'file,date,account,net' then one row per dropped duplicate
  ('file' its base name, 'net' the 2-decimal normalised amount).
  The file ends with a newline: every line above, the last one included, is
  terminated by '\n', so the text is the lines joined by '\n' plus a final '\n'.
"""


def build_report(data_dir, out_path):
    raise NotImplementedError
