"""Convert Jalali calendar dates entered in the admin to Gregorian dates."""

from datetime import date


def _jalali_to_gregorian(year: int, month: int, day: int) -> date:
    # Integer calendar conversion; the admin supports contemporary dates.
    jy = year + 1595
    days = -355668 + 365 * jy + (jy // 33) * 8 + ((jy % 33) + 3) // 4 + day
    days += (month - 1) * 31 if month <= 6 else 186 + (month - 7) * 30
    gy = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        gy += 100 * ((days - 1) // 36524)
        days = (days - 1) % 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1
    leap = gy % 4 == 0 and (gy % 100 != 0 or gy % 400 == 0)
    month_days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    gm = 1
    for length in month_days:
        if gd <= length:
            break
        gd -= length
        gm += 1
    return date(gy, gm, gd)


def parse_jalali_date(value: str) -> date:
    digits = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
    parts = value.translate(digits).strip().replace("-", "/").split("/")
    if len(parts) != 3 or any(not part.isdecimal() for part in parts):
        raise ValueError("Enter a Jalali date as YYYY/MM/DD.")
    year, month, day = map(int, parts)
    if not 1200 <= year <= 1700 or not 1 <= month <= 12:
        raise ValueError("Enter a valid Jalali date.")
    first = _jalali_to_gregorian(year, month, 1)
    next_month = _jalali_to_gregorian(
        year + (month == 12), 1 if month == 12 else month + 1, 1
    )
    if not 1 <= day <= (next_month - first).days:
        raise ValueError("Enter a valid Jalali day for that month.")
    return _jalali_to_gregorian(year, month, day)
