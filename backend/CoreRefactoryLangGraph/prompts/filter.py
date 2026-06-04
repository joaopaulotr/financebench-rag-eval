QUERY_FILTER_SYSTEM = (
    "Extract the company name and fiscal year from the financial question. "
    "Return a JSON object with two fields:\n"
    "- 'filter_token': company name + year combined (e.g. ADOBE_2022, JOHNSON_JOHNSON_2022). "
    "  If year is unknown, return just the company name (e.g. ADOBE). "
    "  If no company is mentioned, return NONE.\n"
    "- 'company_filter': company name only, no year (e.g. ADOBE, JOHNSON_JOHNSON, 3M). "
    "  If no company is mentioned, return NONE.\n"
    "Use SEC filing filename conventions: JOHNSON_JOHNSON, ADOBE, AMD, 3M, AMCOR, "
    "ACTIVISIONBLIZZARD, KRAFTHEINZ, MGMRESORTS, JPMORGAN, COCACOLA, BESTBUY, etc.\n"
    "Return ONLY the JSON object, nothing else."
)
