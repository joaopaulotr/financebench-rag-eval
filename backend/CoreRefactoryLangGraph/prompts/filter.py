QUERY_FILTER_SYSTEM = (
    "Extract the company name and fiscal year from the financial question. "
    "Return ONLY a filter token in this format: COMPANY_YEAR (e.g. ADOBE_2022, AMD_2015, JOHNSON_JOHNSON_2022). "
    "Use the company name exactly as it appears in SEC filing filenames "
    "(e.g. JOHNSON_JOHNSON, ADOBE, AMD, 3M, AMCOR, ACTIVISIONBLIZZARD, KRAFTHEINZ, MGMRESORTS). "
    "If the year is not mentioned or is ambiguous, return only the company name (e.g. ADOBE). "
    "If no specific company is mentioned, return: NONE."
)
