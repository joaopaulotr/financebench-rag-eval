# Missing PDFs Audit

**Date:** 2026-05-26
**Dataset:** PatronusAI/financebench (HuggingFace)

| Stat | Count |
|------|-------|
| Dataset unique docs | 84 |
| PDFs in `data/pdfs/` | 79 |
| Missing | 5 |

## Missing Documents

| doc_name | download_link |
|----------|---------------|
| JOHNSON_JOHNSON_2022Q4_EARNINGS | https://johnsonandjohnson.gcs-web.com/static-files/ca8c3ac2-15ab-4f8d-9693-f604d50be358 |
| JOHNSON_JOHNSON_2022_10K | https://johnsonandjohnson.gcs-web.com/static-files/9b012500-471a-4df9-93fc-6cee2b420678 |
| JOHNSON_JOHNSON_2023Q2_EARNINGS | https://johnsonandjohnson.gcs-web.com/static-files/6626623f-0619-46dc-b7b6-57568124c517 |
| JOHNSON_JOHNSON_2023_8K_dated-2023-08-30 | https://johnsonandjohnson.gcs-web.com/static-files/fa9ff302-f93d-450a-a73a-2ac9fb67d2ee |
| MGMRESORTS_2022Q4_EARNINGS | https://s22.q4cdn.com/513010314/files/doc_financials/2022/q4/r/MGM-Resorts-Exhibit-99.1.pdf |

## Notes

- All 4 missing JnJ docs explain why JnJ queries have high retrieval miss rate — chunks never ingested
- MGM miss affects 1 query
- Adobe PDFs were missing earlier but downloaded from SEC EDGAR and re-ingested before Phase03 run
