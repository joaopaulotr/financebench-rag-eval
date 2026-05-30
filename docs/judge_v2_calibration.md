# Judge v2 Calibration — Phase03b

**Change:** Replaced A|GT prompt with strict numerical extraction + tolerance rules.
**Rule:** Numbers differing >15% from ground truth → score ≤ 2, regardless of reasoning quality.

---

## Summary

| Metric | Judge v1 | Judge v2 | Delta |
|--------|---------|---------|-------|
| Nominal correct (n=100) | 63 | 51 | -12 |
| TPR (catches correct answers) | 0.93 | 0.71 | -0.21 |
| TNR (catches wrong answers) | 0.75 | 0.94 | +0.19 |
| False positives (30 sample) | 4 | 1 | -3 |
| Human-corrected estimate | ~47/100 | ~47/100 | — |

**Key insight:** v2 judge TNR improved from 0.75 → 0.94 — better at catching numerically wrong answers that sound convincing.

---

## Score Changes (36 queries)

| ID | v1 | v2 | Delta | Question |
|----|----|----|-------|---------|
| `financebench_id_01346` | 4 | 1 | -3 | How much has the effective tax rate of Corning changed betwe |
| `financebench_id_01487` | 4 | 1 | -3 | Did JnJ's net earnings as a percent of sales increase in Q2  |
| `financebench_id_00394` | 4 | 1 | -3 | In 2022 Q2, which of JPM's business segments had the highest |
| `financebench_id_01226` | 3 | 1 | -2 | What drove operating margin change as of FY2022 for 3M? If o |
| `financebench_id_01198` | 3 | 1 | -2 | What drove revenue change as of the FY22 for AMD? |
| `financebench_id_00070` | 3 | 1 | -2 | Does American Water Works have positive working capital base |
| `financebench_id_06272` | 4 | 2 | -2 | What is Coca Cola's FY2022 dividend payout ratio (using tota |
| `financebench_id_10130` | 3 | 1 | -2 | Based on the information provided primarily in the balance s |
| `financebench_id_00669` | 4 | 2 | -2 | What drove gross margin change as of FY2022 for JnJ? If gros |
| `financebench_id_00807` | 2 | 1 | -1 | Does 3M have a reasonably healthy liquidity profile based on |
| `financebench_id_07966` | 2 | 1 | -1 | What is the FY2017 - FY2019 3 year average of capex as a % o |
| `financebench_id_00438` | 2 | 1 | -1 | Does Adobe have an improving operating margin profile as of  |
| `financebench_id_10420` | 2 | 1 | -1 | Based on the information provided primarily in the statement |
| `financebench_id_06655` | 4 | 3 | -1 | What is Amazon's FY2017 days payable outstanding (DPO)? DPO  |
| `financebench_id_01935` | 4 | 3 | -1 | What was the key agenda of the AMCOR's 8k filing dated 1st J |
| `financebench_id_01079` | 3 | 2 | -1 | What are major acquisitions that AMCOR has done in FY2023, F |
| `financebench_id_01936` | 4 | 3 | -1 | What is the nature & purpose of AMCOR's restructuring liabil |
| `financebench_id_00222` | 5 | 4 | -1 | Does AMD have a reasonably healthy liquidity profile based o |
| `financebench_id_00917` | 4 | 3 | -1 | What drove operating margin change as of the FY22 for AMD? I |
| `financebench_id_00757` | 4 | 3 | -1 | Did AMD report customer concentration in FY22? |
| `financebench_id_01028` | 3 | 2 | -1 | What are the geographies that American Express primarily ope |
| `financebench_id_01964` | 2 | 1 | -1 | What was the largest liability in American Express's Balance |
| `financebench_id_00685` | 2 | 1 | -1 | Are Best Buy's gross margins historically consistent (not fl |
| `financebench_id_01275` | 3 | 2 | -1 | Among operations, investing, and financing activities, which |
| `financebench_id_00460` | 2 | 1 | -1 | Was there any change in the number of Best Buy stores betwee |
| `financebench_id_01091` | 4 | 3 | -1 | Has Boeing reported any materially important ongoing legal b |
| `financebench_id_00585` | 2 | 1 | -1 | How does Boeing's effective tax rate in FY2022 compare to FY |
| `financebench_id_03473` | 2 | 1 | -1 | What is the FY2017 return on assets (ROA) for Coca Cola? ROA |
| `financebench_id_10136` | 4 | 3 | -1 | We want to calculate a financial metric. Please help us comp |
| `financebench_id_00711` | 2 | 1 | -1 | Roughly how many times has JnJ sold its inventory in FY2022? |
| `financebench_id_02119` | 2 | 1 | -1 | If JPM went bankrupted by the end by 2021 Q1 and liquidated  |
| `financebench_id_03882` | 4 | 5 | +1 | What is Amcor's year end FY2020 net AR (in USD millions)? Ad |
| `financebench_id_01077` | 4 | 5 | +1 | What are major acquisitions that Best Buy has done in FY2023 |
| `financebench_id_01290` | 4 | 5 | +1 | Who are the primary customers of Boeing as of FY2022? |
| `financebench_id_00005` | 4 | 5 | +1 | Does Corning have positive working capital based on FY2022 d |
| `financebench_id_00790` | 4 | 5 | +1 | Is CVS Health a capital-intensive business based on FY2022 d |

---

## False Positive Analysis

### v1 False Positives (4 cases)

**`financebench_id_00222`** (score 5/5)
- Q: Does AMD have a reasonably healthy liquidity profile based on its quick ratio fo
- Expected: `Yes. The quick ratio is 1.57, calculated as (cash and cash equivalents+Short ter`
- Model: `To assess AMD's liquidity profile for FY22, we need to look at the quick ratio, which is calculated as follows:

\[
\tex`

**`financebench_id_06272`** (score 4/5)
- Q: What is Coca Cola's FY2022 dividend payout ratio (using total cash dividends pai
- Expected: `0.8`
- Model: `To calculate Coca-Cola's FY2022 dividend payout ratio, we need the total cash dividends paid and the net income attribut`

**`financebench_id_10136`** (score 4/5)
- Q: We want to calculate a financial metric. Please help us compute it by basing you
- Expected: `0.54`
- Model: `To calculate the FY2022 retention ratio for General Mills, we need the net income attributable to shareholders and total`

**`financebench_id_01487`** (score 4/5)
- Q: Did JnJ's net earnings as a percent of sales increase in Q2 of FY2023 compared t
- Expected: `Yes, net earnings as a percent of sales increased from 20% in Q2 of FY2022 to 20`
- Model: `From the available data, here are the net earnings and sales percentages for Johnson & Johnson (J&J) in Q2 of FY2023 com`

### v2 False Positives (1 cases)

**`financebench_id_00222`** (score 4/5)
- Q: Does AMD have a reasonably healthy liquidity profile based on its quick ratio fo
- Expected: `Yes. The quick ratio is 1.57, calculated as (cash and cash equivalents+Short ter`

---

## Conclusion

v2 judge is stricter on numerical accuracy. Nominal 'correct' count dropped from 63 → 51/100. TNR improved 0.75 → 0.94, meaning fewer wrong-but-fluent answers pass as correct. Human-calibrated real accuracy remains ~47/100.
