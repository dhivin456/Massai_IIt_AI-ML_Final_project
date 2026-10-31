# Instamart Customer Intelligence & Support Copilot

## Overview

A reproducible, four-part capstone implementation using deterministic fictional data and the offline `mock_llm.py`. No network calls, API keys, paid services, or real customer data are used. All monetary values are in INR.

## Reproduce

Run in order from the repository root with Python 3.11 or newer. On Windows, activate `.venv\Scripts\activate` instead of the POSIX activation command.

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`
5. `python generate_data.py`
6. `python src/part_a.py`
7. `python src/part_b.py`
8. `python src/part_c.py`
9. `python src/part_d.py`
10. `python src/validate.py`

The generator must print exactly `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The numbered commands regenerate raw inputs, clean CSVs, SQLite, model artifacts, evaluations, triage outputs, ledgers and checkpoints. `requirements.txt` pins the direct dependencies; `.gitignore` excludes `.env`, virtual environments and Python caches.

## Part A

### Raw audit

**`customers.csv` — shape `(1200, 5)`**

| Column | Raw dtype | Missing count | Missing % |
|---|---|---:|---:|
| `customer_id` | `object` | 0 | 0.000% |
| `city` | `object` | 0 | 0.000% |
| `signup_date` | `object` | 0 | 0.000% |
| `age` | `float64` | 61 | 5.083% |
| `gender` | `object` | 30 | 2.500% |

**`products.csv` — shape `(40, 4)`**

| Column | Raw dtype | Missing count | Missing % |
|---|---|---:|---:|
| `product_id` | `object` | 0 | 0.000% |
| `product_name` | `object` | 0 | 0.000% |
| `category` | `object` | 0 | 0.000% |
| `price_inr` | `int64` | 0 | 0.000% |

**`orders.csv` — shape `(13084, 9)`**

| Column | Raw dtype | Missing count | Missing % |
|---|---|---:|---:|
| `order_id` | `object` | 0 | 0.000% |
| `customer_id` | `object` | 0 | 0.000% |
| `product_id` | `object` | 0 | 0.000% |
| `quantity` | `int64` | 0 | 0.000% |
| `order_ts` | `object` | 0 | 0.000% |
| `status` | `object` | 0 | 0.000% |
| `discount_pct` | `int64` | 0 | 0.000% |
| `order_amount` | `object` | 53 | 0.405% |
| `delivery_minutes` | `float64` | 2541 | 19.421% |

**Specific raw audit values**

| Measure | Count |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Pending rows | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric amounts | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2,541 |
| Blank ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank gender spellings | 10 |
| Blank genders | 30 |

### Cleaning, storage and validation

- Customers: standardisation changed 476 city strings, collapsing 24 raw spellings to six cities; 950 gender values were normalised to `male`/`female`/`unknown` (30 blanks become `unknown`). Of 1,200 ages, 61 were blank and 6 were outside 10–100; all 67 values were imputed using the city median with `groupby` + `transform`. All 1,200 signup dates were parsed.
- Orders: all 13,084 timestamps were parsed; 119 exact duplicate rows were removed (13,084 → 12,965), then 656 earlier/history rows were removed by retaining the latest timestamp per `order_id` (12,965 → 12,309). The raw export had 120 non-numeric amounts; 8 were on rows removed during deduplication, and the remaining 112 amounts were filled using `round(price_inr × quantity × (1 − discount_pct/100), 2)`. The raw export had 106 negative delivery durations; 105 remained after deduplication and were set missing. Of 1,868 missing delivery durations after deduplication, all 669 missing values on completed orders were imputed using the median of completed orders in that product category; all 1,199 cancelled-order durations remained missing.
- Products are preserved from the synthetic catalogue. Clean CSVs are stored in `data/clean/`; SQLite tables are `customers`, `products`, and `orders`, with timestamps stored as ISO text.

| Clean checkpoint | Result |
|---|---:|
| Customers | 1200 |
| Products | 40 |
| Orders | 12309 |
| Completed orders | 11110 |
| Cancelled orders | 1199 |
| Completed revenue (INR) | 2,998,668.80 |
| Missing completed delivery times | 0 |
| Missing cancelled delivery times | 1199 |

Clean city counts: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Gender counts: female 610, male 560, unknown 30. Clean ages have no missing values and range from 18 to 60.

### SQL and pandas analysis

Seven queries in `sql/queries.sql` are run from Python with `pandas.read_sql_query`.

| Query | Result |
|---|---|
| Q1 WHERE then HAVING | 30 customers |
| Q2 city × category revenue | 30 cells; total INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 price above catalogue average | 11 products |
| Q4 no completed order | 25 customers |
| Q5 spend above average | 550 customers; average INR 2,552.06 |
| Q6 city spend rankings | 1175 customers with spend; 8 rows have RANK different from ROW_NUMBER |
| Q7 running daily revenue | 183 days; final cumulative INR 2,998,668.80 |

Top completed-spend customer per city: Bengaluru C0082 INR 12,856.20; Chennai C0930 INR 10,151.40; Delhi C0291 INR 13,742.60; Hyderabad C0743 INR 12,730.05; Mumbai C0693 INR 14,766.55; Pune C0426 INR 10,377.10.
RANK and DENSE_RANK give tied values the same rank; ROW_NUMBER always assigns a unique position, so ties make RANK differ from ROW_NUMBER.

The pandas pivot table with margins totals INR 2,998,668.80, matching SQL. Completed-order amount mean INR 269.91, median INR 135.00, skewness 4.609, Q1 INR 64.80, Q3 INR 297.00, lower/upper IQR fences INR -283.50 / INR 645.30; outliers below/above fences: 0/897.

| City | Sum (INR) | Mean (INR) | Count |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1894 |
| Chennai | 463,629.15 | 272.72 | 1700 |
| Delhi | 502,829.80 | 243.62 | 2064 |
| Hyderabad | 528,589.90 | 284.95 | 1855 |
| Mumbai | 479,058.50 | 297.00 | 1613 |
| Pune | 500,918.05 | 252.48 | 1984 |

Correlation matrix (completed orders):

| | Order amount | Discount % | Quantity | Delivery minutes |
|---|---:|---:|---:|---:|
| Order amount | 1.0000 | -0.0951 | 0.0873 | -0.0031 |
| Discount % | -0.0951 | 1.0000 | -0.0004 | -0.0098 |
| Quantity | 0.0873 | -0.0004 | 1.0000 | -0.0126 |
| Delivery minutes | -0.0031 | -0.0098 | -0.0126 | 1.0000 |

**Exploratory summary (under 200 words).** These synthetic records describe completed grocery and small-electronics orders across six cities. Amounts are strongly right-skewed: mean INR 269.91 exceeds median INR 135.00, skewness is about 4.61, and 897 orders exceed the INR 645.30 upper fence. This reflects a mix of low-cost repeat essentials and fewer high-value purchases. The largest absolute correlation is order amount versus discount percentage (about −0.095), followed by order amount versus quantity (about +0.087). Both associations are weak; correlation of this size does not establish causation. Bengaluru / Electronics is the largest city-category revenue cell.

`python src/validate.py` checks the reference checkpoints and prints `ALL CHECKPOINTS PASSED` when all checks hold.

## Part B

### RFM segmentation

RFM snapshot: 2025-10-01 00:00:00. There are 1,175 customers with completed orders; recency median/min/max 19/0/180 days, frequency mean/max 9.46/33, monetary mean/max INR 2,552.06/INR 14,766.55.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 is selected because it has the best silhouette score in this sweep and gives actionable profiles. Inertia decreases as K increases because more centroids fit the data more closely; lower inertia at K=8 than K=2 alone does not mean eight segments are better.

| Segment ID | Name | Size | Mean recency | Mean frequency | Mean monetary (INR) | Action | Premium? |
|---:|---|---:|---:|---:|---:|---|---|
| 0 | At-Risk / Dormant | 436 | 28.86 days | 3.77 | 1,057.62 | Win-back message and reactivation offer | No |
| 1 | Premium High Value | 138 | 24.12 days | 6.77 | 7,129.70 | Concierge service and high-value retention offers | Yes |
| 2 | Frequent Essentials | 419 | 8.77 days | 19.40 | 3,410.02 | Replenishment reminders and bundles | No |
| 3 | At-Risk / Dormant | 182 | 109.04 days | 2.20 | 685.99 | Win-back message and reactivation offer | No |

PCA explained variance ratios: 0.634, 0.218; total 0.852. Cluster IDs are labels, not an ordinal ranking. `outputs/customer_segments.csv` contains all required fields.

### Lapse prediction

Pre-cutoff cohort: 1156 customers, 443 lapsed (38.32%); train/test rows 924/232. Baseline always predicts not-lapsed: accuracy 0.616, lapsed-class F1 0.0.

Features use only final-status orders dated strictly before 2025-09-01 plus the clean customer table; no orders on/after 2025-09-01 enter features, and the RFM segment labels are not features. September completed-order activity defines the label only. Numeric scalers and one-hot encoders (`drop="first"`) are fit within training-only pipelines; the test set is only transformed, never fitted. Code asserts no missing feature values. The scaler training mean of frequency is 8.115, versus full-cohort mean 8.029, because the scaler is fit only on the training split.

The tuning grids are `LogisticRegression(max_iter=1000)` with `C ∈ {0.01, 0.1, 1, 10, 100}` and `RandomForestClassifier(n_estimators=100, random_state=42)` with `max_depth ∈ {3, 5, 8}` and `min_samples_leaf ∈ {5, 10, 20}` (45 forest fits, five-fold CV; scoring is positive-class F1).

| Model | Best parameters | CV F1 | Test confusion matrix (rows true 0,1; columns predicted 0,1) | Precision | Recall | F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic regression | `{"model__C": 100}` | 0.651 | `[[111, 32], [31, 58]]` | 0.644 | 0.652 | 0.648 |
| Random forest | `{"model__max_depth": 8, "model__min_samples_leaf": 5}` | 0.689 | `[[105, 38], [35, 54]]` | 0.587 | 0.607 | 0.597 |

Selected model: **random_forest**, test lapsed F1 0.597. Missing a lapsing customer loses a retention opportunity; targeting a loyal customer wastes campaign budget and may annoy them. The selected model meets the required F1 threshold of 0.55.

Class-balance experiment (oversampling training data only): unbalanced logistic precision/recall/F1 0.644/0.652/0.648; balanced logistic 0.589/0.854/0.697. Recall moves most, helping find more at-risk customers at the cost of more false positives.

Top five random-forest features:

| Feature | Importance | Business reading |
|---|---:|---|
| `num__frequency` | 0.268 | Ordering habit strength |
| `num__monetary` | 0.242 | Historical customer value |
| `num__recency` | 0.113 | Time since last purchase |
| `num__avg_delivery` | 0.065 | Delivery experience |
| `num__cancellation_rate` | 0.057 | Potential friction signal |

Three largest-magnitude logistic coefficient signs:

| Feature | Sign and coefficient | Reading |
|---|---|---|
| `num__frequency` | negative (-1.697) | Lower modelled lapse odds |
| `cat__favourite_category_Electronics` | negative (-1.135) | Lower modelled lapse odds |
| `cat__favourite_category_Grocery` | negative (-0.449) | Lower modelled lapse odds |

Reloaded model reproduces test predictions exactly: **True**. Premium test retention list: 10 rows, 2 truly lapsed. **The retention list is illustrative, not a leakage-free evaluation, because its Premium flag is a post-hoc whole-window label that includes September activity.**

## Part C

### Guardrails, privacy and classification

Four empty tickets are `skipped_empty`; four long tickets are `flagged_long` (734 estimated tokens each, above the 400-token cap). Redaction order is email → `[EMAIL]`, phone → `[PHONE]`, order ID → `[ORDER_ID]`: 55 emails, 91 phones, 132 order IDs; 278 substitutions total, and 0 PII-pattern matches remain after redaction. The model user message is exactly the redacted ticket text.

Both system prompts include CLASSIFY, the four allowed categories, and JSON-only output instructions. Zero-shot does not contain `EXAMPLES:`; few-shot includes labelled examples including fraud. Calls retry transient failures with exponential backoff and jitter; all 19 transient errors are recovered. Invalid JSON/category outputs are never guessed and are routed to human review.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| JSON classifications | 135 | 135 |
| Keyword classifications | 12 | 12 |
| Invalid outputs | 9 | 9 |
| Accuracy on classified tickets | 85.7% | 100.0% |
| Total response tokens | 12364 | 24376 |
| Tokens per classified ticket | 84.1 | 165.8 |
| Successful model calls | 156 | 156 |
| Recovered transient failures | 19 | 19 |

Zero-shot confusion matrix (true category rows; predicted category columns):

| True / predicted | Billing | Delivery | Fraud | Technical |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Fraud | 11 | 0 | 23 | 0 |
| Technical | 10 | 0 | 0 | 29 |

Few-shot uses 81.7 extra tokens per classified ticket and improves accuracy from 85.7% to 100.0%. That cost is worthwhile when fraud misrouting is expensive; lower-risk, high-volume workflows should compare the token cost against the cost of errors.

Token-budget demo: budget 6000; 80 tickets were marked `budget_exhausted`. Final few-shot run uses budget 100,000. `outputs/triage_results.csv` has 164 rows and routes guardrail/invalid/budget statuses to `human_review`/NORMAL, fraud to `security_team`/HIGH, Premium to `premium_desk`/HIGH, and all others to `support_queue`/NORMAL. Only validated classifications are auto-routed; flagged outputs are routed to humans.

## Part D

### Policy RAG

The five policy files yield 15 chunks with 40-word windows and 10-word overlap: delivery_policy.txt 3, escalation_policy.txt 3, payment_policy.txt 3, privacy_policy.txt 2, refund_policy.txt 4. TF measures within-chunk term frequency; IDF downweights terms common across chunks. TF-IDF is lexical and can miss equivalent phrasings that use different words.

| Evaluation | 10-word overlap | Zero overlap |
|---|---:|---:|
| Expected document in top three | 12/12 | 12/12 |
| Expected fact substring | 12/12 | 10/12 |
| Correct unanswerable fallback | 2/2 | 2/2 |

Zero-overlap expected-fact failures: What delivery time makes a completed order eligible for a full refund?; How soon is money returned when a payment fails but is deducted?. Relevant wording is split across chunk boundaries. Both unanswerable questions receive the fallback beginning `I could not find`. For “refund policy late delivery”, the top three chunks are all from `refund_policy.txt`. `answer(question)` sends the required CONTEXT/QUESTION structure and returns the answer and sources.

### Refund tools, ReAct loop and recovery

The five declared tools are `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`, mapped through `TOOL_MAP`. Orders come from Part A SQLite; policy search reuses the same TF-IDF retriever. Only completed orders delivered in more than 45 minutes qualify for a 100% refund; refunds above INR 1,000 are escalated and not automatically issued. Cancelled/on-time orders are denied. Issued/escalated actions append timestamped JSON lines to `outputs/refund_ledger.jsonl`.

| Scenario | Status | Refund amount (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 917 |
| S2 | REFUND | 135.00 | 5 | 917 |
| S3 | ESCALATE | 1,999.00 | 5 | 904 |
| S4 | DENIED | 0.00 | 4 | 621 |
| S5 | DENIED | — | 3 | 349 |
| S6 | NOT_FOUND | — | 2 | 80 |
| S7 | FALLBACK_MAX_ITERATIONS | — | 6 | 3024 |

Token-limit demo (S1, limit 200): **FALLBACK_TOKEN_LIMIT**. Timeout demo (time limit 0 seconds): **FALLBACK_TIMEOUT**. Both return a human-handoff summary and completed steps.

Crash-and-resume result: **REFUND**, `REFUND: INR 128.25 issued.`. Tools executed across crash and resume, in order: get_order → search_policy → calculate_refund → issue_refund. The checkpoint is saved after each observation, so completed tool calls are not repeated.

### ReAct versus plan-and-execute (under 250 words)

ReAct chooses the next tool after seeing each observation: S1 takes 5 model calls, S5 takes 3, S6 takes 2, and the loop-protection case S7 stops after 6. This usually costs more tokens and latency than plan-and-execute, which can ask for a complete plan in one call and replan only on errors. Refunds have consequential side effects: an order may be missing, cancelled, on-time, or above the auto-approval limit. ReAct can inspect actual database results before calculating, denying, issuing, or escalating, so it is less likely to follow a stale plan. Plan-and-execute is efficient for predictable cases but needs validation before every action. For this workflow I prefer the guarded ReAct loop because decisions depend on current tool observations and it has iteration, token, timeout and checkpoint controls. A production system should also use idempotency keys and transactional ledger writes.

## Limitations and Next Steps

- Data are synthetic and the model is an offline deterministic mock; results do not establish real-world performance.
- TF-IDF is lexical rather than semantic; evaluate paraphrases, adversarial questions and policy changes.
- Lapse prediction uses one fixed temporal split; monitor by city and segment, calibrate probabilities, and evaluate on a later untouched time window.
- The Premium flag is based on whole-window RFM including September activity. The retention list is illustrative and not a leakage-free evaluation, although model features use only pre-cutoff orders.
- Refund ledger actions are simulated; production needs transactions, idempotency, access controls, audit retention and human approval integration.
- Re-test privacy redaction whenever ticket formats change.

## Repository map

```text
generate_data.py                 # deterministic synthetic data generator
mock_llm.py                      # offline mock LLM
src/part_a.py                   # audit, cleaning, SQLite, SQL/pandas analysis
src/part_b.py                   # RFM, clustering, PCA and lapse models
src/part_c.py                   # redaction, retries, validation and routing
src/part_d.py                   # policy RAG, refund agent, checkpoints/resume
src/validate.py                 # acceptance-critical validation
sql/queries.sql                 # seven SQL queries
data/clean/                     # clean CSV tables
data/instamart.db               # SQLite analytical database
outputs/                        # metrics, models, triage and agent logs
state/                          # resumable agent checkpoints
```
