# 05. Data Specification

## 0. Data Strategy (Malaysia)
| Question | Answer |
|---|---|
| Is there a public customer-level churn dataset from Malaysia? | **No.** Malaysian telcos do not publish customer records; PDPA 2010 restricts sharing personal data |
| What real Malaysian data exists? | National-level figures only (subscriptions by plan type, annual) on data.gov.my |
| Approach | **Hybrid**: model on IBM Telco (real customer-level records, US origin) + frame the business case with real Malaysian market data |

| Layer | Dataset | Level | Role |
|---|---|---|---|
| Modelling | IBM Telco Customer Churn (section 1) | Customer | Train and evaluate model |
| Malaysia context | data.gov.my Cellular Subscribers by Plan Type | National, annual | EDA chart + business framing (postpaid vs prepaid trend) |
| Malaysia context | Telco annual reports / news (The Star, The Edge) | Company | Cite market facts in README and problem doc |

### Malaysia Context Dataset
| Item | Value |
|---|---|
| Name | Cellular Subscribers by Plan Type |
| Link | https://data.gov.my/data-catalogue/cellular_subscribers |
| CSV | https://storage.data.gov.my/communications/cellular_subscribers.csv |
| Columns | date, plan_type (total / postpaid / prepaid), subscriptions |
| Licence | CC BY 4.0 (credit data.gov.my) |
| Save to | `data/raw/my_cellular_subscribers.csv` |
| Note | Counts subscriptions, not unique people |

## 1. Source (Modelling Dataset)
| Item | Value |
|---|---|
| Dataset | IBM Telco Customer Churn |
| Link | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| File | `WA_Fn-UseC_-Telco-Customer-Churn.csv` -> save to `data/raw/telco_churn.csv` |
| Rows / Columns | 7,043 / 21 |
| Target | `Churn` (Yes/No), about 26.5% Yes |
| License | Public sample dataset (IBM); used for educational portfolio |

## 2. Schema
| Column | Type | Allowed values / range | Group |
|---|---|---|---|
| customerID | str | unique, not null | ID (drop for modelling) |
| gender | cat | Male, Female | Demographic |
| SeniorCitizen | int | 0, 1 | Demographic |
| Partner | cat | Yes, No | Demographic |
| Dependents | cat | Yes, No | Demographic |
| tenure | int | 0 to 72 (months) | Account |
| PhoneService | cat | Yes, No | Service |
| MultipleLines | cat | Yes, No, No phone service | Service |
| InternetService | cat | DSL, Fiber optic, No | Service |
| OnlineSecurity | cat | Yes, No, No internet service | Service |
| OnlineBackup | cat | Yes, No, No internet service | Service |
| DeviceProtection | cat | Yes, No, No internet service | Service |
| TechSupport | cat | Yes, No, No internet service | Service |
| StreamingTV | cat | Yes, No, No internet service | Service |
| StreamingMovies | cat | Yes, No, No internet service | Service |
| Contract | cat | Month-to-month, One year, Two year | Account |
| PaperlessBilling | cat | Yes, No | Account |
| PaymentMethod | cat | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) | Account |
| MonthlyCharges | float | over 0 | Billing |
| TotalCharges | str -> float | over or equal 0; **11 blank strings** where tenure = 0 | Billing |
| Churn | cat | Yes, No -> 1, 0 | Target |

## 3. Known Data Issues and Handling
| Issue | Handling | Decision ID |
|---|---|---|
| `TotalCharges` stored as string, 11 blanks (tenure = 0) | Convert to numeric; blanks -> 0 (new customers, nothing billed yet) | D-003 |
| "No internet service" / "No phone service" values | Keep as own category (carries information) | D-004 |
| Class imbalance (about 26.5%) | Class weights + threshold tuning; SMOTE only as an experiment inside CV | D-005 |
| No time dimension | Treat as snapshot; note limitation in README | - |

## 4. Validation Checks (`src/churnguard/data/validate.py`, pandera)
- Column set and dtypes match schema
- `customerID` unique, no nulls
- Categorical values within allowed sets
- `tenure` in 0 to 72; `MonthlyCharges` over 0
- `TotalCharges` at least `MonthlyCharges` when `tenure` at least 1 (warn, not fail)
- Target only {Yes, No}
- Row count at least 5,000 (guard against truncated files)

## 5. Feature Engineering Plan
| Feature | Formula / logic | Hypothesis |
|---|---|---|
| `tenure_bucket` | 0-6, 7-12, 13-24, 25-48, 49-72 months | Early-life customers churn most |
| `avg_monthly_spend` | TotalCharges / max(tenure, 1) | Spend history vs current charge |
| `charge_increase_ratio` | MonthlyCharges / avg_monthly_spend | Recent price rise drives churn |
| `num_services` | count of Yes across 8 add-on / service columns | More services = stickier |
| `has_protection_bundle` | OnlineSecurity or TechSupport = Yes | Support reduces churn |
| `is_auto_pay` | PaymentMethod contains "automatic" | Auto-pay = lower friction |
| `is_month_to_month` | Contract == Month-to-month | No lock-in |
| `fiber_no_support` | Fiber optic and TechSupport = No | Known high-risk combo |

All implemented in `FeatureEngineer` (sklearn transformer), unit-tested.

## 6. EDA Checklist (Phase 1)
- [ ] Target distribution
- [ ] Churn rate by each categorical (bar + rate table)
- [ ] Numeric distributions by churn (tenure, MonthlyCharges, TotalCharges)
- [ ] Correlation / Cramer's V for categoricals
- [ ] Churn by tenure bucket x contract (heatmap)
- [ ] Write **at least 5 business insights** in `notebooks/01_eda.ipynb` and copy to progress log

## 7. Data Governance
- `data/` gitignored; download script documents source
- No PII beyond synthetic `customerID`
- Fairness: track metrics by `gender` and `SeniorCitizen` (NFR7); these features are kept in v1 but the effect of removing them is tested (D-006)
