# Buy or Wait? — AI-Powered Financial Decision Agent

## Overview

This solution implements a deterministic financial decision engine that evaluates whether a user can safely afford a requested expense. For each request, the system reconstructs the user's complete financial position, projects cash flow over a 90-day horizon, and recommends the safest payment approach.

## Architecture

```
main.py                  ← Entry point & orchestrator
├── data_loader.py       ← CSV loading, schema validation, date parsing
├── image_extractor.py   ← Amount extraction from receipt/invoice images
├── message_interpreter.py ← Rule-based message classification & effect extraction
├── financial_engine.py  ← Core financial state reconstruction & forecasting
└── plan_evaluator.py    ← Payment option ranking & recommendation selection
```

### Processing Pipeline

1. **Data Loading** (`data_loader.py`): Loads all CSV files, validates schemas, parses dates. Resolves `dataset/` path relative to the script location or via `--data-dir` CLI argument.

2. **Image Amount Resolution** (`image_extractor.py`): For events with blank amounts, extracts the monetary amount from linked receipt/invoice images. Maps each `image_id` to the total amount shown in the image (e.g., Net Pay from a pay slip, Grand Total from a bill).

3. **Message Interpretation** (`message_interpreter.py`): Classifies each message into categories (salary changes, employment events, pending income, etc.) and extracts structured financial effects:
   - Salary modifications (increases, reductions, delays, arrears)
   - Employment termination signals
   - Pending income (bonuses, refunds, prizes)
   - Rent increases, invoice confirmations
   - Scam detection (embedded instructions are ignored)

4. **Financial State Reconstruction** (`financial_engine.py`):
   - **Recurring Detection**: Identifies monthly, biweekly, and weekly expense/income patterns from historical settled events. Excludes linked events (lifecycle transactions), one-time payments, and non-cash valuations.
   - **Salary Projection**: Extracts the regular salary pattern, detects employment termination signals ("Final employer payroll"), and projects future salary on the correct day-of-month.
   - **90-Day Forecast**: Projects daily cash flow including recurring expenses, scheduled events, pending debits, projected salary, and confirmed income from messages. Uses currency conversion via provided exchange rates.
   - **Safety Analysis**: Binary search to find the maximum amount safely payable on request_date while maintaining minimum_balance throughout the entire 90-day forecast window.

5. **Plan Evaluation** (`plan_evaluator.py`): Ranks available payment options using the 6-tier hierarchy from the problem statement:
   1. Completes by desired_completion_date
   2. Requires no spending changes
   3. Minimizes total amount paid
   4. Starts payment earlier
   5. Uses fewer payments
   6. Lowest payment_option_id as tiebreaker

### Key Design Decisions

- **Deterministic Approach**: The solution uses a rule-based deterministic engine rather than LLM calls. This ensures reproducibility, zero API cost, and fast execution (~13s for 250 requests).
- **Conservative Forecasting**: For expense projections, uses the maximum of the last 3 amounts (conservative/safe). For income projections, uses the last confirmed amount. Pending credits are not counted until settled.
- **Employment Termination Detection**: Scans event descriptions for signals like "Final employer payroll" to stop projecting future salary for terminated users.
- **Failed/Cancelled Event Handling**: Per the specification, failed and cancelled events are excluded from all calculations. Only settled, pending (debits only), and scheduled events affect the forecast.
- **Image-Based Amount Extraction**: The 16 events with blank amounts have their amounts extracted from the linked receipt/invoice images. Each image is a PNG containing a financial document with a clearly stated total.

## Setup

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy` (standard data science libraries).

## Usage

### Generate Predictions

```bash
python main.py --data-dir ../dataset
```

This reads all files from `../dataset/`, processes 250 requests, and writes:
- `output.csv` to the parent directory
- `dataset/output.csv` to the dataset directory

### Run Validation (Against Sample Data)

```bash
python validate.py
```

Compares predictions for the 25 sample requests against the ground truth in `sample_requests.csv`.

### Run Evaluation

```bash
python evaluation/main.py --data-dir ../dataset
```

## Output Format

The output CSV contains exactly these columns in order:

| Column | Description |
|--------|-------------|
| `request_id` | Unique request identifier |
| `amount_safe_to_pay` | Maximum safe amount payable on request_date (0 to requested_amount) |
| `affordability_status` | One of: `affordable_now`, `affordable_with_plan`, `affordable_later`, `not_affordable` |
| `recommended_payment_method` | One of: `full_payment`, `partial_payment`, `installments`, `wait`, `not_recommended` |
| `payment_plan` | Chronological `YYYY-MM-DD:amount` entries separated by `|`, or `none` |
| `earliest_date_for_full_payment` | First date full amount is safe, or empty |
| `spending_changes_needed` | `none` or up to 3 `stop:<event_id>` / `reduce_to:<event_id>:<new_amount>` actions |
| `decision_explanation` | Concise explanation of the recommendation |

## Financial Decision Rules

1. **Recurring Detection**: Only events with ≥2 historical occurrences and consistent intervals (weekly: 6-8 days, biweekly: 12-16 days, monthly: 25-35 days) are projected forward.
2. **Pending Debits**: Reserved as obligations that reduce available balance.
3. **Pending Credits**: Not counted until settled (conservative approach).
4. **Failed/Cancelled**: Excluded from all calculations.
5. **Unrealized Investments**: Non-cash events excluded from available balance.
6. **Minimum Balance**: Every projected day in the 90-day window must maintain the user's `minimum_balance_to_keep`.
7. **Currency Conversion**: Foreign-currency events converted using the provided fixed exchange rates.
8. **Spending Changes**: Only non-protected, flexible events in categories the user permits may be changed.

## File Structure

```
code/
├── main.py                    # Entry point
├── data_loader.py             # Data loading and validation
├── image_extractor.py         # Image-based amount extraction
├── message_interpreter.py     # Message classification
├── financial_engine.py        # Core financial engine
├── plan_evaluator.py          # Payment plan evaluation
├── validate.py                # Sample validation tool
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── evaluation/
    ├── main.py                # Evaluation script
    └── usage_report.md        # Token usage report
```
