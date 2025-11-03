# FieldCloser – Context-Driven Release

Auto-injects Business Context from `/data/*.json` and uses **tabs** for outputs in Streamlit and Gradio. Tier shows as `<h3>` with tight bottom margin; option name is italic.

## Run backend
```bash
cd fieldcloser_context_release
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

## Test
```bash
curl -s -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d @sample_request.json > out.json

curl -s -X POST http://127.0.0.1:8000/render-html \
  -H "Content-Type: application/json" \
  -d @out.json | jq -r .html > option_sheet.html
```

## Streamlit
```bash
export BACKEND_URL=http://127.0.0.1:8000
streamlit run streamlit_app.py
```

## Gradio
```bash
export BACKEND_URL=http://127.0.0.1:8000
python gradio_app.py
```

## Edit business context
- data/pricing.json
- data/warranty.json
- data/membership.json
- data/catalog.json

Reload without restart:
```bash
curl -X POST http://127.0.0.1:8000/admin/reload-context
```

## Warranty Pricing
By default, FieldCloser will show warranty coverage text (from warranty.json) but will not add any cost for extended labor or system warranties unless you provide it.

If you’d like warranty upgrades to be included in tier pricing (e.g., Premium and Premium Plus), you can add fields under a prices object in your tenant’s warranty.json. FieldCloser will automatically detect and apply them.

Example warranty.json:
```json
{
  "repair_default": "90-day labor coverage on all repairs",
  "standard_replacement": "1-year parts + labor warranty on new systems",
  "premium_replacement": "10-year parts warranty, 1-year labor",
  "prices": {
    "extended_labor_5_years_cad": "450",
    "extended_labor_10_years_cad": "850"
  }
}
```
# How it works
If extended_labor_10_years_cad is present, FieldCloser will add that cost to the Premium Plus tier total.
If extended_labor_5_years_cad is present, you can configure config.json to use it for the Premium tier.
If no priced warranties are included, FieldCloser assumes $0 and simply lists the warranty benefits in text.

# Supported keys
FieldCloser will look for any of the following keys (case-insensitive):
extended_labor_10_years_cad
extended_labor_5_years_cad
extended_labor_cad
extended_labor
(Values should be in CAD unless you’re standardizing differently for your tenant.)

# Best practice
Keep warranty prices realistic and consistent with your vendor’s extended labor coverage pricing.
If you offer both 5-year and 10-year options, include both keys so FieldCloser can map them to Standard Plus / Premium tiers as appropriate.
If you only offer one, just provide that — FieldCloser will use it where it fits.

## 💳 Membership Plans & Pricing

FieldCloser can automatically include the cost of your maintenance membership programs in mid- to high-tier pricing (Standard Plus, Premium, Premium Plus).
By default, if a plan is referenced in the technician’s form (or set as a tenant default), FieldCloser will add the annualized membership fee into those tiers’ totals.

Example membership.json
```json
{
  "Cross Advantage Program": {
    "price_cad": "18.95",
    "billing_cycle": "monthly",
    "terms": "Semi-annual HVAC inspections, 1-year repair warranty, 10% discount on repairs, 20% discount on filters"
  },
  "VIP Comfort Club": {
    "price_cad": "199",
    "billing_cycle": "yearly",
    "terms": "Annual HVAC tune-up, priority scheduling, 15% repair discount"
  }
}
```

## How it works
FieldCloser looks for a numeric price_cad (or price_usd) and a billing_cycle string.
If the cycle includes "monthly", the system multiplies by 12 to compute the annual cost.
If the cycle includes "yearly", the system uses the number as-is.
The resulting annualized fee is added into the Standard Plus, Premium, and Premium Plus tier totals automatically.

# Supported fields
price_cad — Required. Should be a string or number representing the monthly or yearly fee.
billing_cycle — Required. Must include "monthly" or "yearly" (case-insensitive).
terms — Optional, but recommended. Customer-facing text listing the benefits. This will appear in the option sheet.

# Best practice
Keep membership fees in CAD (or USD if your business is US-based) and label them consistently.
Use "monthly" or "yearly" for billing cycles to ensure FieldCloser annualizes correctly.
Make the terms customer-friendly — this text is shown directly to homeowners in the proposal.

## 💵 Pricing Data

FieldCloser depends on your pricing.json to compute grounded, varied ranges for repairs, replacements, and IAQ add-ons.
This file is the backbone for tier totals. Keep it structured and complete for the best results.

Example pricing.json (simplified)
```json
{
  "defaults": {
    "labor_rate_per_hour_cad": 140,
    "refrigerant_r410a_per_lb_cad": 120
  },
  "repairs": {
    "furnace_ignitor_replacement": {
      "parts_range_cad": [90, 150],
      "labor_hours_range": [0.5, 1.0]
    },
    "condenser_fan_motor": {
      "parts_range_cad": [180, 350],
      "labor_hours_range": [1.0, 2.0]
    }
  },
  "replacements": {
    "2_5_ton_standard": {
      "installed_range_cad": [4200, 5800]
    },
    "2_5_ton_high_eff": {
      "installed_range_cad": [6000, 8200]
    }
  },
  "iaq_addons": {
    "uv_light": {
      "parts_range_cad": [350, 500],
      "labor_hours_range": [0.5, 1.0]
    },
    "media_filter": {
      "parts_range_cad": [150, 250],
      "labor_hours_range": [0.5, 1.0]
    }
  }
}
```

# How it works
Repairs: Each repair entry should include a parts price range and (ideally) labor hours. FieldCloser multiplies labor hours by the tenant’s labor_rate_per_hour_cad.

Replacements: Each replacement entry should include an installed price range. For tier variety, include at least one standard and one high-efficiency system.

IAQ Add-ons: Each IAQ item should include parts + labor ranges. These get added to mid and top tiers to expand scope.

Defaults: At minimum, include labor_rate_per_hour_cad. If refrigerant costs are relevant, add them as refrigerant_r410a_per_lb_cad.

# Best practice
Always use arrays [low, high] for ranges — FieldCloser expects two values.
Use consistent units (CAD across all entries).

Include at least:
4–6 common repairs
2+ replacement systems (standard vs high efficiency)
1–3 IAQ add-ons

Keep ranges realistic: too narrow and quotes look fake, too wide and they lose credibility.
