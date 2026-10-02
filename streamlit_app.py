import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet

st.set_page_config(
    page_title="Grocery Demand Forecasting",
    page_icon="📦",
    layout="wide",
)


@st.cache_data
def load_data():
    df = pd.read_csv("data/favorita_weekly_sales.csv")
    df["Week_Start"] = pd.to_datetime(df["Week_Start"])
    df["Week_End"] = pd.to_datetime(df["Week_End"])
    return df


def build_model():
    return Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.80,
    )


def estimate_promotion_uplift(category_df):
    """Estimate a simple historical promotion scenario uplift.

    We compare promoted, non-holiday weeks with the mean of the previous
    four weeks. Because promotion effects are irregular, we use the median
    of positive observed uplifts and exclude very large (>50%) outliers.
    This is a scenario estimate, not a causal claim.
    """
    temp = category_df.sort_values("Week_Start").copy()
    temp["Previous_4_Week_Avg"] = (
        temp["Weekly_Sales"].shift(1).rolling(window=4, min_periods=4).mean()
    )

    promo_weeks = temp[
        (temp["Promotion_Flag"] == 1)
        & (temp["Holiday_Event_Flag"] == 0)
        & temp["Previous_4_Week_Avg"].notna()
        & (temp["Previous_4_Week_Avg"] > 0)
    ].copy()

    promo_weeks["Observed_Uplift"] = (
        promo_weeks["Weekly_Sales"] - promo_weeks["Previous_4_Week_Avg"]
    ) / promo_weeks["Previous_4_Week_Avg"]

    usable = promo_weeks[
        (promo_weeks["Observed_Uplift"] > 0)
        & (promo_weeks["Observed_Uplift"] <= 0.50)
    ]["Observed_Uplift"]

    if len(usable) < 5:
        return 0.0, len(usable)

    return float(usable.median()), len(usable)


# -----------------------------
# Load data and select category
# -----------------------------
df = load_data()

st.title("AI-Powered Grocery Demand Forecasting & Inventory Decision Support")
st.caption("Demand Forecasting & Inventory Decision Support • Corporación Favorita")

st.sidebar.header("Decision Controls")

categories = sorted(df["Product_Category"].dropna().unique())
selected_category = st.sidebar.selectbox(
    "Select product category",
    categories,
)

category_df = (
    df[df["Product_Category"] == selected_category]
    .sort_values("Week_Start")
    .copy()
)

# Prophet format: ds = date, y = target value
series = category_df[["Week_Start", "Weekly_Sales"]].rename(
    columns={"Week_Start": "ds", "Weekly_Sales": "y"}
)

# -----------------------------
# 1) Forecast validation
# -----------------------------
train = series.iloc[:-4].copy()
test = series.iloc[-4:].copy()

validation_model = build_model()
validation_model.fit(train)
validation_forecast = validation_model.predict(test[["ds"]])

validation_results = test.copy()
validation_results["Predicted"] = validation_forecast["yhat"].clip(lower=0).values
validation_results["Actual"] = validation_results["y"]

mae = float(np.mean(np.abs(validation_results["Actual"] - validation_results["Predicted"])))
nonzero = validation_results["Actual"] != 0
if nonzero.any():
    mape = float(
        np.mean(
            np.abs(
                (validation_results.loc[nonzero, "Actual"] - validation_results.loc[nonzero, "Predicted"])
                / validation_results.loc[nonzero, "Actual"]
            )
        )
        * 100
    )
else:
    mape = np.nan

# -----------------------------
# 2) Future four-week forecast
# -----------------------------
final_model = build_model()
final_model.fit(series)
future_dates = final_model.make_future_dataframe(
    periods=4,
    freq="W-MON",
    include_history=False,
)
future_forecast = final_model.predict(future_dates)[
    ["ds", "yhat", "yhat_lower", "yhat_upper"]
].copy()

for col in ["yhat", "yhat_lower", "yhat_upper"]:
    future_forecast[col] = future_forecast[col].clip(lower=0)

# -----------------------------
# 3) Promotion scenario
# -----------------------------
promotion_uplift, promo_observations = estimate_promotion_uplift(category_df)

promotion_choice = st.sidebar.selectbox(
    "Promotion planned",
    ["No promotion", "Week 1", "Week 2", "Week 3", "Week 4"],
)

future_forecast["Baseline_Forecast"] = future_forecast["yhat"]
future_forecast["Adjusted_Forecast"] = future_forecast["Baseline_Forecast"]
future_forecast["Promotion_Adjustment"] = 0.0

if promotion_choice != "No promotion" and promotion_uplift > 0:
    promo_week_index = int(promotion_choice.split()[-1]) - 1
    baseline_value = future_forecast.loc[promo_week_index, "Baseline_Forecast"]
    adjusted_value = baseline_value * (1 + promotion_uplift)
    future_forecast.loc[promo_week_index, "Adjusted_Forecast"] = adjusted_value
    future_forecast.loc[promo_week_index, "Promotion_Adjustment"] = adjusted_value - baseline_value

baseline_total = float(future_forecast["Baseline_Forecast"].sum())
adjusted_total = float(future_forecast["Adjusted_Forecast"].sum())

# -----------------------------
# 4) Inventory scenario inputs
# -----------------------------
default_inventory = max(0, int(round(adjusted_total * 0.60)))
default_safety = max(0, int(round(adjusted_total * 0.10)))

current_inventory = st.sidebar.number_input(
    "Current inventory (scenario)",
    min_value=0,
    value=default_inventory,
    step=100,
)

safety_stock = st.sidebar.number_input(
    "Safety stock (scenario)",
    min_value=0,
    value=default_safety,
    step=100,
)

recommended_replenishment = max(
    0,
    int(np.ceil(adjusted_total + safety_stock - current_inventory)),
)

if current_inventory < adjusted_total:
    stock_risk = "High"
elif current_inventory < adjusted_total + safety_stock:
    stock_risk = "Medium"
else:
    stock_risk = "Low"

# -----------------------------
# Dashboard summary
# -----------------------------
summary1, summary2, summary3, summary4 = st.columns(4)
summary1.metric("Selected category", selected_category)
summary2.metric("Historical weeks", len(category_df))
summary3.metric("4-week adjusted demand", f"{adjusted_total:,.0f}")
summary4.metric("Recommended order", f"{recommended_replenishment:,.0f}")

# -----------------------------
# Historical sales
# -----------------------------
st.subheader("Historical Weekly Sales")
historical_chart = category_df.set_index("Week_Start")[["Weekly_Sales"]]
st.line_chart(historical_chart)

# -----------------------------
# Forecast validation output
# -----------------------------
st.subheader("Forecast Validation — Last 4 Historical Weeks")
validation_chart = validation_results.set_index("ds")[["Actual", "Predicted"]]
st.line_chart(validation_chart)

v1, v2 = st.columns(2)
v1.metric("MAE", f"{mae:,.0f} units")
if np.isnan(mape):
    v2.metric("MAPE", "Not available")
else:
    v2.metric("MAPE", f"{mape:.1f}%")

st.caption(
    "Validation approach: the final 4 historical weeks are held back, predicted using earlier data, "
    "and then compared with their actual sales values."
)

# -----------------------------
# Future forecast output
# -----------------------------
st.subheader("Next 4-Week Demand Forecast")

history_tail = series.tail(52).set_index("ds")[["y"]].rename(columns={"y": "Historical"})
forecast_chart = future_forecast.set_index("ds")[["Baseline_Forecast", "Adjusted_Forecast"]]
combined_chart = pd.concat([history_tail, forecast_chart], axis=1)
st.line_chart(combined_chart)

forecast_table = future_forecast[
    ["ds", "Baseline_Forecast", "Promotion_Adjustment", "Adjusted_Forecast"]
].copy()
forecast_table["Week"] = [f"Week {i}" for i in range(1, 5)]
forecast_table = forecast_table[
    ["Week", "ds", "Baseline_Forecast", "Promotion_Adjustment", "Adjusted_Forecast"]
]
forecast_table.columns = [
    "Forecast Week",
    "Week Start",
    "Baseline Forecast",
    "Promotion Adjustment",
    "Adjusted Forecast",
]
for col in ["Baseline Forecast", "Promotion Adjustment", "Adjusted Forecast"]:
    forecast_table[col] = forecast_table[col].round(0).astype(int)

st.dataframe(forecast_table, use_container_width=True, hide_index=True)

if promotion_uplift > 0:
    st.caption(
        f"Estimated historical promotion uplift for {selected_category}: "
        f"{promotion_uplift * 100:.1f}% (based on {promo_observations} positive promoted-week observations, "
        "excluding holiday weeks). Promotion is applied only when a future promotion is selected."
    )
else:
    st.caption(
        "No reliable positive historical promotion uplift was available for this category. "
        "The baseline forecast is therefore left unchanged."
    )

# -----------------------------
# Prescriptive inventory output
# -----------------------------
st.subheader("Inventory Decision Support")

d1, d2, d3, d4 = st.columns(4)
d1.metric("Adjusted 4-week demand", f"{adjusted_total:,.0f}")
d2.metric("Current inventory", f"{current_inventory:,.0f}")
d3.metric("Safety stock", f"{safety_stock:,.0f}")
d4.metric("Recommended replenishment", f"{recommended_replenishment:,.0f}")

if stock_risk == "High":
    st.error("Stock risk: HIGH — current inventory is below the adjusted 4-week demand forecast.")
elif stock_risk == "Medium":
    st.warning("Stock risk: MEDIUM — inventory covers forecast demand but does not fully cover the safety-stock buffer.")
else:
    st.success("Stock risk: LOW — inventory covers forecast demand and the selected safety-stock buffer.")

st.caption(
    "Decision rule: Recommended Replenishment = Adjusted Forecast Demand + Safety Stock − Current Inventory. "
    "Inventory and safety-stock values are scenario inputs because they are not included in the public sales dataset."
)

with st.expander("View cleaned historical data"):
    st.dataframe(
        category_df[
            [
                "Week_Start",
                "Weekly_Sales",
                "Promo_Intensity",
                "Promotion_Flag",
                "Holiday_Event_Flag",
                "Holiday_Event_Description",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
