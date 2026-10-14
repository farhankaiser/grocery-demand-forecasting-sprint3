import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

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
def create_rf_features(data):
    rf_data = data.sort_values("Week_Start").copy()

    rf_data["lag_1"] = rf_data["Weekly_Sales"].shift(1)
    rf_data["lag_2"] = rf_data["Weekly_Sales"].shift(2)
    rf_data["lag_4"] = rf_data["Weekly_Sales"].shift(4)
    rf_data["lag_52"] = rf_data["Weekly_Sales"].shift(52)

    rf_data["rolling_4"] = (
        rf_data["Weekly_Sales"]
        .shift(1)
        .rolling(window=4)
        .mean()
    )

    rf_data["week_of_year"] = (
        rf_data["Week_Start"].dt.isocalendar().week.astype(int)
    )
    rf_data["month"] = rf_data["Week_Start"].dt.month

    return rf_data

def validate_random_forest(category_df, validation_weeks=4):
    data = category_df.sort_values("Week_Start").copy()

    train_data = data.iloc[:-validation_weeks].copy()
    test_data = data.iloc[-validation_weeks:].copy()

    rf_train = create_rf_features(train_data).dropna().copy()

    feature_columns = [
        "lag_1",
        "lag_2",
        "lag_4",
        "lag_52",
        "rolling_4",
        "week_of_year",
        "month",
    ]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        rf_train[feature_columns],
        rf_train["Weekly_Sales"]
    )

    history = train_data["Weekly_Sales"].tolist()
    predictions = []

    for _, row in test_data.iterrows():
        target_date = row["Week_Start"]

        features = pd.DataFrame([{
            "lag_1": history[-1],
            "lag_2": history[-2],
            "lag_4": history[-4],
            "lag_52": history[-52],
            "rolling_4": np.mean(history[-4:]),
            "week_of_year": int(target_date.isocalendar().week),
            "month": target_date.month,
        }])

        prediction = model.predict(features)[0]
        prediction = max(0, prediction)

        predictions.append(prediction)

        # Use the predicted value for the next forecast step
        history.append(prediction)

    actual = test_data["Weekly_Sales"].values

    mae = mean_absolute_error(actual, predictions)
    mape = mean_absolute_percentage_error(actual, predictions) * 100

    results = pd.DataFrame({
        "Week_Start": test_data["Week_Start"].values,
        "Actual": actual,
        "Random_Forest": predictions,
    })

    return results, mae, mape
def forecast_random_forest(category_df, periods=4):
    data = category_df.sort_values("Week_Start").copy()

    rf_data = create_rf_features(data).dropna().copy()

    feature_columns = [
        "lag_1",
        "lag_2",
        "lag_4",
        "lag_52",
        "rolling_4",
        "week_of_year",
        "month",
    ]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        rf_data[feature_columns],
        rf_data["Weekly_Sales"]
    )

    history = data["Weekly_Sales"].tolist()
    last_date = data["Week_Start"].max()

    forecasts = []

    for step in range(1, periods + 1):
        target_date = last_date + pd.Timedelta(weeks=step)

        features = pd.DataFrame([{
            "lag_1": history[-1],
            "lag_2": history[-2],
            "lag_4": history[-4],
            "lag_52": history[-52],
            "rolling_4": np.mean(history[-4:]),
            "week_of_year": int(target_date.isocalendar().week),
            "month": target_date.month,
        }])

        prediction = model.predict(features)[0]
        prediction = max(0, prediction)

        forecasts.append({
            "ds": target_date,
            "yhat": prediction,
        })

        history.append(prediction)

    return pd.DataFrame(forecasts)
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
rf_validation_results, rf_mae, rf_mape = validate_random_forest(
category_df,
validation_weeks=4
)

# -----------------------------
# 2) Future four-week forecast
# -----------------------------
if not np.isnan(mape) and not np.isnan(rf_mape):
    selected_forecast_model = (
        "Random Forest" if rf_mape < mape else "Prophet"
    )
else:
    selected_forecast_model = (
        "Random Forest" if rf_mae < mae else "Prophet"
    )

if selected_forecast_model == "Random Forest":
    future_forecast = forecast_random_forest(
        category_df,
        periods=4
    )
else:
    final_model = build_model()
    final_model.fit(series)

    future_dates = final_model.make_future_dataframe(
        periods=4,
        freq="W-MON",
        include_history=False,
    )

    future_forecast = final_model.predict(future_dates)[
        ["ds", "yhat"]
    ].copy()

future_forecast["yhat"] = future_forecast["yhat"].clip(lower=0)

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

# ------------------comparison_chart =-----------
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
st.subheader("Forecast Model Accuracy Comparison")

accuracy_comparison = pd.DataFrame({
    "Model": ["Prophet", "Random Forest"],
    "MAPE (%)": [mape, rf_mape]
}).set_index("Model")

acc1, acc2 = st.columns(2)

acc1.metric("Prophet MAPE", f"{mape:.1f}%")
acc2.metric("Random Forest MAPE", f"{rf_mape:.1f}%")
st.bar_chart(accuracy_comparison)

st.caption(
    "Lower MAPE indicates lower forecasting error and better validation performance."
)

st.success(
    f"Selected model for the next 4-week forecast: {selected_forecast_model}"
)


st.subheader("Historical Weekly Sales")
historical_chart = category_df.set_index("Week_Start")[["Weekly_Sales"]]
st.line_chart(historical_chart)

# -----------------------------
# Forecast validation output
# -----------------------------
st.subheader("Forecast Model Comparison — Last 4 Historical Weeks")

st.markdown("#### Prophet Validation")

prophet_chart = (
validation_results[["ds", "Actual", "Predicted"]]
.rename(columns={"Predicted": "Prophet"})
.set_index("ds")[["Actual", "Prophet"]]
)

st.line_chart(prophet_chart)

st.markdown("#### Random Forest Validation")

random_forest_chart = (
rf_validation_results[["Week_Start", "Actual", "Random_Forest"]]
.set_index("Week_Start")[["Actual", "Random_Forest"]]
)

st.line_chart(random_forest_chart)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Prophet MAE", f"{mae:,.0f} units")

if np.isnan(mape):
    m2.metric("Prophet MAPE", "Not available")
else:
    m2.metric("Prophet MAPE", f"{mape:.1f}%")

m3.metric("Random Forest MAE", f"{rf_mae:,.0f} units")

if np.isnan(rf_mape):
    m4.metric("Random Forest MAPE", "Not available")
else:
    m4.metric("Random Forest MAPE", f"{rf_mape:.1f}%")

if not np.isnan(mape) and not np.isnan(rf_mape):
    better_model = "Random Forest" if rf_mape < mape else "Prophet"
else:
    better_model = "Random Forest" if rf_mae < mae else "Prophet"

st.success(
    f"Better validation performance: {better_model}. "
    "Lower MAE and MAPE indicate lower forecasting error."
)

st.caption(
    "Both Prophet and Random Forest are tested on the same final 4 historical weeks "
    "and compared against the actual recorded sales."
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
