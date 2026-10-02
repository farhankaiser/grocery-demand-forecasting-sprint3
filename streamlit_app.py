import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Grocery Demand Forecasting Prototype",
    page_icon="📦",
    layout="wide",
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/favorita_weekly_sales.csv")
    df["Week_Start"] = pd.to_datetime(df["Week_Start"])
    df["Week_End"] = pd.to_datetime(df["Week_End"])
    return df

df = load_data()

st.title("AI-Powered Demand Forecasting & Inventory Decision Support")
st.caption("Sprint 3 prototype • Corporación Favorita • Store 44")

st.sidebar.header("Prototype Controls")

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

col1, col2, col3 = st.columns(3)
col1.metric("Selected category", selected_category)
col2.metric("Historical weeks", len(category_df))
col3.metric("Latest weekly sales", f"{category_df['Weekly_Sales'].iloc[-1]:,.0f}")

st.subheader("Historical Weekly Sales")
chart_df = category_df.set_index("Week_Start")[["Weekly_Sales"]]
st.line_chart(chart_df)

with st.expander("View the cleaned data used by the prototype"):
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
    )

st.info(
    "This is the starter version. Next we will add the 4-week forecast, "
    "forecast accuracy, promotion scenario, and replenishment recommendation."
)
