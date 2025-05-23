import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sklearn.metrics import mean_absolute_error

#  Use full page width
st.set_page_config(layout="wide")

# 🗂 Available CSV files and display names
files = {
    "Apple (AAPL)": "df_AAPL.csv",
    "Anheuser-Busch (ABNB)": "df_ABNB.csv",
    "Google (GOOG)": "df_GOOG.csv",
    "Amazon (AMZN)": "df_AMZN.csv",
    "Boeing (BA)": "df_BA.csv",
    "American Tower (AMT)": "df_AMT.csv"
}

# ℹ Company descriptions
descriptions = {
    "Apple (AAPL)": "Apple Inc. is a global technology company known for the iPhone, Mac, and innovative software services.",
    "Anheuser-Busch (ABNB)": "Airbnb, Inc. operates an online marketplace for lodging, primarily homestays for vacation rentals.",
    "Google (GOOG)": "Google LLC is a leading tech company specializing in internet-related services, AI, and cloud computing.",
    "Amazon (AMZN)": "Amazon.com, Inc. is a global e-commerce and cloud computing giant headquartered in Seattle.",
    "Boeing (BA)": "The Boeing Company designs, manufactures, and sells airplanes, rotorcraft, rockets, and satellites worldwide.",
    "American Tower (AMT)": "American Tower Corporation is a real estate investment trust that owns and operates wireless towers."
}

#  Load selected CSV file
data_folder = os.path.dirname(__file__)
selected_company = st.sidebar.selectbox(
    "📈 Select a company", list(files.keys()))
file_path = os.path.join(data_folder, files[selected_company])
df = pd.read_csv(file_path)

#  Normalize column names
df.columns = [col.strip().lower() for col in df.columns]

#  Check and prepare data
if "date" in df.columns and "close" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Display logo and title
    symbol = files[selected_company].split('_')[1].split('.')[0]  # e.g., AAPL
    logo_path = os.path.join(data_folder, "logos", f"{symbol}.png")

    header_col1, header_col2 = st.columns([1, 8])
    with header_col1:
        st.image(logo_path, width=150)
    with header_col2:
        st.markdown(
            f"<h1 style='margin-bottom: 0;'>Stock Price Dashboard – {selected_company}</h1>",
            unsafe_allow_html=True)
        st.caption("Interactive Analysis with Forecast and Comparison")

    #  Date range filter
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.slider("Select a date range", min_value=min_date, max_value=max_date,
                           value=(min_date, max_date))
    df_filtered = df[(df["date"] >= pd.to_datetime(date_range[0])) &
                     (df["date"] <= pd.to_datetime(date_range[1]))]

    #  Tabs including forecast
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Line Chart", "📊 Bar Chart", "🔀 Compare", "📉 Forecast"])

    with tab1:
        st.subheader("Interactive Line Chart")
        fig = px.line(df_filtered, x="date", y="close",
                      title="Closing Price Over Time")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Interactive Bar Chart")
        fig = px.bar(df_filtered, x="date", y="close",
                     title="Closing Price Bar Chart")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Compare with Other Companies")
        others = st.multiselect("Select companies to compare", [
                                k for k in files.keys() if k != selected_company])
        fig = px.line(df_filtered, x="date", y="close",
                      labels={"close": selected_company})
        for other in others:
            other_path = os.path.join(data_folder, files[other])
            other_df = pd.read_csv(other_path)
            other_df.columns = [col.strip().lower()
                                for col in other_df.columns]
            if "date" in other_df.columns and "close" in other_df.columns:
                other_df["date"] = pd.to_datetime(other_df["date"])
                other_df = other_df.sort_values("date")
                filtered = other_df[(other_df["date"] >= pd.to_datetime(date_range[0])) &
                                    (other_df["date"] <= pd.to_datetime(date_range[1]))]
                fig.add_scatter(
                    x=filtered["date"], y=filtered["close"], mode="lines", name=other)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Forecasted vs Real Prices (RNN Model)")
        forecast_path = os.path.join(data_folder, f"forecast_{symbol}.csv")

        if os.path.exists(forecast_path):
            forecast_df = pd.read_csv(forecast_path)
            forecast_df.columns = [col.strip().lower()
                                   for col in forecast_df.columns]
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])

            expected_cols = {"real_close", "predicted_close", "date"}
            if expected_cols.issubset(forecast_df.columns):
                forecast_long = forecast_df.melt(
                    id_vars="date",
                    value_vars=["real_close", "predicted_close"],
                    var_name="Type", value_name="Price"
                )

                fig = px.line(forecast_long, x="date", y="Price", color="Type",
                              labels={"date": "Date",
                                      "Price": "Price", "Type": "Legend"},
                              title="Real vs Predicted Closing Prices")
                fig.update_traces(mode="lines")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Forecast Table:**")
                st.dataframe(forecast_df)

                mae = mean_absolute_error(
                    forecast_df["real_close"], forecast_df["predicted_close"])
                st.info(f" **Mean Absolute Error (MAE):** {mae:.4f}")

                st.caption(
                    "Forecast generated by an RNN model trained on past 7-day windows of closing prices.")
            else:
                st.error(
                    " Forecast file is missing required columns: 'real_close' or 'predicted_close'.")
                st.write("Available columns:", list(forecast_df.columns))
        else:
            st.warning(" Forecast file not found for this company.")

    #  Dynamic description & Tufts principles (separado dos expanders)
    with st.expander(" About this Dashboard"):
        total_close = df_filtered["close"].sum()
        formatted_total = f"${total_close:,.2f}"

        st.markdown(f"###  About {selected_company.split('(')[0].strip()}")
        st.write(descriptions[selected_company])
        st.info(
            f"**Total closing value in selected period:** {formatted_total}")

    # NÃO aninhar expanders
    st.markdown("### 📚 Discussion – Tufts Principles in this Dashboard")
    st.markdown("""
        This dashboard follows **Edward Tufte’s principles** of effective data visualization:

        - **Show the data** clearly through line and bar charts with minimal clutter.
        - **Avoid distortion** by maintaining proportional scale and true representations.
        - **Present many numbers in a small space** with interactive tabs and filters.
        - **Encourage data comparison** across companies using multi-select features.
        - **Reveal data at several levels of detail**, from overviews to forecasted trends.
        - **Integrate text and graphics** by providing context and company info directly with visuals.
    """)

else:
    st.error(" Date not found.")
