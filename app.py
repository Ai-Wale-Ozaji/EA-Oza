import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="HR Attrition Analytics Dashboard", layout="wide")
st.title("HR Attrition Analytics Dashboard")
st.caption("Your go-to insights platform for employee attrition analysis")
st.markdown("---")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("EA.csv")
    if "ID" in df.columns:
        df = df.drop("ID", axis=1)
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filter the data")
departments = st.sidebar.multiselect("Department", sorted(df["Department"].unique()), default=list(df["Department"].unique()))
genders = st.sidebar.multiselect("Gender", sorted(df["Gender"].unique()), default=list(df["Gender"].unique()))
ages = st.sidebar.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
years = st.sidebar.slider("Years at Company", int(df["YearsAtCompany"].min()), int(df["YearsAtCompany"].max()), (int(df["YearsAtCompany"].min()), int(df["YearsAtCompany"].max())))

filtered_df = df[
    (df["Department"].isin(departments)) &
    (df["Gender"].isin(genders)) &
    (df["Age"].between(ages[0], ages[1])) &
    (df["YearsAtCompany"].between(years[0], years[1]))
]

# Tabs for Macro & Micro Analysis
tab1, tab2, tab3 = st.tabs(["Macro Analysis", "Micro Analysis", "Data & Download"])

with tab1:
    st.subheader("Macro Trends & Overview")

    st.write("**Attrition Rate**: The overall percentage of employees who left the organization.")
    attrition_rate = filtered_df["Attrition"].value_counts(normalize=True).get(1, 0) * 100
    st.metric("Overall Attrition Rate (%)", f"{attrition_rate:.2f}")

    st.write("**1. Attrition Distribution**: Number of employees who left vs. stayed.")
    fig1 = px.histogram(filtered_df, x="Attrition", color="Attrition", nbins=2,
                       labels={"Attrition": "Attrition (0=No, 1=Yes)"},
                       color_discrete_map={0:"green", 1:"red"})
    st.plotly_chart(fig1, use_container_width=True)

    st.write("**2. Attrition by Department**: Shows which department has highest attrition.")
    fig2 = px.histogram(filtered_df, x="Department", color="Attrition", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

    st.write("**3. Attrition by Gender**: Analyses gender-wise attrition patterns.")
    fig3 = px.histogram(filtered_df, x="Gender", color="Attrition", barmode="group")
    st.plotly_chart(fig3, use_container_width=True)

    st.write("**4. Attrition by Age Group**: Are certain age groups more likely to leave?")
    fig4 = px.histogram(filtered_df, x="Age", color="Attrition", nbins=20)
    st.plotly_chart(fig4, use_container_width=True)

    st.write("**5. Attrition by Years at Company**: Relationship between experience and attrition.")
    fig5 = px.box(filtered_df, x="Attrition", y="YearsAtCompany", color="Attrition")
    st.plotly_chart(fig5, use_container_width=True)

    st.write("**6. Attrition by Job Role**: Which roles are most impacted?")
    fig6 = px.histogram(filtered_df, x="JobRole", color="Attrition", barmode="group")
    st.plotly_chart(fig6, use_container_width=True)

    st.write("**7. Attrition by Education Level**")
    fig7 = px.histogram(filtered_df, x="Education", color="Attrition", barmode="group")
    st.plotly_chart(fig7, use_container_width=True)

    st.write("**8. Attrition by Marital Status**")
    fig8 = px.histogram(filtered_df, x="MaritalStatus", color="Attrition", barmode="group")
    st.plotly_chart(fig8, use_container_width=True)

    st.write("**9. Monthly Income Distribution**")
    fig9 = px.box(filtered_df, x="Attrition", y="MonthlyIncome", color="Attrition")
    st.plotly_chart(fig9, use_container_width=True)

    st.write("**10. Attrition by OverTime**")
    fig10 = px.histogram(filtered_df, x="OverTime", color="Attrition", barmode="group")
    st.plotly_chart(fig10, use_container_width=True)

with tab2:
    st.subheader("Micro-Level Insights & Correlations")

    st.write("**11. Age vs. Monthly Income**: Explore relationship between age and salary.")
    fig11 = px.scatter(filtered_df, x="Age", y="MonthlyIncome", color="Attrition", trendline="ols")
    st.plotly_chart(fig11, use_container_width=True)

    st.write("**12. Years at Company vs. Attrition**")
    fig12 = px.violin(filtered_df, y="YearsAtCompany", x="Attrition", color="Attrition", box=True)
    st.plotly_chart(fig12, use_container_width=True)

    st.write("**13. Job Satisfaction Distribution**")
    fig13 = px.histogram(filtered_df, x="JobSatisfaction", color="Attrition", barmode="group")
    st.plotly_chart(fig13, use_container_width=True)

    st.write("**14. Environment Satisfaction**")
    fig14 = px.histogram(filtered_df, x="EnvironmentSatisfaction", color="Attrition", barmode="group")
    st.plotly_chart(fig14, use_container_width=True)

    st.write("**15. Number of Trainings Attended**")
    fig15 = px.histogram(filtered_df, x="NumCompaniesWorked", color="Attrition", barmode="group")
    st.plotly_chart(fig15, use_container_width=True)

    st.write("**16. Years Since Last Promotion**")
    fig16 = px.box(filtered_df, x="Attrition", y="YearsSinceLastPromotion", color="Attrition")
    st.plotly_chart(fig16, use_container_width=True)

    st.write("**17. Years in Current Role vs. Attrition**")
    fig17 = px.box(filtered_df, x="Attrition", y="YearsInCurrentRole", color="Attrition")
    st.plotly_chart(fig17, use_container_width=True)

    st.write("**18. Heatmap: Correlation Between Features**")
    import seaborn as sns
    import matplotlib.pyplot as plt
    corr = filtered_df.select_dtypes(include=np.number).corr()
    fig18, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig18)

    st.write("**19. Cross Tab: Attrition vs Marital Status**")
    cross_tab = pd.crosstab(filtered_df["Attrition"], filtered_df["MaritalStatus"], normalize='index')
    st.dataframe(cross_tab)

    st.write("**20. Pivot Table: Average Income by Role & Department**")
    pivot = filtered_df.pivot_table(index="JobRole", columns="Department", values="MonthlyIncome", aggfunc="mean")
    st.dataframe(pivot)

    st.write("**21. Download Filtered Data**")
    st.download_button("Download Current Filtered Data as CSV", filtered_df.to_csv(index=False), file_name="filtered_EA.csv")

with tab3:
    st.subheader("Explore Data Directly")
    st.write("**Complete dataset is shown below with applied filters:**")
    st.dataframe(filtered_df, use_container_width=True)

    st.write("**Summary Statistics**")
    st.dataframe(filtered_df.describe())

    st.write("**Column-wise Null Value Count**")
    st.write(filtered_df.isnull().sum())

# Footer
st.markdown("---")
st.caption("Dashboard created by Dhiren Oza | Powered by Streamlit, Plotly, Pandas, Seaborn")
