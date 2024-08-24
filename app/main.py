import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pandas.plotting import lag_plot

# Page configuration
st.set_page_config(page_title="EDA Dashboard Apps", layout="wide")

# Main app function
def main():
    st.title("Exploratory Data Analysis (EDA) App")

    # Upload a dataset
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Drop the "Comments" column if it exists and is completely null
        if 'Comments' in df.columns and df['Comments'].isnull().all():
            df = df.drop(columns=['Comments'])

        # Display the dataset
        st.subheader("Dataset Overview")
        st.dataframe(df.head())

        # Show summary statistics
        st.subheader("Summary Statistics")
        st.write(df.describe())
        
    else:
        st.warning("Please upload a CSV file to proceed.")
        return
    
    # Sidebar for user input
    st.sidebar.header("User Input Features")
    
    # Select visualization type
    viz_type = st.sidebar.selectbox("Select Visualization Type", ["Line Plot", "Bar Plot", "Pie Chart", "Heatmap", 
                                                                  "Pair Plot", "Scatter Plot", "Histogram", "Box Plot", 
                                                                  "KDE Plot", "Density Plot", "Radial Bar Chart", 
                                                                  "Time Series Heatmap by Month", "Time Series Heatmap by Date", 
                                                                  "Heatmap with Annotations", "Violin Plot", "Lag Plot", 
                                                                  "Bubble Chart", "Polar Plot", "Correlation Analysis", 
                                                                  "Sensor Cleaning Impact", "Time Series Analysis"])
    
    # Select columns for visualization
    selected_columns = st.sidebar.multiselect("Select Columns for Visualization", df.columns[1:], default=df.columns[1])
    
    # Filter data by date range
    st.sidebar.subheader("Filter by Date Range")
    min_date = st.sidebar.date_input("Start date", pd.to_datetime(df['Timestamp']).min())
    max_date = st.sidebar.date_input("End date", pd.to_datetime(df['Timestamp']).max())
    
    if min_date > max_date:
        st.sidebar.error("Error: End date must be after start date.")
    else:
        df = df[(pd.to_datetime(df['Timestamp']) >= pd.Timestamp(min_date)) & (pd.to_datetime(df['Timestamp']) <= pd.Timestamp(max_date))]

    # Visualize data based on user input
    st.subheader("Visualization")
    
    # Univariate Analysis Plots
    if viz_type == "Histogram":
        if selected_columns:
            for col in selected_columns:
                fig = px.histogram(df, x=col)
                st.plotly_chart(fig)
        else:
            st.error("Please select at least one column for the histogram.")
    
    elif viz_type == "Box Plot":
        if selected_columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df[selected_columns], ax=ax)
            st.pyplot(fig)
        else:
            st.error("Please select at least one column for the box plot.")
    
    elif viz_type == "KDE Plot":
        if selected_columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            for col in selected_columns:
                sns.kdeplot(df[col], ax=ax, label=col)
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("Please select at least one column for the KDE plot.")
    
    elif viz_type == "Density Plot":
        if selected_columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.kdeplot(df[selected_columns[0]], fill=True, ax=ax)
            st.pyplot(fig)
        else:
            st.error("Please select at least one column for the density plot.")
    
    elif viz_type == "Radial Bar Chart":
        if selected_columns:
            for col in selected_columns:
                fig = px.bar_polar(df, r=col, theta='Timestamp', color=col)
                st.plotly_chart(fig)
        else:
            st.error("Please select at least one column for the radial bar chart.")
    
    # Bivariate Analysis Plots
    elif viz_type == "Scatter Plot":
        if len(selected_columns) == 2:
            fig = px.scatter(df, x=selected_columns[0], y=selected_columns[1])
            st.plotly_chart(fig)
        else:
            st.error("Please select exactly two columns for the scatter plot.")
    
    elif viz_type == "Bubble Chart":
        if len(selected_columns) >= 2:
            size_col = st.sidebar.selectbox("Select column for bubble size", df.columns[1:], index=2)
            fig = px.scatter(df, x=selected_columns[0], y=selected_columns[1], size=size_col, color=selected_columns[1])
            st.plotly_chart(fig)
        else:
            st.error("Please select at least two columns for the bubble chart.")
    
    elif viz_type == "Lag Plot":
        if selected_columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            lag_plot(df[selected_columns[0]], ax=ax)
            st.pyplot(fig)
        else:
            st.error("Please select a column for the lag plot.")
    
    elif viz_type == "Violin Plot":
        if selected_columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(data=df[selected_columns], ax=ax)
            st.pyplot(fig)
        else:
            st.error("Please select at least one column for the violin plot.")
    
    elif viz_type == "Polar Plot":
        if selected_columns:
            fig = px.line_polar(df, r=selected_columns[0], theta='Timestamp', line_close=True)
            st.plotly_chart(fig)
        else:
            st.error("Please select at least one column for the polar plot.")
    
    elif viz_type == "Correlation Analysis":
        # Select only numeric columns for correlation analysis
        numeric_df = df.select_dtypes(include=[np.number])
        
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.error("No numeric columns available for correlation analysis.")

    # Multivariate Analysis Plots
    elif viz_type == "Pair Plot":
        if len(selected_columns) > 1:
            fig = sns.pairplot(df[selected_columns])
            st.pyplot(fig)
        else:
            st.error("Please select at least two columns for the pair plot.")
    
    elif viz_type == "Heatmap":
        if len(selected_columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[selected_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.error("Please select at least two columns for the heatmap.")
    
    elif viz_type == "Heatmap with Annotations":
        if len(selected_columns) > 1:
            fig = px.imshow(df[selected_columns].corr(), text_auto=True)
            st.plotly_chart(fig)
        else:
            st.error("Please select at least two columns for the heatmap with annotations.")
    
    elif viz_type == "Time Series Heatmap by Month":
        df['Month'] = pd.to_datetime(df['Timestamp']).dt.to_period('M')
        if selected_columns:
            monthly_avg = df.groupby('Month')[selected_columns].mean()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(monthly_avg, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.error("Please select at least one column for the time series heatmap by month.")
    
    elif viz_type == "Time Series Heatmap by Date":
        df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
        if selected_columns:
            daily_avg = df.groupby('Date')[selected_columns].mean()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(daily_avg, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.error("Please select at least one column for the time series heatmap by date.")
    
    elif viz_type == "Sensor Cleaning Impact":
        if 'Cleaning' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Cleaning', y=selected_columns[0], data=df, ax=ax)
            st.pyplot(fig)
        else:
            st.error("The dataset does not contain a 'Cleaning' column.")
    
    elif viz_type == "Time Series Analysis":
        if selected_columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            for col in selected_columns:
                ax.plot(df['Timestamp'], df[col], label=col)
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("Please select at least one column for the time series analysis.")

if __name__ == "__main__":
    main()
