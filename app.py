"""
Excel Data Analysis Chatbot Dashboard
Built with Streamlit and FastMCP
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import base64
import io
from typing import Dict, Any, Optional
import numpy as np
from ai_agent import DataAnalysisAgent

# Configure Streamlit page
st.set_page_config(
    page_title="Excel Data Analysis Chatbot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'ai_agent' not in st.session_state:
    st.session_state.ai_agent = DataAnalysisAgent()

def load_excel_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Load and process uploaded Excel file"""
    try:
        # Read Excel file
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
            return df
        else:
            st.error("Please upload a valid Excel file (.xlsx or .xls)")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def generate_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive data summary"""
    summary = {
        "basic_info": {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict()
        },
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": {},
        "categorical_summary": {}
    }
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary["numeric_summary"] = df[numeric_cols].describe().to_dict()
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        summary["categorical_summary"][col] = {
            "unique_count": df[col].nunique(),
            "top_values": df[col].value_counts().head(5).to_dict()
        }
    
    return summary

def create_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str = None, 
                color_col: str = None, title: str = None):
    """Create different types of charts"""
    try:
        if chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "bar":
            if y_col:
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title)
            else:
                value_counts = df[x_col].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                           title=title or f"Count of {x_col}")
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x_col, color=color_col, title=title)
        elif chart_type == "box":
            fig = px.box(df, x=x_col, y=y_col, color=color_col, title=title)
        elif chart_type == "heatmap":
            numeric_data = df.select_dtypes(include=[np.number])
            if len(numeric_data.columns) < 2:
                st.error("Need at least 2 numeric columns for heatmap")
                return None
            corr_matrix = numeric_data.corr()
            fig = px.imshow(corr_matrix, text_auto=True, title=title or "Correlation Heatmap")
        else:
            st.error(f"Unsupported chart type: {chart_type}")
            return None
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def generate_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate automated insights from the data"""
    insights = {
        "dataset_overview": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_data_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        },
        "column_insights": {},
        "correlations": [],
        "outliers": {}
    }
    
    # Column-specific insights
    for col in df.columns:
        col_insights = {"type": str(df[col].dtype)}
        
        if df[col].dtype in ['int64', 'float64']:
            col_insights.update({
                "mean": df[col].mean(),
                "median": df[col].median(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "missing_count": df[col].isnull().sum()
            })
            
            # Detect outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
            insights["outliers"][col] = len(outliers)
        else:
            col_insights.update({
                "unique_values": df[col].nunique(),
                "most_common": df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                "missing_count": df[col].isnull().sum()
            })
        
        insights["column_insights"][col] = col_insights
    
    # Correlation insights for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_correlations.append({
                        "column1": corr_matrix.columns[i],
                        "column2": corr_matrix.columns[j],
                        "correlation": corr_val
                    })
        insights["correlations"] = strong_correlations
    
    return insights

def display_analysis_result(result, query_id):
    """Display the analysis result with text, visualizations, and insights"""
    # Display main response
    st.markdown(result.text_response)
    
    # Display visualizations
    if result.visualizations:
        st.subheader("📊 Supporting Visualizations")
        for i, viz in enumerate(result.visualizations):
            if viz["type"] == "plotly":
                try:
                    st.plotly_chart(viz["figure"], use_container_width=True, key=f"viz_{query_id}_{i}")
                    if viz.get("caption"):
                        st.caption(viz["caption"])
                except Exception as e:
                    st.error(f"Visualization error: {str(e)}")
                    st.info("The analysis completed successfully, but there was an issue displaying this chart.")
    
    # Display insights
    if result.insights:
        st.subheader("💡 Key Insights")
        for insight in result.insights:
            st.info(f"• {insight}")
    
    # Display recommendations
    if result.recommendations:
        st.subheader("🎯 Recommendations")
        for rec in result.recommendations:
            st.success(f"• {rec}")

# Main App Layout
st.title("📊 Excel Data Analysis Chatbot")
st.markdown("Upload your Excel file and start exploring your data with interactive visualizations and AI-powered insights!")

# Sidebar for file upload and controls
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        if st.session_state.data is None or st.button("Reload Data"):
            with st.spinner("Loading data..."):
                st.session_state.data = load_excel_file(uploaded_file)
                if st.session_state.data is not None:
                    st.session_state.insights = generate_insights(st.session_state.data)
                    # Set up AI agent with the new data
                    st.session_state.ai_agent.set_data(st.session_state.data)
                    # Clear chat history when new data is loaded
                    st.session_state.chat_history = []
                    st.success(f"Loaded {len(st.session_state.data)} rows and {len(st.session_state.data.columns)} columns")

if st.session_state.data is not None:
    df = st.session_state.data
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "💬 Chat", "📈 Visualizations", "🔍 Insights"])
    
    with tab1:
        st.header("Data Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Quick statistics
        st.subheader("Quick Statistics")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    with tab2:
        st.header("💬 AI Data Analysis Chat")
        st.markdown("Ask me anything about your data! I'll analyze it and create visualizations to support my insights.")
        
        # Suggestion pills
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("📊 Data Summary", help="Get an overview of your dataset"):
                query = "Give me a comprehensive summary of this dataset"
        with col2:
            if st.button("🔗 Find Correlations", help="Discover relationships between variables"):
                query = "What are the correlations in my data?"
        with col3:
            if st.button("📈 Show Trends", help="Analyze patterns and trends"):
                query = "Show me trends and patterns in the data"
        with col4:
            if st.button("🎯 Find Outliers", help="Detect unusual data points"):
                query = "Are there any outliers in my data?"
        
        # Chat input
        user_input = st.text_input(
            "Ask me about your data:", 
            placeholder="e.g., 'Compare sales by region', 'Show me the distribution of prices', 'What insights can you find?'",
            key="chat_input"
        )
        
        # Use query from button or user input
        if 'query' in locals():
            final_query = query
        else:
            final_query = user_input
        
        if st.button("🚀 Analyze", type="primary") and final_query:
            with st.spinner("🤖 Analyzing your data..."):
                try:
                    # Get analysis from AI agent
                    result = st.session_state.ai_agent.analyze_query(final_query)
                    
                    # Store in chat history
                    st.session_state.chat_history.append((final_query, result))
                    
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                    st.session_state.chat_history.append((final_query, f"Error: {str(e)}"))
        
        # Display chat history
        st.subheader("💬 Conversation History")
        if st.session_state.chat_history:
            for i, (query, result) in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"🧑 {query}", expanded=(i == 0)):  # Expand most recent
                    st.markdown("**Your Question:**")
                    st.write(query)
                    st.markdown("**AI Analysis:**")
                    
                    if isinstance(result, str):
                        # Handle old string responses
                        st.write(result)
                    else:
                        # Handle new AnalysisResult objects
                        display_analysis_result(result, f"chat_{len(st.session_state.chat_history)-i}")
                    
                    st.divider()
        else:
            st.info("👆 Start by asking a question about your data or use the suggestion buttons above!")
        
        # Column suggestions
        if st.session_state.data is not None:
            with st.expander("💡 Available Columns", expanded=False):
                suggestions = st.session_state.ai_agent.get_column_suggestions()
                
                if suggestions["numeric"]:
                    st.write("**📊 Numeric columns:**", ", ".join(suggestions["numeric"]))
                if suggestions["categorical"]:
                    st.write("**📝 Categorical columns:**", ", ".join(suggestions["categorical"]))
                if suggestions["datetime"]:
                    st.write("**📅 Date/Time columns:**", ", ".join(suggestions["datetime"]))
    
    with tab3:
        st.header("📈 Interactive Visualizations")
        
        # Visualization controls
        col1, col2 = st.columns(2)
        
        with col1:
            chart_type = st.selectbox("Chart Type", 
                                    ["scatter", "line", "bar", "histogram", "box", "heatmap"])
            x_column = st.selectbox("X-axis Column", df.columns)
        
        with col2:
            y_columns = [None] + list(df.columns)
            y_column = st.selectbox("Y-axis Column (optional)", y_columns)
            color_columns = [None] + list(df.columns)
            color_column = st.selectbox("Color Column (optional)", color_columns)
        
        chart_title = st.text_input("Chart Title (optional)")
        
        if st.button("Create Visualization"):
            fig = create_chart(df, chart_type, x_column, y_column, color_column, chart_title)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("🔍 Data Insights")
        
        if st.session_state.insights:
            insights = st.session_state.insights
            
            # Dataset overview
            st.subheader("Dataset Overview")
            overview = insights["dataset_overview"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", overview["total_rows"])
            with col2:
                st.metric("Total Columns", overview["total_columns"])
            with col3:
                st.metric("Missing Data %", f"{overview['missing_data_percentage']:.2f}%")
            
            # Column insights
            st.subheader("Column Analysis")
            for col, col_info in insights["column_insights"].items():
                with st.expander(f"📊 {col} ({col_info['type']})"):
                    if col_info["type"] in ['int64', 'float64']:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean", f"{col_info['mean']:.2f}")
                            st.metric("Min", f"{col_info['min']:.2f}")
                        with col2:
                            st.metric("Median", f"{col_info['median']:.2f}")
                            st.metric("Max", f"{col_info['max']:.2f}")
                        with col3:
                            st.metric("Std Dev", f"{col_info['std']:.2f}")
                            st.metric("Missing", col_info['missing_count'])
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Unique Values", col_info['unique_values'])
                            st.metric("Missing", col_info['missing_count'])
                        with col2:
                            st.write("**Most Common:**", col_info['most_common'])
            
            # Correlations
            if insights["correlations"]:
                st.subheader("Strong Correlations")
                for corr in insights["correlations"]:
                    st.write(f"• **{corr['column1']}** ↔ **{corr['column2']}**: {corr['correlation']:.3f}")
            
            # Outliers
            if insights["outliers"]:
                st.subheader("Outliers Detected")
                outlier_data = [(col, count) for col, count in insights["outliers"].items() if count > 0]
                if outlier_data:
                    for col, count in outlier_data:
                        st.write(f"• **{col}**: {count} outliers")
                else:
                    st.write("No outliers detected in numeric columns.")

else:
    # Welcome screen
    st.info("👆 Please upload an Excel file using the sidebar to get started!")
    
    # Feature showcase
    st.subheader("🚀 Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Interactive Dashboard**
        - Data overview and statistics
        - Missing value analysis
        - Column type detection
        """)
    
    with col2:
        st.markdown("""
        **💬 AI Chat Interface**
        - Natural language queries
        - Automated insights
        - Data exploration help
        """)
    
    with col3:
        st.markdown("""
        **📈 Rich Visualizations**
        - Multiple chart types
        - Interactive plots
        - Correlation heatmaps
        """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, FastMCP, and Plotly")