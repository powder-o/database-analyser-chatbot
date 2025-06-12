"""
MCP Tools for Excel Data Analysis
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fastmcp import FastMCP
from typing import Dict, List, Any, Optional
import io
import base64

# Initialize FastMCP server
mcp = FastMCP("Excel Analysis Tools")

# Global variable to store the current dataset
current_data: Optional[pd.DataFrame] = None

@mcp.tool()
def load_excel_data(file_content: str, sheet_name: str = None) -> Dict[str, Any]:
    """
    Load Excel data from base64 encoded file content.
    
    Args:
        file_content: Base64 encoded Excel file content
        sheet_name: Optional sheet name to load (default: first sheet)
    
    Returns:
        Dictionary with success status, data info, and column names
    """
    global current_data
    
    try:
        # Decode base64 content
        file_bytes = base64.b64decode(file_content)
        
        # Load Excel file
        if sheet_name:
            current_data = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
        else:
            current_data = pd.read_excel(io.BytesIO(file_bytes))
        
        return {
            "success": True,
            "message": f"Successfully loaded {len(current_data)} rows and {len(current_data.columns)} columns",
            "columns": list(current_data.columns),
            "shape": current_data.shape,
            "dtypes": current_data.dtypes.to_dict()
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error loading Excel file: {str(e)}"
        }

@mcp.tool()
def get_data_summary() -> Dict[str, Any]:
    """
    Get a comprehensive summary of the current dataset.
    
    Returns:
        Dictionary with dataset summary statistics and info
    """
    global current_data
    
    if current_data is None:
        return {"success": False, "message": "No data loaded. Please load an Excel file first."}
    
    try:
        summary = {
            "success": True,
            "basic_info": {
                "shape": current_data.shape,
                "columns": list(current_data.columns),
                "dtypes": current_data.dtypes.to_dict()
            },
            "missing_values": current_data.isnull().sum().to_dict(),
            "numeric_summary": current_data.describe().to_dict() if len(current_data.select_dtypes(include=[np.number]).columns) > 0 else {},
            "categorical_summary": {}
        }
        
        # Add categorical column summaries
        categorical_cols = current_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            summary["categorical_summary"][col] = {
                "unique_count": current_data[col].nunique(),
                "top_values": current_data[col].value_counts().head(5).to_dict()
            }
        
        return summary
    except Exception as e:
        return {"success": False, "message": f"Error generating summary: {str(e)}"}

@mcp.tool()
def create_visualization(chart_type: str, x_column: str, y_column: str = None, 
                        color_column: str = None, title: str = None) -> Dict[str, Any]:
    """
    Create a visualization of the data.
    
    Args:
        chart_type: Type of chart (scatter, line, bar, histogram, box, heatmap)
        x_column: Column name for x-axis
        y_column: Column name for y-axis (optional for some chart types)
        color_column: Column name for color coding (optional)
        title: Chart title (optional)
    
    Returns:
        Dictionary with success status and chart data
    """
    global current_data
    
    if current_data is None:
        return {"success": False, "message": "No data loaded. Please load an Excel file first."}
    
    if x_column not in current_data.columns:
        return {"success": False, "message": f"Column '{x_column}' not found in dataset"}
    
    try:
        fig = None
        
        if chart_type == "scatter":
            if not y_column:
                return {"success": False, "message": "y_column required for scatter plot"}
            fig = px.scatter(current_data, x=x_column, y=y_column, color=color_column, title=title)
        
        elif chart_type == "line":
            if not y_column:
                return {"success": False, "message": "y_column required for line plot"}
            fig = px.line(current_data, x=x_column, y=y_column, color=color_column, title=title)
        
        elif chart_type == "bar":
            if y_column:
                fig = px.bar(current_data, x=x_column, y=y_column, color=color_column, title=title)
            else:
                # Count plot
                value_counts = current_data[x_column].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values, title=title or f"Count of {x_column}")
        
        elif chart_type == "histogram":
            fig = px.histogram(current_data, x=x_column, color=color_column, title=title)
        
        elif chart_type == "box":
            fig = px.box(current_data, x=x_column, y=y_column, color=color_column, title=title)
        
        elif chart_type == "heatmap":
            # Create correlation heatmap for numeric columns
            numeric_data = current_data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) < 2:
                return {"success": False, "message": "Need at least 2 numeric columns for heatmap"}
            corr_matrix = numeric_data.corr()
            fig = px.imshow(corr_matrix, text_auto=True, title=title or "Correlation Heatmap")
        
        else:
            return {"success": False, "message": f"Unsupported chart type: {chart_type}"}
        
        # Convert to JSON for transmission
        chart_json = fig.to_json()
        
        return {
            "success": True,
            "message": f"Successfully created {chart_type} chart",
            "chart_data": chart_json
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error creating visualization: {str(e)}"}

@mcp.tool()
def filter_data(column: str, operation: str, value: Any) -> Dict[str, Any]:
    """
    Filter the current dataset based on specified conditions.
    
    Args:
        column: Column name to filter on
        operation: Filter operation (eq, ne, gt, lt, gte, lte, contains, isin)
        value: Value to compare against
    
    Returns:
        Dictionary with filtered data info
    """
    global current_data
    
    if current_data is None:
        return {"success": False, "message": "No data loaded. Please load an Excel file first."}
    
    if column not in current_data.columns:
        return {"success": False, "message": f"Column '{column}' not found in dataset"}
    
    try:
        original_count = len(current_data)
        
        if operation == "eq":
            current_data = current_data[current_data[column] == value]
        elif operation == "ne":
            current_data = current_data[current_data[column] != value]
        elif operation == "gt":
            current_data = current_data[current_data[column] > value]
        elif operation == "lt":
            current_data = current_data[current_data[column] < value]
        elif operation == "gte":
            current_data = current_data[current_data[column] >= value]
        elif operation == "lte":
            current_data = current_data[current_data[column] <= value]
        elif operation == "contains":
            current_data = current_data[current_data[column].str.contains(str(value), na=False)]
        elif operation == "isin":
            if not isinstance(value, list):
                value = [value]
            current_data = current_data[current_data[column].isin(value)]
        else:
            return {"success": False, "message": f"Unsupported operation: {operation}"}
        
        filtered_count = len(current_data)
        
        return {
            "success": True,
            "message": f"Filtered data from {original_count} to {filtered_count} rows",
            "original_count": original_count,
            "filtered_count": filtered_count,
            "shape": current_data.shape
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error filtering data: {str(e)}"}

@mcp.tool()
def get_insights() -> Dict[str, Any]:
    """
    Generate automated insights from the current dataset.
    
    Returns:
        Dictionary with various insights about the data
    """
    global current_data
    
    if current_data is None:
        return {"success": False, "message": "No data loaded. Please load an Excel file first."}
    
    try:
        insights = {
            "success": True,
            "dataset_overview": {
                "total_rows": len(current_data),
                "total_columns": len(current_data.columns),
                "missing_data_percentage": (current_data.isnull().sum().sum() / (len(current_data) * len(current_data.columns))) * 100
            },
            "column_insights": {},
            "correlations": {},
            "outliers": {}
        }
        
        # Column-specific insights
        for col in current_data.columns:
            col_insights = {"type": str(current_data[col].dtype)}
            
            if current_data[col].dtype in ['int64', 'float64']:
                col_insights.update({
                    "mean": current_data[col].mean(),
                    "median": current_data[col].median(),
                    "std": current_data[col].std(),
                    "min": current_data[col].min(),
                    "max": current_data[col].max(),
                    "missing_count": current_data[col].isnull().sum()
                })
                
                # Detect outliers using IQR method
                Q1 = current_data[col].quantile(0.25)
                Q3 = current_data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = current_data[(current_data[col] < (Q1 - 1.5 * IQR)) | 
                                      (current_data[col] > (Q3 + 1.5 * IQR))]
                insights["outliers"][col] = len(outliers)
                
            else:
                col_insights.update({
                    "unique_values": current_data[col].nunique(),
                    "most_common": current_data[col].mode().iloc[0] if len(current_data[col].mode()) > 0 else None,
                    "missing_count": current_data[col].isnull().sum()
                })
            
            insights["column_insights"][col] = col_insights
        
        # Correlation insights for numeric columns
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = current_data[numeric_cols].corr()
            # Find strongest correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # Strong correlation threshold
                        strong_correlations.append({
                            "column1": corr_matrix.columns[i],
                            "column2": corr_matrix.columns[j],
                            "correlation": corr_val
                        })
            insights["correlations"] = strong_correlations
        
        return insights
        
    except Exception as e:
        return {"success": False, "message": f"Error generating insights: {str(e)}"}

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()