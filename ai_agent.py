"""
AI Agent for Excel Data Analysis
Provides intelligent analysis and visualization generation
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import json
import re
from dataclasses import dataclass
from enum import Enum

class AnalysisType(Enum):
    SUMMARY = "summary"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    COMPARISON = "comparison"
    TREND = "trend"
    OUTLIER = "outlier"
    MISSING_DATA = "missing_data"
    CATEGORICAL = "categorical"

@dataclass
class AnalysisResult:
    text_response: str
    visualizations: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[str]

class DataAnalysisAgent:
    def __init__(self):
        self.data = None
        self.context = []
        
    def set_data(self, df: pd.DataFrame):
        """Set the dataset for analysis"""
        self.data = df.copy()
    
    def _clean_data_for_plotting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data to avoid JSON serialization issues with Plotly"""
        cleaned_df = df.copy()
        
        # Convert problematic dtypes
        for col in cleaned_df.columns:
            dtype_str = str(cleaned_df[col].dtype)
            
            # Handle object dtypes (includes strings and mixed types)
            if cleaned_df[col].dtype == 'object' or 'object' in dtype_str:
                # Handle NaN values and convert to string
                cleaned_df[col] = cleaned_df[col].fillna('Missing').astype(str)
            
            # Handle datetime columns
            elif 'datetime' in dtype_str:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
            
            # Handle problematic pandas extension dtypes
            elif 'Int64' in dtype_str or 'Float64' in dtype_str or 'string' in dtype_str:
                if cleaned_df[col].dtype.name.startswith(('Int', 'Float')):
                    # Convert to standard numpy dtypes
                    cleaned_df[col] = cleaned_df[col].astype('float64')
                else:
                    cleaned_df[col] = cleaned_df[col].astype(str)
            
            # Ensure standard numeric columns are proper types
            elif cleaned_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        return cleaned_df
    
    def _safe_plotly_figure(self, plot_func, *args, **kwargs):
        """Safely create plotly figure with data cleaning"""
        try:
            # Clean the data before plotting
            if 'data_frame' in kwargs:
                kwargs['data_frame'] = self._clean_data_for_plotting(kwargs['data_frame'])
            elif len(args) > 0 and isinstance(args[0], pd.DataFrame):
                args = (self._clean_data_for_plotting(args[0]),) + args[1:]
            
            return plot_func(*args, **kwargs)
        except Exception as e:
            # Fallback: create a simple error plot
            fig = go.Figure()
            fig.add_annotation(text=f"Visualization Error: {str(e)}", 
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
    def analyze_query(self, query: str) -> AnalysisResult:
        """Main method to analyze user query and generate response with visualizations"""
        if self.data is None:
            return AnalysisResult(
                text_response="Please upload a dataset first to start the analysis.",
                visualizations=[],
                insights=[],
                recommendations=[]
            )
        
        # Determine analysis type based on query
        analysis_type = self._classify_query(query)
        
        # Store context
        self.context.append(query)
        
        # Perform analysis based on type
        if analysis_type == AnalysisType.SUMMARY:
            return self._analyze_summary(query)
        elif analysis_type == AnalysisType.CORRELATION:
            return self._analyze_correlation(query)
        elif analysis_type == AnalysisType.DISTRIBUTION:
            return self._analyze_distribution(query)
        elif analysis_type == AnalysisType.COMPARISON:
            return self._analyze_comparison(query)
        elif analysis_type == AnalysisType.TREND:
            return self._analyze_trend(query)
        elif analysis_type == AnalysisType.OUTLIER:
            return self._analyze_outliers(query)
        elif analysis_type == AnalysisType.MISSING_DATA:
            return self._analyze_missing_data(query)
        elif analysis_type == AnalysisType.CATEGORICAL:
            return self._analyze_categorical(query)
        else:
            return self._general_analysis(query)
    
    def _classify_query(self, query: str) -> AnalysisType:
        """Classify the user query to determine analysis type"""
        query_lower = query.lower()
        
        # Keywords for different analysis types
        keywords = {
            AnalysisType.SUMMARY: ["summary", "overview", "describe", "general", "basic info"],
            AnalysisType.CORRELATION: ["correlation", "relationship", "associate", "relate", "connect"],
            AnalysisType.DISTRIBUTION: ["distribution", "spread", "histogram", "frequency", "range"],
            AnalysisType.COMPARISON: ["compare", "difference", "versus", "vs", "between"],
            AnalysisType.TREND: ["trend", "pattern", "over time", "temporal", "sequence"],
            AnalysisType.OUTLIER: ["outlier", "anomaly", "unusual", "extreme", "abnormal"],
            AnalysisType.MISSING_DATA: ["missing", "null", "empty", "incomplete", "nan"],
            AnalysisType.CATEGORICAL: ["category", "group", "class", "type", "segment"]
        }
        
        for analysis_type, words in keywords.items():
            if any(word in query_lower for word in words):
                return analysis_type
        
        return AnalysisType.SUMMARY  # Default
    
    def _analyze_summary(self, query: str) -> AnalysisResult:
        """Provide comprehensive data summary with visualizations"""
        insights = []
        recommendations = []
        visualizations = []
        
        # Basic statistics
        total_rows = len(self.data)
        total_cols = len(self.data.columns)
        missing_count = self.data.isnull().sum().sum()
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        insights.append(f"Dataset contains {total_rows:,} rows and {total_cols} columns")
        insights.append(f"Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
        
        if missing_count > 0:
            missing_pct = (missing_count / (total_rows * total_cols)) * 100
            insights.append(f"Dataset has {missing_count:,} missing values ({missing_pct:.1f}% of total)")
            recommendations.append("Consider handling missing values before analysis")
        
        # Create visualizations
        if len(numeric_cols) > 0:
            # Clean data for plotting
            clean_data = self._clean_data_for_plotting(self.data)
            
            # Numeric columns distribution overview
            fig_numeric = make_subplots(
                rows=min(2, len(numeric_cols)),
                cols=min(2, (len(numeric_cols) + 1) // 2),
                subplot_titles=list(numeric_cols[:4])
            )
            
            for i, col in enumerate(numeric_cols[:4]):
                row = i // 2 + 1
                col_pos = i % 2 + 1
                fig_numeric.add_trace(
                    go.Histogram(x=clean_data[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig_numeric.update_layout(title="Distribution of Numeric Columns", height=600)
            visualizations.append({
                "type": "plotly",
                "figure": fig_numeric,
                "caption": "Overview of numeric column distributions"
            })
        
        if len(categorical_cols) > 0:
            # Top categorical column analysis
            top_cat_col = categorical_cols[0]
            clean_data = self._clean_data_for_plotting(self.data)
            value_counts = clean_data[top_cat_col].value_counts().head(10)
            
            fig_cat = px.bar(
                x=value_counts.values,
                y=value_counts.index,
                orientation='h',
                title=f"Top Values in {top_cat_col}",
                labels={'x': 'Count', 'y': top_cat_col}
            )
            visualizations.append({
                "type": "plotly", 
                "figure": fig_cat,
                "caption": f"Most frequent values in {top_cat_col}"
            })
        
        # Data types overview
        dtype_counts = self.data.dtypes.value_counts()
        fig_dtypes = px.pie(
            values=dtype_counts.values,
            names=dtype_counts.index,
            title="Data Types Distribution"
        )
        visualizations.append({
            "type": "plotly",
            "figure": fig_dtypes,
            "caption": "Distribution of column data types"
        })
        
        text_response = f"""
## 📊 Dataset Summary

Your dataset contains **{total_rows:,} rows** and **{total_cols} columns**.

### Column Breakdown:
- **Numeric columns**: {len(numeric_cols)} ({', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''})
- **Categorical columns**: {len(categorical_cols)} ({', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''})

### Data Quality:
- **Missing values**: {missing_count:,} ({(missing_count/(total_rows*total_cols)*100):.1f}% of total)
- **Complete rows**: {total_rows - self.data.isnull().any(axis=1).sum():,}

The visualizations above show the distribution patterns and data type composition of your dataset.
        """
        
        return AnalysisResult(text_response, visualizations, insights, recommendations)
    
    def _analyze_correlation(self, query: str) -> AnalysisResult:
        """Analyze correlations between variables"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return AnalysisResult(
                "Need at least 2 numeric columns to analyze correlations.",
                [], [], ["Add more numeric data for correlation analysis"]
            )
        
        # Calculate correlation matrix
        corr_matrix = self.data[numeric_cols].corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Moderate to strong correlation
                    strong_correlations.append({
                        'col1': corr_matrix.columns[i],
                        'col2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        # Sort by absolute correlation strength
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        visualizations = []
        insights = []
        recommendations = []
        
        # Correlation heatmap
        try:
            fig_heatmap = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix",
                color_continuous_scale="RdBu",
                range_color=[-1, 1]
            )
            visualizations.append({
                "type": "plotly",
                "figure": fig_heatmap,
                "caption": "Correlation heatmap showing relationships between numeric variables"
            })
        except Exception as e:
            print(f"Error creating heatmap: {e}")
        
        # Scatter plot for strongest correlation
        if strong_correlations:
            strongest = strong_correlations[0]
            try:
                clean_data = self._clean_data_for_plotting(self.data)
                fig_scatter = px.scatter(
                    clean_data,
                    x=strongest['col1'],
                    y=strongest['col2'],
                    title=f"Strongest Correlation: {strongest['col1']} vs {strongest['col2']}",
                    trendline="ols"
                )
                visualizations.append({
                    "type": "plotly",
                    "figure": fig_scatter,
                    "caption": f"Scatter plot showing correlation of {strongest['correlation']:.3f}"
                })
            except Exception as e:
                print(f"Error creating scatter plot: {e}")
        
        # Generate insights
        if strong_correlations:
            insights.append(f"Found {len(strong_correlations)} strong correlations (|r| > 0.5)")
            for corr in strong_correlations[:3]:  # Top 3
                direction = "positive" if corr['correlation'] > 0 else "negative"
                strength = "very strong" if abs(corr['correlation']) > 0.8 else "strong"
                insights.append(f"{corr['col1']} and {corr['col2']}: {strength} {direction} correlation ({corr['correlation']:.3f})")
        else:
            insights.append("No strong correlations found between numeric variables")
        
        if len(strong_correlations) > 3:
            recommendations.append("Consider investigating the strongest correlations for business insights")
        
        text_response = f"""
## 🔗 Correlation Analysis

I found **{len(strong_correlations)} significant correlations** (|r| > 0.5) in your numeric data.

### Key Findings:
"""
        
        if strong_correlations:
            for i, corr in enumerate(strong_correlations[:5], 1):
                direction = "📈 Positive" if corr['correlation'] > 0 else "📉 Negative"
                text_response += f"\n{i}. **{corr['col1']}** ↔ **{corr['col2']}**: {direction} ({corr['correlation']:.3f})"
        else:
            text_response += "\nNo strong correlations detected between numeric variables."
        
        text_response += f"\n\nThe correlation matrix above shows all pairwise relationships, with values ranging from -1 (perfect negative) to +1 (perfect positive)."
        
        return AnalysisResult(text_response, visualizations, insights, recommendations)
    
    def _analyze_distribution(self, query: str) -> AnalysisResult:
        """Analyze data distributions"""
        # Extract column name from query if mentioned
        mentioned_cols = [col for col in self.data.columns if col.lower() in query.lower()]
        
        if mentioned_cols:
            target_cols = mentioned_cols[:2]  # Analyze up to 2 mentioned columns
        else:
            # Default to first few numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            target_cols = list(numeric_cols[:2])
        
        if not target_cols:
            return AnalysisResult(
                "No suitable columns found for distribution analysis.",
                [], [], ["Specify a numeric column for distribution analysis"]
            )
        
        visualizations = []
        insights = []
        recommendations = []
        
        for col in target_cols:
            if self.data[col].dtype in ['int64', 'float64']:
                try:
                    # Clean data for plotting
                    clean_data = self._clean_data_for_plotting(self.data)
                    
                    # Histogram with statistics
                    fig_hist = px.histogram(
                        clean_data,
                        x=col,
                        nbins=30,
                        title=f"Distribution of {col}",
                        marginal="box"
                    )
                    
                    # Add statistics annotations
                    mean_val = clean_data[col].mean()
                    median_val = clean_data[col].median()
                    
                    fig_hist.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                                     annotation_text=f"Mean: {mean_val:.2f}")
                    fig_hist.add_vline(x=median_val, line_dash="dash", line_color="green",
                                     annotation_text=f"Median: {median_val:.2f}")
                    
                    visualizations.append({
                        "type": "plotly",
                        "figure": fig_hist,
                        "caption": f"Distribution of {col} with mean and median lines"
                    })
                except Exception as e:
                    print(f"Error creating histogram for {col}: {e}")
                
                # Generate insights about distribution
                skewness = self.data[col].skew()
                if abs(skewness) < 0.5:
                    dist_shape = "approximately normal"
                elif skewness > 0.5:
                    dist_shape = "right-skewed (positively skewed)"
                else:
                    dist_shape = "left-skewed (negatively skewed)"
                
                insights.append(f"{col} distribution is {dist_shape} (skewness: {skewness:.2f})")
                
                # Check for outliers
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = self.data[(self.data[col] < (Q1 - 1.5 * IQR)) | 
                                   (self.data[col] > (Q3 + 1.5 * IQR))]
                
                if len(outliers) > 0:
                    insights.append(f"Found {len(outliers)} potential outliers in {col}")
                    recommendations.append(f"Consider investigating outliers in {col}")
        
        text_response = f"""
## 📊 Distribution Analysis

Analyzed the distribution of **{', '.join(target_cols)}**.

### Statistical Summary:
"""
        
        for col in target_cols:
            if self.data[col].dtype in ['int64', 'float64']:
                stats = self.data[col].describe()
                text_response += f"""
**{col}:**
- Mean: {stats['mean']:.2f}
- Median: {stats['50%']:.2f}  
- Std Dev: {stats['std']:.2f}
- Range: {stats['min']:.2f} to {stats['max']:.2f}
"""
        
        return AnalysisResult(text_response, visualizations, insights, recommendations)
    
    def _analyze_comparison(self, query: str) -> AnalysisResult:
        """Compare different groups or categories"""
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) == 0 or len(numeric_cols) == 0:
            return AnalysisResult(
                "Need both categorical and numeric columns for comparison analysis.",
                [], [], ["Ensure you have both categorical and numeric data"]
            )
        
        # Use first categorical and numeric column for comparison
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        
        # Limit to top categories to avoid cluttered plots
        top_categories = self.data[cat_col].value_counts().head(8).index
        filtered_data = self.data[self.data[cat_col].isin(top_categories)]
        
        visualizations = []
        insights = []
        recommendations = []
        
        # Box plot for comparison
        fig_box = px.box(
            filtered_data,
            x=cat_col,
            y=num_col,
            title=f"Comparison of {num_col} across {cat_col}"
        )
        fig_box.update_xaxes(tickangle=45)
        visualizations.append({
            "type": "plotly",
            "figure": fig_box,
            "caption": f"Box plot comparing {num_col} distribution across different {cat_col} categories"
        })
        
        # Bar chart with mean values
        mean_by_category = filtered_data.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
        fig_bar = px.bar(
            x=mean_by_category.index,
            y=mean_by_category.values,
            title=f"Average {num_col} by {cat_col}",
            labels={'x': cat_col, 'y': f'Average {num_col}'}
        )
        fig_bar.update_xaxes(tickangle=45)
        visualizations.append({
            "type": "plotly",
            "figure": fig_bar,
            "caption": f"Average {num_col} values for each {cat_col} category"
        })
        
        # Generate insights
        highest_cat = mean_by_category.index[0]
        lowest_cat = mean_by_category.index[-1]
        difference = mean_by_category.iloc[0] - mean_by_category.iloc[-1]
        
        insights.append(f"Highest average {num_col}: {highest_cat} ({mean_by_category.iloc[0]:.2f})")
        insights.append(f"Lowest average {num_col}: {lowest_cat} ({mean_by_category.iloc[-1]:.2f})")
        insights.append(f"Difference between highest and lowest: {difference:.2f}")
        
        # Statistical significance could be added here
        recommendations.append(f"Consider investigating why {highest_cat} has higher {num_col} values")
        
        text_response = f"""
## 🔄 Comparison Analysis

Comparing **{num_col}** across different **{cat_col}** categories.

### Key Findings:
- **Highest performing**: {highest_cat} (avg: {mean_by_category.iloc[0]:.2f})
- **Lowest performing**: {lowest_cat} (avg: {mean_by_category.iloc[-1]:.2f})
- **Range**: {difference:.2f} difference between top and bottom categories

The visualizations show both the distribution (box plot) and average values (bar chart) for easy comparison.
        """
        
        return AnalysisResult(text_response, visualizations, insights, recommendations)
    
    def _analyze_trend(self, query: str) -> AnalysisResult:
        """Analyze trends over time or sequences"""
        # Look for date columns or sequential data
        date_cols = self.data.select_dtypes(include=['datetime64']).columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        visualizations = []
        insights = []
        recommendations = []
        
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            # Time series analysis
            date_col = date_cols[0]
            num_col = numeric_cols[0]
            
            # Sort by date
            trend_data = self.data.sort_values(date_col)
            
            fig_trend = px.line(
                trend_data,
                x=date_col,
                y=num_col,
                title=f"Trend of {num_col} over time"
            )
            visualizations.append({
                "type": "plotly",
                "figure": fig_trend,
                "caption": f"Time series showing {num_col} trend"
            })
            
            # Calculate trend direction
            correlation_with_time = np.corrcoef(
                range(len(trend_data)), 
                trend_data[num_col].fillna(trend_data[num_col].mean())
            )[0, 1]
            
            if correlation_with_time > 0.1:
                trend_direction = "increasing"
            elif correlation_with_time < -0.1:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
            
            insights.append(f"Overall trend is {trend_direction} (correlation: {correlation_with_time:.3f})")
            
        else:
            # Sequential analysis without dates
            if len(numeric_cols) >= 2:
                col1, col2 = numeric_cols[0], numeric_cols[1]
                
                fig_seq = px.line(
                    self.data.reset_index(),
                    x='index',
                    y=[col1, col2],
                    title=f"Sequential Pattern: {col1} and {col2}",
                    labels={'index': 'Row Number', 'value': 'Value'}
                )
                visualizations.append({
                    "type": "plotly",
                    "figure": fig_seq,
                    "caption": "Sequential patterns in the data"
                })
                
                insights.append("Analyzed sequential patterns in numeric data")
            else:
                return AnalysisResult(
                    "Need date column or multiple numeric columns for trend analysis.",
                    [], [], ["Add date/time column for proper trend analysis"]
                )
        
        text_response = """
## 📈 Trend Analysis

Analyzed temporal or sequential patterns in your data.
        """
        
        return AnalysisResult(text_response, visualizations, insights, recommendations)
    
    def _analyze_outliers(self, query: str) -> AnalysisResult:
        """Detect and analyze outliers"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return AnalysisResult(
                "No numeric columns found for outlier analysis.",
                [], [], ["Add numeric data for outlier detection"]
            )
        
        visualizations = []
        insights = []
        recommendations = []
        
        outlier_summary = {}
        
        for col in numeric_cols[:3]:  # Analyze top 3 numeric columns
            # IQR method for outlier detection
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            outlier_summary[col] = len(outliers)
            
            if len(outliers) > 0:
                # Box plot to show outliers
                fig_box = px.box(
                    self.data,
                    y=col,
                    title=f"Outliers in {col}",
                    points="outliers"
                )
                visualizations.append({
                    "type": "plotly",
                    "figure": fig_box,
                    "caption": f"Box plot highlighting outliers in {col}"
                })
                
                insights.append(f"Found {len(outliers)} outliers in {col} ({len(outliers)/len(self.data)*100:.1f}% of data)")
                
                if len(outliers) > len(self.data) * 0.1:  # More than 10% outliers
                    recommendations.append(f"High number of outliers in {col} - investigate data quality")
        
        # Summary visualization
        if outlier_summary:
            fig_summary = px.bar(
                x=list(outlier_summary.keys()),
                y=list(outlier_summary.values()),
                title="Outlier Count by Column",
                labels={'x': 'Column', 'y': 'Number of Outliers'}
            )
            visualizations.append({
                "type": "plotly",
                "figure": fig_summary,
                "caption": "Summary of outliers found in each numeric column"
            })
        
        total_outliers = sum(outlier_summary.values())
        text_response = f"""
## 🎯 Outlier Analysis

Found **{total_outliers} total outliers** across {len(outlier_summary)} numeric columns.

### Outlier Summary:
"""
        
        for col, count in outlier_summary.items():
            percentage = (count / len(self.data)) * 100
            text_response += f"\n- **{col}**: {count} outliers ({percentage:.1f}% of data)"
        
        if total_outliers == 0:
            text_response += "\n🎉 No significant outliers detected in your numeric data!"
        
        return AnalysisResult(text_response, visualizations, insights, recommendations)
    
    def _analyze_missing_data(self, query: str) -> AnalysisResult:
        """Analyze missing data patterns"""
        missing_counts = self.data.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if len(missing_cols) == 0:
            return AnalysisResult(
                "🎉 Great news! No missing values found in your dataset.",
                [], ["Dataset is complete with no missing values"], 
                ["Your data quality is excellent"]
            )
        
        visualizations = []
        insights = []
        recommendations = []
        
        # Missing data heatmap
        if len(missing_cols) > 1:
            missing_matrix = self.data[missing_cols.index].isnull()
            fig_heatmap = px.imshow(
                missing_matrix.T,
                title="Missing Data Pattern",
                labels={'x': 'Row Index', 'y': 'Columns', 'color': 'Missing'},
                color_continuous_scale=["white", "red"]
            )
            visualizations.append({
                "type": "plotly",
                "figure": fig_heatmap,
                "caption": "Heatmap showing missing data patterns (red = missing)"
            })
        
        # Bar chart of missing counts
        fig_bar = px.bar(
            x=missing_cols.index,
            y=missing_cols.values,
            title="Missing Values by Column",
            labels={'x': 'Column', 'y': 'Missing Count'}
        )
        fig_bar.update_xaxes(tickangle=45)
        visualizations.append({
            "type": "plotly",
            "figure": fig_bar,
            "caption": "Count of missing values in each column"
        })
        
        # Generate insights
        total_missing = missing_counts.sum()
        total_cells = len(self.data) * len(self.data.columns)
        missing_percentage = (total_missing / total_cells) * 100
        
        insights.append(f"Total missing values: {total_missing:,} ({missing_percentage:.1f}% of all data)")
        insights.append(f"Columns affected: {len(missing_cols)} out of {len(self.data.columns)}")
        
        # Most problematic columns
        worst_col = missing_cols.index[0]
        worst_count = missing_cols.iloc[0]
        worst_pct = (worst_count / len(self.data)) * 100
        
        insights.append(f"Most missing data in: {worst_col} ({worst_count} values, {worst_pct:.1f}%)")
        
        # Recommendations based on missing data severity
        if worst_pct > 50:
            recommendations.append(f"Consider dropping {worst_col} column (>50% missing)")
        elif worst_pct > 20:
            recommendations.append(f"Investigate why {worst_col} has high missing rate")
        else:
            recommendations.append("Missing data levels are manageable with proper imputation")
        
        text_response = f"""
## 🕳️ Missing Data Analysis

Found missing values in **{len(missing_cols)} columns** affecting {missing_percentage:.1f}% of your dataset.

### Missing Data Summary:
"""
        
        for col, count in missing_cols.head(5).items():
            percentage = (count / len(self.data)) * 100
            text_response += f"\n- **{col}**: {count:,} missing ({percentage:.1f}%)"
        
        if len(missing_cols) > 5:
            text_response += f"\n- ... and {len(missing_cols) - 5} more columns"
        
        return AnalysisResult(text_response, visualizations, insights, recommendations)
    
    def _analyze_categorical(self, query: str) -> AnalysisResult:
        """Analyze categorical data"""
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            return AnalysisResult(
                "No categorical columns found for analysis.",
                [], [], ["Add categorical/text columns for this analysis"]
            )
        
        visualizations = []
        insights = []  
        recommendations = []
        
        # Analyze first few categorical columns
        for col in categorical_cols[:2]:
            value_counts = self.data[col].value_counts()
            
            # Bar chart for top values
            fig_bar = px.bar(
                x=value_counts.head(10).values,
                y=value_counts.head(10).index,
                orientation='h',
                title=f"Top Values in {col}",
                labels={'x': 'Count', 'y': col}
            )
            visualizations.append({
                "type": "plotly",
                "figure": fig_bar,
                "caption": f"Most frequent values in {col}"
            })
            
            # Generate insights
            unique_count = self.data[col].nunique()
            most_common = value_counts.index[0]
            most_common_count = value_counts.iloc[0]
            most_common_pct = (most_common_count / len(self.data)) * 100
            
            insights.append(f"{col} has {unique_count} unique values")
            insights.append(f"Most common: '{most_common}' ({most_common_count} occurrences, {most_common_pct:.1f}%)")
            
            # Check for data quality issues
            if unique_count == len(self.data):
                recommendations.append(f"{col} appears to be an identifier (all unique values)")
            elif unique_count < 5:
                recommendations.append(f"{col} has very few categories - good for grouping analysis")
        
        text_response = f"""
## 📊 Categorical Data Analysis

Analyzed **{len(categorical_cols)} categorical columns** in your dataset.

### Column Summary:
"""
        
        for col in categorical_cols:
            unique_count = self.data[col].nunique()
            text_response += f"\n- **{col}**: {unique_count} unique values"
        
        return AnalysisResult(text_response, visualizations, insights, recommendations)
    
    def _general_analysis(self, query: str) -> AnalysisResult:
        """Handle general queries that don't fit specific categories"""
        # Try to extract column names from the query
        mentioned_cols = [col for col in self.data.columns if col.lower() in query.lower()]
        
        if mentioned_cols:
            col = mentioned_cols[0]
            if self.data[col].dtype in ['int64', 'float64']:
                return self._analyze_distribution(f"distribution of {col}")
            else:
                return self._analyze_categorical(f"analyze {col}")
        else:
            return self._analyze_summary("general overview")
    
    def get_column_suggestions(self) -> Dict[str, List[str]]:
        """Get column suggestions for user queries"""
        return {
            "numeric": list(self.data.select_dtypes(include=[np.number]).columns),
            "categorical": list(self.data.select_dtypes(include=['object']).columns),
            "datetime": list(self.data.select_dtypes(include=['datetime64']).columns)
        }