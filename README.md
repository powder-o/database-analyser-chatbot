# 📊 Excel Data Analysis Chatbot

A powerful web application that combines the simplicity of Excel data upload with AI-powered analysis and interactive visualizations. Built with Streamlit, FastMCP, and Plotly.

## ✨ Features

- **📁 Easy Data Upload**: Drag and drop Excel files (.xlsx, .xls)
- **💬 AI Chatbot Interface**: Ask questions about your data in natural language
- **📊 Interactive Dashboard**: Overview of your dataset with key metrics
- **📈 Rich Visualizations**: Multiple chart types (scatter, line, bar, histogram, box plots, heatmaps)
- **🔍 Automated Insights**: AI-generated insights including correlations, outliers, and statistical summaries
- **🎯 Real-time Analysis**: Instant feedback and analysis of your data

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📖 How to Use

### 1. Upload Your Data
- Use the sidebar to upload your Excel file
- Supported formats: `.xlsx` and `.xls`
- The app will automatically load and analyze your data

### 2. Explore the Dashboard
- **Dashboard Tab**: View key metrics, data preview, and quick statistics
- **Chat Tab**: Ask questions about your data in natural language
- **Visualizations Tab**: Create interactive charts and plots
- **Insights Tab**: View automated analysis and insights

### 3. Chat with Your Data
Ask questions like:
- "What's the summary of my data?"
- "Show me correlations"
- "Are there any missing values?"
- "Tell me about outliers"

### 4. Create Visualizations
- Choose from multiple chart types
- Select columns for X and Y axes
- Add color coding with categorical variables
- Customize chart titles

## 🛠️ Technical Architecture

### Core Components

- **Streamlit**: Web application framework
- **FastMCP**: Model Context Protocol for AI tools
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### File Structure

```
├── app.py              # Main Streamlit application
├── mcp_tools.py        # FastMCP tools for data analysis
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variables template
└── README.md          # This file
```

## 🎨 Customization

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

### Adding New Features

The modular architecture makes it easy to extend:

1. **New MCP Tools**: Add functions to `mcp_tools.py`
2. **UI Components**: Modify `app.py` to add new tabs or sections
3. **Chart Types**: Extend the visualization functions

## 📊 Supported Visualizations

- **Scatter Plot**: Explore relationships between numeric variables
- **Line Chart**: Show trends over time or sequences
- **Bar Chart**: Compare categories or show distributions
- **Histogram**: Understand data distributions
- **Box Plot**: Identify outliers and quartiles
- **Heatmap**: Visualize correlation matrices

## 🔍 AI-Powered Insights

The app automatically generates:

- **Statistical Summaries**: Mean, median, standard deviation for numeric columns
- **Missing Value Analysis**: Identify and quantify missing data
- **Correlation Detection**: Find strong relationships between variables
- **Outlier Detection**: Identify unusual data points using IQR method
- **Data Type Analysis**: Automatic detection of numeric vs categorical data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

**File Upload Issues**:
- Ensure your Excel file is not corrupted
- Check that the file is in .xlsx or .xls format
- Try with a smaller file if experiencing memory issues

**Visualization Errors**:
- Ensure you have selected appropriate columns for the chart type
- Numeric columns are required for scatter plots and line charts
- Check for missing values in selected columns

**Performance Issues**:
- Large datasets (>100MB) may take time to load
- Consider using a subset of your data for initial exploration

### Getting Help

If you encounter issues:
1. Check the error messages in the app
2. Verify your data format and structure
3. Try with a sample Excel file first

## 🎯 Roadmap

Future enhancements planned:
- [ ] Support for CSV and other data formats
- [ ] Advanced statistical analysis tools
- [ ] Machine learning model integration
- [ ] Data export capabilities
- [ ] Custom dashboard layouts
- [ ] Multi-sheet Excel support

---

Built with ❤️ using modern Python data science tools.