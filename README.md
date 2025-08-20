# Blinkit - Analysis (Interactive Dashbord using excel and  Data Analysis Python Project)
## Problem Statement 
To conduct a comprehensive analysis of Blinkit's sales performance, customer satisfaction, and inventory distribution to identify key insights and opportunities for optimization using various KPIs and visualizations in Power BI.
##  Data Set 
- <a href= "https://github.com/Sushmita-source/Blinkit_Analysis/blob/main/BlinkIT%20_Grocery%20_Data.csv"> Data Set</a>
## Question KPI'S 
- Total Sales: The overall revenue generated from all items sold.
- Average Sales: The average revenue per sale.
- Number of Items: The total count of different items sold.
- Average Rating: The average customer rating for items sold.

- Dashord Intraction <a href= "https://github.com/Sushmita-source/Blinkit_Analysis/blob/main/Blinkit_Analysis.png" > View Dashbord </a>

## Granualar Requirement
- Total Sales by Fat Content
- Total Sales by Item Type	
- Fat Content by Outlet for Total Sales
- Total Sales by Outlet Establishment

## Chart's Requirement
- Percentage of Sales by Outlet Size
- Sales by Outlet Location
- All Metrics by Outlet Type
  
## Process For Creating Power BI Dashboard 
- Collect and Prepare Data – Gather relevant datasets and ensure they are accurate, clean, and ready for analysis.
- Import Data into Power BI Desktop – Load datasets into Power BI Desktop for processing and visualization.
- Clean & Transform Data in Power Query – Use Power Query to remove errors, format fields, and shape data for reporting.
- Build Data Model & Relationships – Create relationships between tables and design a structured data model.
- Create DAX Measures – Write DAX formulas to calculate metrics and KPIs dynamically.
- Design Visuals & Charts – Build meaningful charts, graphs, and visuals to represent insights clearly.
- Add Slicers, Filters & Interactivity – Implement slicers and filters to allow dynamic data exploration.
- Format & Polish Dashboard – Apply consistent styling, align visuals, and enhance readability.
- Publish to Power BI Service – Upload the report to Power BI Service for online access.
- Share & Set Refresh Schedules – Share dashboards with stakeholders and configure automatic data refresh.
- Maintain & Update Dashboard – Regularly review, update, and improve the dashboard for ongoing accuracy.

## Dashboard

<img width="1328" height="746" alt="Blinkit_Analysis" src="https://github.com/user-attachments/assets/3184dd9e-fc24-472a-aa3b-ab81c9e681cc" />

## Process For Data Analysis using Python(Numpy,Pandas,Seaborn)
- Define the question and key KPIs – Clearly outline the business problem and determine measurable KPIs for analysis.
- Identify and collect data sources – Gather relevant datasets from CSV, Excel, databases, or APIs using Python tools.
- Load data into Python (pandas) – Use pandas.read_csv() or similar functions to load datasets into DataFrames for manipulation.
- Inspect and audit the dataset (shape, types, missing) – Apply .shape, .info(), .isna().sum() to check structure, data types, and missing values.
- Clean and pre-process data (missing values, duplicates, types) – Use dropna(), fillna(), drop_duplicates(), and as type() to clean and format data.
- Engineer features and date/time variables – Create new columns, extract dates with pd.to_datetime() and .dt attributes, or use NumPy operations.
- Perform exploratory data analysis (visuals and summaries) – Generate summaries with .describe() and visualize data using Matplotlib or Seaborn.
- Run statistical checks and correlation analysis – Use df.corr() or numpy.corrcoef() to measure relationships and validate assumptions.
- Aggregate metrics and build pivot tables – Apply groupby() and pivot_table() in pandas to summarize key metrics.
- Evaluate results and test business validity – Compare model metrics or aggregated KPIs against real-world expectations.
- Create final visuals and a concise narrative of insights – Present insights with clear plots and a well-structured summary.
- Ensure reproducibility (notebooks, scripts, requirements.txt) – Save Jupyter Notebooks, Python scripts, and dependency files for future runs.
- Version control and push to GitHub – Track changes with Git and upload the complete project to GitHub.
- Share results, deploy/automate, and monitor over time – Distribute reports, set automated analysis scripts, and review periodically.

## python File 
- <a href= "https://github.com/Sushmita-source/Blinkit_Analysis/blob/main/Blinkit%20Analysis%20in%20Python.ipynb"> Python File </a>
## Data Modeling 
## Project Overview
- This project predicts grocery item sales using Machine Learning Regression models, We aim to help retail businesses optimize pricing, and outlet performance.
## Steps for Data Modeling
- Data Collection-Load BlinkIT dataset (CSV provided).  
- Data Cleaning - Handle missing values, outliers, inconsistent data.  
- Exploratory Data Analysis (EDA)** → Visualize trends & relationships.  
- Feature Engineering - Encode categorical variables, scaling.  
- Train-Test Split - Prepare training & testing datasets.  
- Model Selection - Random Forest & XGBoost Regressor.  
- Model Training - Fit models on training data.  
- Model Evaluation -  Compare performance using R², RMSE, MAE.  
- Hyperparameter Tuning -  Optimize parameters for better accuracy.  
## Algorithms Used
- Random Forest Regressor - Robust ensemble method, handles non-linearity.  
- XGBoost Regressor - Gradient boosting, usually higher accuracy on tabular data.
## Results
- Evaluated models using metrics: R² Score, RMSE, MAE.  
- XGBoost generally outperformed Random Forest on accuracy.
## Python File 
- <a href= "https://github.com/Sushmita-source/Blinkit_Analysis/blob/main/Blinkit%20Analysis%20in%20Python.py"> Python File </a>
## Summary 
Analyzed BlinkIT Grocery data with preprocessing, EDA, and feature engineering. Built regression models (Random Forest, XGBoost) to predict sales, optimized performance using metrics (R², RMSE), showcasing end-to-end data analysis and machine learning workflow







  

