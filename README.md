# Airline Operations Intelligence Platform

## Overview

The Airline Operations Intelligence Platform is an end-to-end Data Science and Machine Learning project built using real-world U.S. commercial flight operations data from the Bureau of Transportation Statistics (BTS). The project analyzes airline performance, identifies operational bottlenecks, quantifies delay impacts, and develops predictive models to help airlines proactively manage disruptions.

Using over 3.38 million flight records from January–June 2025, the project combines large-scale data engineering, feature engineering, exploratory data analysis, delay propagation analysis, and machine learning to generate actionable operational intelligence.

---

## Business Objectives

- Identify major causes of flight delays and cancellations
- Measure airline and airport operational performance
- Analyze delay propagation across aircraft rotations
- Quantify the operational and financial impact of delays
- Predict whether a flight will be delayed before departure
- Estimate expected delay duration for proactive decision-making

---

## Dataset

Source: U.S. Bureau of Transportation Statistics (BTS)

Coverage: January 2025 – June 2025

### Dataset Scale

- 3.38 Million Flight Records
- 192 Airports
- 4,742 Routes
- 2.5+ Million Aircraft Rotations Analyzed
- 336,876 Delay Propagation Events Identified

---

## Data Engineering Pipeline

The project begins with a large-scale data processing pipeline:

1. Data Cleaning
2. Missing Value Handling
3. Data Type Optimization
4. Feature Creation
5. Dataset Integration
6. Analytical Dataset Generation

The final dataset contains more than 45 engineered operational features.

---

## Feature Engineering

Examples of engineered features include:

- Airline Reliability Score
- Route Risk Score
- Airport Congestion Score
- Historical Delay Rate
- Peak Hour Indicator
- Holiday Period Flag
- Delay Propagation Features
- Route-Level Performance Metrics
- Airport Traffic Indicators

---

## Delay Propagation Analysis

A key component of the project is aircraft-level delay propagation analysis using aircraft tail numbers.

By tracking the same aircraft across consecutive flights:

- 2.5+ million aircraft rotations were analyzed
- 336,876 cascade delay events were detected
- Flights following a delayed arrival showed a 72.9% probability of departing late

This analysis helps quantify how operational disruptions spread throughout airline networks.

---

## Exploratory Data Analysis

EDA was conducted to analyze:

- Airline Reliability
- Airport Performance
- Delay Causes
- Route Risk
- Cancellation Trends
- Delay Distributions
- Congestion Patterns

### Key Finding

The total estimated delay-related financial impact across the analyzed period was approximately:

$4.24 Billion

---

## Machine Learning Models

### Model 1A: Pre-Departure Delay Classification

Objective:

Predict whether a flight will be delayed before departure.

Performance
- ROC-AUC: 0.78

---

### Model 1B: Gate-Level Delay Classification

Objective:

Predict flight delay using operational signals available closer to departure.

Performance
- ROC-AUC: 0.94

---

### Model 2B: Delay Duration Regression

Objective:

Estimate arrival delay duration in minutes.

Performance
- R² Score: 0.90
- Average Prediction Error: ~14 Minutes

---

## Project Structure

text airline-operations-intelligence-platform/ │ ├── data/ │   ├── raw/ │   └── processed/ │ ├── notebooks/ │ ├── src/ │ ├── outputs/ │   ├── figures/ │   ├── metrics/ │   └── exports/ │ ├── README.md ├── requirements.txt └── .gitignore 

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SQL
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Key Results

| Metric | Value |
|----------|----------|
| Flight Records | 3.38 Million |
| Airports | 192 |
| Routes | 4,742 |
| Engineered Features | 45+ |
| Aircraft Rotations Analyzed | 2.5+ Million |
| Cascade Delay Events | 336,876 |
| Delay Propagation Probability | 72.9% |
| Financial Impact Identified | $4.24 Billion |
| Delay Classification ROC-AUC | 0.94 |
| Delay Duration R² | 0.90 |

---

## Future Enhancements

- Weather Data Integration
- Real-Time Prediction Pipeline
- Interactive BI Dashboard
- Model Monitoring Framework
- Airline Operations Recommendation Engine

---

## Author

Developed as an end-to-end Data Science and Machine Learning project focused on airline operations analytics, delay prediction, and operational intelligence.