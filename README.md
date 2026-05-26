# Insurance Risk Analytics for AlphaCare Insurance Solutions (ACIS)

## Project Overview

This project analyzes 18 months of historical car insurance data (Feb 2014 - Aug 2015) for ACIS in South Africa. The goal is to identify low-risk segments for premium optimization and build predictive models for risk-based pricing.

## Business Objectives

- Discover low-risk customer segments for targeted marketing
- Develop data-driven pricing models
- Optimize insurance premiums based on risk assessment

## Project Structure

```
insurance-risk-analytics/
├── .github/workflows/    # CI/CD pipelines
├── data/                 # Datasets (tracked by DVC)
├── notebooks/            # Analysis notebooks
├── reports/              # Generated analysis reports
├── src/                  # Source code modules
│   ├── data_loader.py    # Data loading utilities
│   ├── eda_utils.py      # EDA visualization utilities
│   ├── hypothesis_tests.py   # Statistical tests
│   └── modeling.py       # ML models
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
├── .dvc/                 # DVC configuration
└── README.md             # This file
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/elu-di/insurance-risk-analytics.git
cd insurance-risk-analytics
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download data with DVC:
```bash
dvc pull
```

## Project Stages

### 1. Exploratory Data Analysis (EDA)
- Load and clean insurance data
- Analyze claim patterns and loss ratios
- Identify risk drivers by province, gender, and vehicle type
- Detect outliers and temporal trends

### 2. Hypothesis Testing
- Statistical tests for risk factors
- Validate business assumptions
- Identify significant relationships

### 3. Predictive Modeling
- Build risk prediction models
- Compare model performance
- Generate premium recommendations

## Usage

### Running Analysis Notebooks

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_hypothesis_testing.ipynb
jupyter notebook notebooks/03_modeling.ipynb
```

### Running Tests

```bash
pytest tests/ -v
```

### Running Linting

```bash
flake8 src/ tests/
black src/ tests/
```

## Data

The dataset contains 18 months of car insurance claims data from South Africa (Feb 2014 - Aug 2015).

**Key Metrics:**
- Total Policies: ~19K
- Total Claims: ~7.7M
- Overall Loss Ratio: ~0.48
- Claim Frequency: ~28%

## Key Findings

- **Geographic Risk**: Provinces show significant variation in loss ratios
- **Gender Patterns**: Clear gender-based risk differentiation
- **Vehicle Impact**: Vehicle type and age are strong risk predictors
- **Temporal Trends**: Seasonal patterns in claims and premiums

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -am 'Add feature'`
3. Push to branch: `git push origin feature/your-feature`
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Author

**Elu Di** - Insurance Analytics Project

## Date

**Date:** May 2026  
**Data Period:** Feb 2014 - Aug 2015
