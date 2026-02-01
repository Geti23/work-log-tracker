# ExpenseAlly AI Service

Python microservice for spending anomaly detection using Isolation Forest ML model.

## Features

- **Spending Anomaly Detection** - Detects unusual spending patterns using Isolation Forest with budget integration
- **Spending Period Prediction** - Predicts future high-spending periods and identifies seasonal patterns
- **Spending Pattern Classification** - Classifies user spending behavior into personality types
- **Budget Integration** - Compares spending against budgets and provides recommendations
- **Direct Database Access** - Connects directly to SQL Server
- **JWT Authentication** - Secure communication with backend
- **RESTful API** - Clean API endpoints

## Quick Start

### Prerequisites
- Python 3.10+
- SQL Server with ExpenseAlly database
- Microsoft ODBC Driver 17 for SQL Server

### Installation

#### Windows

1. **Install ODBC Driver**:
   - Download: https://go.microsoft.com/fwlink/?linkid=2249004
   - Or: `winget install Microsoft.ODBCDriver.17`

2. **Setup Python environment**:
   ```bash
   cd ai-service
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   copy env.example .env
   # Edit .env if needed (defaults should work)
   ```

4. **Run service**:
   ```bash
   python main.py
   ```

#### macOS

1. **Install ODBC Driver**:
   ```bash
   # Using Homebrew (recommended)
   brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
   brew update
   brew install msodbcsql17 mssql-tools
   
   # Or download from: https://go.microsoft.com/fwlink/?linkid=2249004
   ```

2. **Setup Python environment**:
   ```bash
   cd ai-service
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp env.example .env
   # Edit .env if needed (defaults should work)
   ```

4. **Run service**:
   ```bash
   python main.py
   ```

Service runs on `http://localhost:8000`

## API Endpoints

### Health Checks
- `GET /` - Root endpoint
- `GET /api/v1/health` - Health check
- `GET /api/v1/health/ready` - Readiness (includes DB check)
- `GET /api/v1/health/live` - Liveness check

### AI Features

#### Service Status
- `GET /api/v1/ai/status` - AI service status and metrics

#### Anomaly Detection
- `POST /api/v1/ai/detect-anomalies` - Detect spending anomalies with budget integration

#### Spending Prediction
- `POST /api/v1/ai/predict-spending-periods` - Predict future high-spending periods

#### Pattern Classification
- `POST /api/v1/ai/classify-pattern` - Classify user spending behavior patterns

### API Examples

#### Anomaly Detection

**Request:**
```json
{
  "user_id": "user-guid-here",
  "period_days": 7
}
```

**Response:**
```json
{
  "anomalies": [
    {
      "category_id": "...",
      "category_name": "Restaurants",
      "current_amount": 500.0,
      "average_historical": 300.0,
      "percentage_change": 66.7,
      "period_days": 7,
      "is_increase": true,
      "anomaly_score": -0.5,
      "budget_limit": 300.0,
      "budget_spent": 450.0,
      "budget_overage": 150.0,
      "budget_percentage_used": 150.0,
      "recommendation": "Consider reducing dining out by $21.43 per day..."
    }
  ],
  "messages": [
    "This week you spent 67% more on Restaurants than usual. You've exceeded your budget by 50%."
  ],
  "status": "sufficient_data",
  "statusMessage": ""
}
```

#### Spending Prediction

**Request:**
```json
{
  "user_id": "user-guid-here",
  "forecast_months": 6
}
```

**Response:**
```json
{
  "predictions": [
    {
      "year": 2024,
      "month": 12,
      "month_name": "December",
      "predicted_amount": 2500.0,
      "average_amount": 2000.0,
      "expected_increase": 25.0,
      "confidence": "high",
      "reasoning": ["Historical high spending in December", "Holiday season pattern"]
    }
  ],
  "seasonal_patterns": [
    {
      "month": 12,
      "month_name": "December",
      "average_spending": 2500.0,
      "pattern_type": "high",
      "consistency": "consistent"
    }
  ],
  "recurring_events": [
    {
      "event_name": "Holiday Season",
      "start_month": 11,
      "end_month": 12,
      "average_increase": 30.0,
      "occurrences": 3
    }
  ],
  "saving_suggestions": [
    {
      "type": "seasonal",
      "month": 12,
      "month_name": "December",
      "year": 2024,
      "suggestion": "Plan ahead for holiday spending",
      "potential_savings": 500.0,
      "priority": "high"
    }
  ],
  "average_monthly_spending": 2000.0,
  "insights": ["Your spending typically increases 25% in December"],
  "status": "sufficient_data"
}
```

#### Pattern Classification

**Request:**
```
POST /api/v1/ai/classify-pattern?user_id=user-guid-here
```

**Response:**
```json
{
  "pattern": "Consistent Spender 📊",
  "confidence": 0.85,
  "explanation": "Your spending is very consistent with low variation. You maintain steady financial habits."
}
```

## Backend Integration

Call from .NET backend with JWT token:

```csharp
var client = new HttpClient();
client.DefaultRequestHeaders.Authorization = 
    new AuthenticationHeaderValue("Bearer", jwtToken);
    
var response = await client.PostAsJsonAsync(
    "http://localhost:8000/api/v1/ai/detect-anomalies",
    new { user_id = userId, period_days = 7 }
);
```

## How It Works

### Anomaly Detection
1. Retrieves user's expense transactions from last 90 days
2. Groups spending by category and time period
3. Uses Isolation Forest ML model combined with statistical analysis to detect anomalies
4. Compares current period spending with historical patterns
5. Integrates budget data to provide overage warnings and recommendations
6. Returns anomalies with percentage changes, budget status, and actionable messages

### Spending Prediction
1. Analyzes 12+ months of historical transaction data
2. Identifies seasonal spending patterns (monthly trends)
3. Detects recurring high-spending events (holidays, vacations, etc.)
4. Predicts future high-spending periods using pattern analysis
5. Provides saving suggestions based on predicted patterns
6. Includes income analysis for surplus calculations

### Pattern Classification
1. Analyzes transaction amounts and frequency
2. Calculates statistical measures (mean, std deviation, coefficient of variation)
3. Classifies spending behavior into personality types:
   - Consistent Spender
   - Impulse Buyer
   - Budget-Conscious
   - High Roller
   - And more...
4. Provides confidence scores and explanations

## Dependencies

- **FastAPI** - Web framework
- **SQLAlchemy** - Database ORM
- **scikit-learn** - Isolation Forest model
- **numpy** - Numerical operations
- **PyODBC** - SQL Server connectivity

## Troubleshooting

**Database connection fails:**
- Check SQL Server is running
- Verify database name in `.env`
- Check ODBC driver:
  - Windows: `odbcinst -q -d`
  - macOS: `odbcinst -q -d` (after installing driver)

**Dependencies missing:**
- Windows: Activate venv: `venv\Scripts\activate`
- macOS: Activate venv: `source venv/bin/activate`
- Install: `pip install -r requirements.txt`

**Memory error installing packages:**
- Install packages one by one: `pip install numpy`, then `pip install scikit-learn`

**ODBC driver issues on macOS:**
- Ensure Homebrew is installed: `brew --version`
- If installation fails, download directly from Microsoft
- Verify installation: `odbcinst -q -d | grep -i odbc`

**Run diagnostic:**
```bash
python check_setup.py
```
