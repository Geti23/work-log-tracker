import logging
from fastapi import APIRouter, Depends, HTTPException, status as http_status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
from uuid import UUID

from app.db.database import get_db
from app.db.models import Transaction, TransactionCategory, Budget, BudgetDetail
from app.core.config import settings
from app.ml.anomaly_detector import SpendingAnomalyDetector
from app.ml.spending_pattern_classifier import SpendingPatternClassifier
from app.ml.spending_predictor import SpendingPeriodPredictor
from app.ml.metrics import metrics

logger = logging.getLogger(__name__)
router = APIRouter()


class AnomalyDetectionRequest(BaseModel):
    """Request model for anomaly detection."""
    user_id: str = Field(..., description="User ID (UUID format)")
    period_days: int = Field(
        default=7, 
        ge=1, 
        le=90, 
        description="Number of days in the current period to analyze (1-90)"
    )
    
    @validator('user_id')
    def validate_user_id(cls, v):
        """Validate that user_id is a valid UUID."""
        try:
            UUID(v)
            return v
        except ValueError:
            raise ValueError('user_id must be a valid UUID')


class AnomalyDto(BaseModel):
    """Anomaly data model."""
    category_id: str
    category_name: str
    current_amount: float
    average_historical: float
    percentage_change: float
    period_days: int
    is_increase: bool
    anomaly_score: float
    budget_limit: Optional[float] = None
    budget_spent: Optional[float] = None
    budget_overage: Optional[float] = None
    budget_percentage_used: Optional[float] = None
    recommendation: Optional[str] = None

class SpendingPatternResponse(BaseModel):
    pattern: str
    confidence: float
    explanation: str


class SpendingPredictionRequest(BaseModel):
    """Request model for spending prediction."""
    user_id: str = Field(..., description="User ID (UUID format)")
    forecast_months: int = Field(
        default=6,
        ge=1,
        le=12,
        description="Number of months to forecast ahead (1-12)"
    )
    
    @validator('user_id')
    def validate_user_id(cls, v):
        """Validate that user_id is a valid UUID."""
        try:
            UUID(v)
            return v
        except ValueError:
            raise ValueError('user_id must be a valid UUID')


class SeasonalPatternDto(BaseModel):
    """Seasonal pattern data model."""
    month: int
    month_name: str
    average_spending: float
    pattern_type: str  # 'high' or 'low'
    consistency: str
    percentage_above_average: Optional[float] = None
    percentage_below_average: Optional[float] = None


class RecurringEventDto(BaseModel):
    """Recurring event data model."""
    event_name: str
    year: Optional[int] = None
    month: Optional[int] = None
    month_name: Optional[str] = None
    amount: Optional[float] = None
    average_increase: float
    occurrences: int
    start_month: Optional[int] = None
    end_month: Optional[int] = None
    period_total: Optional[float] = None
    average_monthly: Optional[float] = None
    consistency: Optional[str] = None


class HistoricalHighMonthDto(BaseModel):
    """Historical high spending month data model."""
    year: int
    month: int
    month_name: str
    amount: float
    percentage_above_average: float
    is_seasonal: bool


class PredictionDto(BaseModel):
    """Prediction data model."""
    year: int
    month: int
    month_name: str
    predicted_amount: float
    average_amount: float
    expected_increase: float
    confidence: str
    reasoning: List[str]


class LowSpendingPeriodDto(BaseModel):
    """Low spending period data model."""
    year: int
    month: int
    month_name: str
    amount: float
    percentage_below_average: float
    potential_savings: float
    is_seasonal: bool


class IncomePatternDto(BaseModel):
    """Income pattern data model."""
    average_monthly_income: float
    average_monthly_spending: float
    average_surplus: float
    high_income_months: List[Dict[str, Any]] = Field(default_factory=list)


class SavingSuggestionDto(BaseModel):
    """Saving suggestion data model."""
    type: str
    month: int
    month_name: str
    year: int
    suggestion: str
    potential_savings: float
    priority: str
    related_period: Optional[str] = None


class SpendingPredictionResponse(BaseModel):
    """Response model for spending prediction."""
    predictions: List[PredictionDto] = Field(default_factory=list)
    seasonal_patterns: List[SeasonalPatternDto] = Field(default_factory=list)
    recurring_events: List[RecurringEventDto] = Field(default_factory=list)
    historical_high_months: List[HistoricalHighMonthDto] = Field(default_factory=list)
    low_spending_periods: List[LowSpendingPeriodDto] = Field(default_factory=list)
    income_patterns: Optional[IncomePatternDto] = None
    saving_suggestions: List[SavingSuggestionDto] = Field(default_factory=list)
    average_monthly_spending: float = 0.0
    insights: List[str] = Field(default_factory=list)
    status: str = Field(
        default="sufficient_data",
        description="Status: sufficient_data, insufficient_data, error"
    )


class AnomalyResponse(BaseModel):
    """Response model for anomaly detection."""
    anomalies: List[AnomalyDto] = Field(default_factory=list)
    messages: List[str] = Field(default_factory=list)
    status: str = Field(
        default="sufficient_data",
        description="Status: sufficient_data, insufficient_data, no_recent_data, no_historical_data"
    )
    statusMessage: str = Field(default="", description="Human-readable status message")
    
    class Config:
        schema_extra = {
            "example": {
                "anomalies": [
                    {
                        "category_id": "123e4567-e89b-12d3-a456-426614174000",
                        "category_name": "Restaurants",
                        "current_amount": 500.0,
                        "average_historical": 200.0,
                        "percentage_change": 150.0,
                        "period_days": 7,
                        "is_increase": True,
                        "anomaly_score": -0.5,
                        "budget_limit": 300.0,
                        "budget_spent": 450.0,
                        "budget_overage": 150.0,
                        "budget_percentage_used": 150.0,
                        "recommendation": "Consider reducing dining out by $21.43 per day. Try meal prepping, cooking at home more often, or choosing more budget-friendly restaurants. You're currently 50% over your $300.00 budget."
                    }
                ],
                "messages": [
                    "This week you spent 150% more on Restaurants than usual. You've exceeded your Restaurants budget by 50% ($150.00 over)."
                ],
                "status": "sufficient_data",
                "statusMessage": ""
            }
        }


@router.get(
    "/status",
    summary="Service health status",
    description="Returns the operational status and metrics of the AI service"
)
async def ai_status():
    """Get service status and metrics."""
    stats = metrics.get_stats()
    return {
        "status": "operational",
        "service": "AI Service",
        "version": settings.VERSION,
        "features": {
            "anomaly_detection": "available",
        },
        "metrics": stats,
        "configuration": {
            "contamination": settings.ANOMALY_DETECTION_CONTAMINATION,
            "percentage_threshold": settings.ANOMALY_PERCENTAGE_THRESHOLD,
            "std_threshold": settings.ANOMALY_STD_THRESHOLD,
            "historical_days": settings.ANOMALY_HISTORICAL_DAYS,
        }
    }


@router.post(
    "/detect-anomalies", 
    response_model=AnomalyResponse,
    summary="Detect spending anomalies",
    description="Analyzes user transactions to detect unusual spending patterns using ML",
    responses={
        200: {"description": "Anomaly detection completed successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"}
    }
)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Detect spending anomalies for a user.
    
    Analyzes the user's expense transactions over the last 90 days and compares
    recent spending (last N days) against historical patterns to identify anomalies.
    
    Args:
        request: AnomalyDetectionRequest with user_id and period_days
        db: Database session
    
    Returns:
        AnomalyResponse with detected anomalies, messages, and status
    """
    start_time = datetime.utcnow()
    user_id = None
    
    try:
        # Validate and parse user_id
        try:
            user_id = UUID(request.user_id)
        except ValueError:
            logger.warning(f"Invalid user_id format: {request.user_id}")
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Invalid user_id format. Must be a valid UUID."
            )
        
        logger.info(
            f"Starting anomaly detection for user {user_id}, "
            f"period_days={request.period_days}"
        )
        
        # Calculate cutoff date
        cutoff_date = datetime.utcnow() - timedelta(days=settings.ANOMALY_HISTORICAL_DAYS)
        
        # Query transactions
        try:
            result = await db.execute(
                select(Transaction, TransactionCategory)
                .join(TransactionCategory, Transaction.CategoryId == TransactionCategory.Id)
                .where(Transaction.CreatedBy == user_id)
                .where(Transaction.Date >= cutoff_date)
                .where(Transaction.Type == 2)  # Only expenses
            )
        except Exception as e:
            logger.error(f"Database query failed for user {user_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve transaction data"
            )
        
        # Transform transactions with error handling
        transactions_data = []
        for transaction, category in result.all():
            try:
                if not transaction or not category:
                    logger.debug("Skipping transaction with missing data")
                    continue
                
                # Validate required fields
                if not transaction.CategoryId or not transaction.Amount:
                    logger.debug(f"Skipping transaction {transaction.Id} with missing required fields")
                    continue
                
                amount = float(transaction.Amount)
                if amount <= 0:
                    logger.debug(f"Skipping transaction {transaction.Id} with non-positive amount")
                    continue
                
                if not transaction.Date:
                    logger.debug(f"Skipping transaction {transaction.Id} with missing date")
                    continue
                
                transactions_data.append({
                    'Id': str(transaction.Id),
                    'CategoryId': str(transaction.CategoryId),
                    'Amount': amount,
                    'Date': transaction.Date,
                    'Type': transaction.Type,
                    'Category': {
                        'Id': str(category.Id),
                        'Name': category.Name if category.Name else 'Unknown',
                    }
                })
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Error processing transaction {transaction.Id if transaction else 'unknown'}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing transaction: {e}", exc_info=True)
                continue
        
        logger.info(f"Retrieved {len(transactions_data)} transactions for user {user_id}")
        
        # Fetch budget data for the current period
        budget_data = None
        try:
            budget_data = await _fetch_budget_data(
                db, user_id, request.period_days
            )
            if budget_data:
                logger.info(f"Retrieved budget data for {len(budget_data)} categories")
        except Exception as e:
            logger.warning(f"Failed to fetch budget data for user {user_id}: {e}")
            # Continue without budget data - anomaly detection still works
            budget_data = None
        
        # Detect anomalies
        try:
            detector = SpendingAnomalyDetector()
            anomalies_raw, status = detector.detect_anomalies(
                transactions_data, 
                request.period_days,
                budget_data=budget_data
            )
        except Exception as e:
            logger.error(f"Anomaly detection failed for user {user_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Anomaly detection failed"
            )
        
        # Format messages
        messages = []
        for anomaly in anomalies_raw:
            try:
                message = detector.format_message_english(anomaly)
                if message:
                    messages.append(message)
            except Exception as e:
                logger.warning(f"Error formatting message for anomaly: {e}")
                # Add a fallback message
                messages.append(
                    f"Unusual spending detected in {anomaly.get('category_name', 'Unknown category')}"
                )
        
        # Convert to response DTOs with error handling
        anomalies = []
        for anomaly in anomalies_raw:
            try:
                anomalies.append(AnomalyDto(**anomaly))
            except Exception as e:
                logger.warning(f"Error converting anomaly to DTO: {e}")
                # Skip invalid anomalies
                continue
        
        # Status messages
        status_messages = {
            'sufficient_data': '',
            'insufficient_data': (
                'Not enough transaction data to analyze. '
                'Add more transactions to enable anomaly detection.'
            ),
            'no_recent_data': (
                f'No recent transactions found. '
                f'Add transactions from the last {request.period_days} days to detect anomalies.'
            ),
            'no_historical_data': (
                'Not enough historical data to compare against. '
                'The system needs at least 2 weeks of transaction history.'
            )
        }
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Record metrics
        metrics.record_detection(
            user_id=str(user_id),
            anomalies_count=len(anomalies),
            processing_time=processing_time,
            status=status,
            success=True
        )
        
        logger.info(
            f"Anomaly detection completed for user {user_id}: "
            f"{len(anomalies)} anomalies found, status={status}, "
            f"processing_time={processing_time:.2f}s"
        )
        
        return AnomalyResponse(
            anomalies=anomalies,
            messages=messages,
            status=status,
            statusMessage=status_messages.get(status, '')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Record failed metrics
        if user_id:
            metrics.record_detection(
                user_id=str(user_id),
                anomalies_count=0,
                processing_time=processing_time,
                status='error',
                success=False
            )
        
        logger.error(
            f"Unexpected error in anomaly detection for user {user_id}: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during anomaly detection"
        )


async def _fetch_budget_data(
    db: AsyncSession,
    user_id: UUID,
    period_days: int
) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Fetch budget data for the current period.
    
    Args:
        db: Database session
        user_id: User UUID
        period_days: Number of days in the analysis period
    
    Returns:
        Dict mapping category_id to budget info, or None if no budget found or error occurs
    """
    try:
        if not user_id:
            logger.warning("_fetch_budget_data called with None user_id")
            return None
        
        now = datetime.utcnow()
        
        # Query budget for current period (active budget)
        try:
            budget_query = select(Budget).where(
                Budget.CreatedBy == user_id
            ).where(
                Budget.StartDate <= now
            ).where(
                Budget.EndDate >= now
            )
            
            budget_result = await db.execute(budget_query)
            budget = budget_result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Database error querying budget for user {user_id}: {e}", exc_info=True)
            return None
        
        if not budget:
            logger.debug(f"No active budget found for user {user_id}")
            return None
        
        # Validate budget dates
        if not budget.StartDate or not budget.EndDate:
            logger.warning(f"Budget {budget.Id} has invalid dates")
            return None
        
        if budget.EndDate < budget.StartDate:
            logger.warning(f"Budget {budget.Id} has end date before start date")
            return None
        
        # Query budget details with categories
        try:
            details_query = select(BudgetDetail, TransactionCategory).join(
                TransactionCategory, BudgetDetail.CategoryId == TransactionCategory.Id
            ).where(
                BudgetDetail.BudgetId == budget.Id
            )
            
            details_result = await db.execute(details_query)
        except Exception as e:
            logger.error(f"Database error querying budget details: {e}", exc_info=True)
            return None
        
        budget_data = {}
        try:
            budget_period_days = (budget.EndDate - budget.StartDate).days + 1
            if budget_period_days <= 0:
                logger.warning(f"Invalid budget period days: {budget_period_days}")
                return None
        except Exception as e:
            logger.error(f"Error calculating budget period: {e}")
            return None
        
        for detail, category in details_result.all():
            try:
                if not detail or not category:
                    continue
                
                category_id = str(detail.CategoryId)
                if not category_id or category_id == 'None':
                    logger.debug("Skipping budget detail with invalid category_id")
                    continue
                
                limit = float(detail.Limit)
                spent = float(detail.Spent)
                
                # Validate values
                if limit < 0:
                    logger.warning(f"Negative budget limit for category {category_id}: {limit}")
                    continue
                
                if spent < 0:
                    logger.warning(f"Negative budget spent for category {category_id}: {spent}")
                    spent = 0  # Correct negative values
                
                budget_data[category_id] = {
                    'limit': limit,
                    'spent': spent,
                    'period_days': budget_period_days
                }
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Error processing budget detail: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing budget detail: {e}", exc_info=True)
                continue
        
        if not budget_data:
            logger.debug(f"No valid budget details found for budget {budget.Id}")
            return None
        
        return budget_data
        
    except Exception as e:
        logger.error(f"Unexpected error fetching budget data for user {user_id}: {e}", exc_info=True)
        return None


@router.post(
    "/predict-spending-periods",
    response_model=SpendingPredictionResponse,
    summary="Predict high spending periods",
    description="Analyzes historical spending to predict future high-spending periods and identify patterns",
    responses={
        200: {"description": "Prediction completed successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"}
    }
)
async def predict_spending_periods(
    request: SpendingPredictionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Predict high-spending periods for a user.
    
    Analyzes historical transaction data to:
    - Identify seasonal spending patterns
    - Detect recurring high-spending events
    - Predict future high-spending months
    
    Args:
        request: SpendingPredictionRequest with user_id and forecast_months
        db: Database session
    
    Returns:
        SpendingPredictionResponse with predictions, patterns, and insights
    """
    start_time = datetime.utcnow()
    user_id = None
    
    try:
        # Validate and parse user_id
        try:
            user_id = UUID(request.user_id)
        except ValueError:
            logger.warning(f"Invalid user_id format: {request.user_id}")
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="Invalid user_id format. Must be a valid UUID."
            )
        
        logger.info(
            f"Starting spending prediction for user {user_id}, "
            f"forecast_months={request.forecast_months}"
        )
        
        # Query transactions - get at least 12 months of history (both income and expenses)
        cutoff_date = datetime.utcnow() - timedelta(days=365)
        
        try:
            result = await db.execute(
                select(Transaction, TransactionCategory)
                .join(TransactionCategory, Transaction.CategoryId == TransactionCategory.Id)
                .where(Transaction.CreatedBy == user_id)
                .where(Transaction.Date >= cutoff_date)
                # Include both income (Type == 1) and expenses (Type == 2)
                .order_by(Transaction.Date)
            )
        except Exception as e:
            logger.error(f"Database query failed for user {user_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve transaction data"
            )
        
        # Transform transactions
        transactions_data = []
        skipped_count = 0
        for transaction, category in result.all():
            try:
                if not transaction or not category:
                    skipped_count += 1
                    continue
                
                if not transaction.CategoryId or not transaction.Amount or not transaction.Date:
                    skipped_count += 1
                    continue
                
                amount = float(transaction.Amount)
                if amount <= 0:
                    skipped_count += 1
                    continue
                
                # Ensure Date is a datetime object
                date = transaction.Date
                if not isinstance(date, datetime):
                    # Try to convert if it's a string or other type
                    if isinstance(date, str):
                        try:
                            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
                        except (ValueError, AttributeError):
                            logger.warning(f"Could not parse date string: {date}")
                            skipped_count += 1
                            continue
                    elif hasattr(date, 'year') and hasattr(date, 'month'):
                        # SQLAlchemy DateTime or similar - convert to Python datetime
                        try:
                            date = datetime(date.year, date.month, date.day if hasattr(date, 'day') else 1,
                                          date.hour if hasattr(date, 'hour') else 0,
                                          date.minute if hasattr(date, 'minute') else 0,
                                          date.second if hasattr(date, 'second') else 0)
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert date object: {type(date)}")
                            skipped_count += 1
                            continue
                    else:
                        logger.warning(f"Transaction {transaction.Id} has invalid date type: {type(date)}")
                        skipped_count += 1
                        continue
                
                transactions_data.append({
                    'Id': str(transaction.Id),
                    'CategoryId': str(transaction.CategoryId),
                    'Amount': amount,
                    'Date': date,
                    'Type': transaction.Type,
                    'Category': {
                        'Id': str(category.Id),
                        'Name': category.Name if category.Name else 'Unknown',
                    }
                })
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Error processing transaction: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing transaction: {e}", exc_info=True)
                skipped_count += 1
                continue
        
        logger.info(f"Retrieved {len(transactions_data)} transactions for user {user_id} (skipped {skipped_count} invalid transactions)")
        
        # Log date range for debugging
        if transactions_data:
            dates = [t['Date'] for t in transactions_data if isinstance(t.get('Date'), datetime)]
            if dates:
                min_date = min(dates)
                max_date = max(dates)
                logger.info(f"Transaction date range: {min_date} to {max_date}")
                
                # Count unique months
                unique_months = len(set((d.year, d.month) for d in dates))
                logger.info(f"Unique months in transaction data: {unique_months}")
        
        # Predict spending periods (include income analysis)
        try:
            predictor = SpendingPeriodPredictor()
            prediction_result = predictor.predict_high_spending_periods(
                transactions_data,
                request.forecast_months,
                include_income=True
            )
        except Exception as e:
            logger.error(f"Spending prediction failed for user {user_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Spending prediction failed"
            )
        
        # Convert to response DTOs
        predictions = [
            PredictionDto(**p) for p in prediction_result.get('predictions', [])
        ]
        
        seasonal_patterns = [
            SeasonalPatternDto(**p) for p in prediction_result.get('seasonal_patterns', [])
        ]
        
        recurring_events = [
            RecurringEventDto(**e) for e in prediction_result.get('recurring_events', [])
        ]
        
        historical_high_months = [
            HistoricalHighMonthDto(**h) for h in prediction_result.get('historical_high_months', [])
        ]
        
        low_spending_periods = [
            LowSpendingPeriodDto(**l) for l in prediction_result.get('low_spending_periods', [])
        ]
        
        income_patterns = None
        if prediction_result.get('income_patterns'):
            ip = prediction_result['income_patterns']
            income_patterns = IncomePatternDto(
                average_monthly_income=ip.get('average_monthly_income', 0.0),
                average_monthly_spending=ip.get('average_monthly_spending', 0.0),
                average_surplus=ip.get('average_surplus', 0.0),
                high_income_months=ip.get('high_income_months', [])
            )
        
        saving_suggestions = [
            SavingSuggestionDto(**s) for s in prediction_result.get('saving_suggestions', [])
        ]
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(
            f"Spending prediction completed for user {user_id}: "
            f"{len(predictions)} predictions, {len(seasonal_patterns)} patterns, "
            f"{len(saving_suggestions)} saving suggestions, "
            f"processing_time={processing_time:.2f}s"
        )
        
        return SpendingPredictionResponse(
            predictions=predictions,
            seasonal_patterns=seasonal_patterns,
            recurring_events=recurring_events,
            historical_high_months=historical_high_months,
            low_spending_periods=low_spending_periods,
            income_patterns=income_patterns,
            saving_suggestions=saving_suggestions,
            average_monthly_spending=prediction_result.get('average_monthly_spending', 0.0),
            insights=prediction_result.get('insights', []),
            status=prediction_result.get('status', 'sufficient_data')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.error(
            f"Unexpected error in spending prediction for user {user_id}: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during spending prediction"
        )


@router.post(
    "/classify-pattern",
    response_model=SpendingPatternResponse,
    summary="Classify spending behavior",
    description="Classifies user spending behavior using a rule + ML hybrid model"
)
async def classify_spending_pattern(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    start_time = datetime.utcnow()

    try:
        user_uuid = UUID(user_id)

        # Fetch transactions (same logic as anomaly detector)
        result = await db.execute(
            select(Transaction)
            .where(Transaction.CreatedBy == user_uuid)
            .where(Transaction.Type == 2)  # expenses only
        )

        transactions = [
            {"Amount": float(t.Amount)}
            for t in result.scalars().all()
        ]

        if not transactions:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="No transactions available for classification"
            )

        classifier = SpendingPatternClassifier()
        output = classifier.classify(transactions)

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        metrics.record_detection(
            user_id=str(user_uuid),
            anomalies_count=0,
            processing_time=processing_time,
            status="pattern_classification",
            success=True
        )

        return SpendingPatternResponse(**output)

    except ValueError:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Invalid user_id"
        )
