"""
Production-ready Spending Anomaly Detector for ExpenseAlly.

Uses a hybrid approach combining statistical methods and Isolation Forest
for robust anomaly detection in financial transaction data.
Now includes budget comparison and spending recommendations.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
from sklearn.ensemble import IsolationForest

from app.core.config import settings

logger = logging.getLogger(__name__)


class SpendingAnomalyDetector:
    """
    Detects spending anomalies using statistical analysis and machine learning.
    
    Uses a hybrid approach:
    1. Statistical method (Z-score) for primary detection
    2. Isolation Forest for additional validation
    3. Percentage-based threshold as fallback
    """
    
    def __init__(self, contamination: Optional[float] = None):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (0.0 to 0.5).
                          If None, uses value from settings.
        """
        self.contamination = contamination or settings.ANOMALY_DETECTION_CONTAMINATION
        self.percentage_threshold = settings.ANOMALY_PERCENTAGE_THRESHOLD
        self.std_threshold = settings.ANOMALY_STD_THRESHOLD
        self.min_transactions = settings.ANOMALY_MIN_TRANSACTIONS
        self.min_historical_periods = settings.ANOMALY_MIN_HISTORICAL_PERIODS
        
        logger.info(
            f"Initialized SpendingAnomalyDetector with contamination={self.contamination}, "
            f"percentage_threshold={self.percentage_threshold}%, std_threshold={self.std_threshold}"
        )
    
    def detect_anomalies(
        self,
        transactions: List[Dict[str, Any]],
        current_period_days: int = 7,
        budget_data: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Detect spending anomalies in transaction data.
        
        Args:
            transactions: List of transaction dictionaries with keys:
                         - 'Date': datetime object
                         - 'Amount': float
                         - 'Type': int (2 for expenses)
                         - 'CategoryId': str
                         - 'Category': dict with 'Name' key
            current_period_days: Number of days in the current period to analyze
            budget_data: Optional dict mapping category_id to budget info:
                        {category_id: {'limit': float, 'spent': float, 'period_days': int}}
        
        Returns:
            Tuple of (anomalies_list, status):
            - anomalies_list: List of detected anomalies with details including budget insights
            - status: One of 'sufficient_data', 'insufficient_data', 
                     'no_recent_data', 'no_historical_data'
        """
        try:
            # Input validation
            if not transactions:
                logger.warning("Empty transactions list provided")
                return [], 'insufficient_data'
            
            if len(transactions) < self.min_transactions:
                logger.info(
                    f"Insufficient transactions: {len(transactions)} < {self.min_transactions}"
                )
                return [], 'insufficient_data'
            
            # Validate period_days
            if current_period_days < 1 or current_period_days > 90:
                logger.warning(f"Invalid period_days: {current_period_days}, using default 7")
                current_period_days = 7
            
            now = datetime.utcnow()
            current_start = now - timedelta(days=current_period_days)
            
            # Separate current and historical transactions
            current_transactions, historical_transactions = self._split_transactions(
                transactions, current_start
            )
            
            if not current_transactions:
                logger.info(f"No transactions found in the last {current_period_days} days")
                return [], 'no_recent_data'
            
            if not historical_transactions:
                logger.info("No historical transactions found for comparison")
                return [], 'no_historical_data'
            
            # Group transactions by category
            current_by_category = self._group_by_category(current_transactions)
            historical_data = self._prepare_historical_data(
                historical_transactions, current_period_days
            )
            
            if not historical_data:
                logger.info("No historical data prepared for any category")
                return [], 'no_historical_data'
            
            # Detect anomalies for each category
            anomalies = []
            has_sufficient_data = False
            
            for category_id, current_amount in current_by_category.items():
                if category_id not in historical_data:
                    logger.debug(f"No historical data for category {category_id}")
                    continue
                
                category_history = historical_data[category_id]
                
                if len(category_history) < self.min_historical_periods:
                    logger.debug(
                        f"Insufficient periods for category {category_id}: "
                        f"{len(category_history)} < {self.min_historical_periods}"
                    )
                    continue
                
                has_sufficient_data = True
                
                # Calculate statistics
                avg_historical = np.mean(category_history)
                percentage_change = (
                    ((current_amount - avg_historical) / avg_historical) * 100 
                    if avg_historical > 0 else 0
                )
                
                # Check for significant percentage change
                significant_change = abs(percentage_change) > self.percentage_threshold
                
                # Use ML detection if we have enough data
                if len(category_history) >= 3:
                    is_anomaly, score = self._detect_with_isolation_forest(
                        category_history, 
                        current_amount
                    )
                else:
                    # For 2 periods, use simple comparison
                    is_anomaly = significant_change
                    score = abs(percentage_change) / 10
                
                # Check budget if available
                budget_info = None
                budget_limit = None
                budget_spent = None
                budget_overage = None
                budget_percentage_used = None
                recommendation = None
                
                try:
                    if budget_data and category_id in budget_data:
                        budget_info = budget_data[category_id]
                        
                        # Validate budget info structure
                        if not isinstance(budget_info, dict):
                            logger.warning(f"Invalid budget_info type for category {category_id}")
                        else:
                            budget_limit = budget_info.get('limit', 0)
                            budget_spent = budget_info.get('spent', 0)
                            budget_period_days = budget_info.get('period_days', current_period_days)
                            
                            # Validate budget values
                            try:
                                budget_limit = float(budget_limit) if budget_limit is not None else 0
                                budget_spent = float(budget_spent) if budget_spent is not None else 0
                                budget_period_days = int(budget_period_days) if budget_period_days is not None else current_period_days
                                
                                if budget_limit < 0:
                                    logger.warning(f"Negative budget limit for category {category_id}")
                                    budget_limit = 0
                                
                                if budget_spent < 0:
                                    logger.warning(f"Negative budget spent for category {category_id}")
                                    budget_spent = 0
                                
                                if budget_period_days <= 0:
                                    logger.warning(f"Invalid budget_period_days: {budget_period_days}")
                                    budget_period_days = current_period_days
                                
                                # Calculate projected spending for the period
                                if budget_period_days > 0 and current_period_days > 0:
                                    # Project current spending to match budget period
                                    try:
                                        projected_spending = (current_amount / current_period_days) * budget_period_days
                                        
                                        if budget_limit > 0:
                                            budget_overage = max(0, projected_spending - budget_limit)
                                            budget_percentage_used = (projected_spending / budget_limit * 100)
                                        else:
                                            budget_percentage_used = 0
                                        
                                        # Generate recommendation if over budget
                                        if projected_spending > budget_limit and budget_limit > 0:
                                            try:
                                                overage_percentage = ((projected_spending - budget_limit) / budget_limit) * 100
                                                recommendation = self._generate_recommendation(
                                                    category_name,
                                                    budget_limit,
                                                    projected_spending,
                                                    overage_percentage,
                                                    budget_period_days
                                                )
                                            except Exception as e:
                                                logger.warning(f"Error generating recommendation: {e}")
                                                recommendation = f"Consider reducing spending on {category_name} to stay within your ${budget_limit:.2f} budget."
                                    except (ZeroDivisionError, ValueError) as e:
                                        logger.warning(f"Error calculating projected spending: {e}")
                                        budget_percentage_used = None
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Error converting budget values for category {category_id}: {e}")
                                budget_info = None
                except Exception as e:
                    logger.error(f"Unexpected error processing budget for category {category_id}: {e}", exc_info=True)
                    # Continue without budget data
                    budget_info = None
                
                # Flag as anomaly if detected by ML OR significant percentage change OR over budget
                is_over_budget = budget_info and budget_limit and budget_percentage_used and budget_percentage_used > 100
                should_flag = is_anomaly or significant_change or is_over_budget
                
                if should_flag:
                    category_name = self._get_category_name(transactions, category_id)
                    
                    anomaly = {
                        'category_id': str(category_id),
                        'category_name': category_name,
                        'current_amount': float(current_amount),
                        'average_historical': float(avg_historical),
                        'percentage_change': round(percentage_change, 1),
                        'period_days': current_period_days,
                        'is_increase': percentage_change > 0,
                        'anomaly_score': float(score) if is_anomaly else abs(percentage_change) / 10,
                        'budget_limit': float(budget_limit) if budget_limit else None,
                        'budget_spent': float(budget_spent) if budget_spent else None,
                        'budget_overage': float(budget_overage) if budget_overage else None,
                        'budget_percentage_used': round(budget_percentage_used, 1) if budget_percentage_used else None,
                        'recommendation': recommendation
                    }
                    
                    anomalies.append(anomaly)
                    log_msg = (
                        f"Anomaly detected: {category_name} - "
                        f"{percentage_change:+.1f}% change "
                        f"(current: ${current_amount:.2f}, avg: ${avg_historical:.2f})"
                    )
                    if is_over_budget:
                        log_msg += f" | OVER BUDGET: {budget_percentage_used:.1f}% used"
                    logger.info(log_msg)
            
            status = 'sufficient_data' if has_sufficient_data else 'insufficient_data'
            
            logger.info(
                f"Anomaly detection complete: {len(anomalies)} anomalies found, "
                f"status: {status}"
            )
            
            return sorted(anomalies, key=lambda x: abs(x['anomaly_score']), reverse=True), status
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}", exc_info=True)
            # Return empty result on error rather than crashing
            return [], 'insufficient_data'
    
    def _split_transactions(
        self, 
        transactions: List[Dict[str, Any]], 
        current_start: datetime
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split transactions into current and historical periods."""
        current_transactions = []
        historical_transactions = []
        
        for t in transactions:
            date = t.get('Date')
            if not isinstance(date, datetime):
                logger.debug(f"Skipping transaction with invalid date: {t.get('Id', 'unknown')}")
                continue
            
            # Remove timezone info if present for comparison
            if date.tzinfo is not None:
                date = date.replace(tzinfo=None)
            
            if date >= current_start:
                current_transactions.append(t)
            else:
                historical_transactions.append(t)
        
        return current_transactions, historical_transactions
    
    def _detect_with_isolation_forest(
        self, 
        historical_values: List[float], 
        current_value: float
    ) -> Tuple[bool, float]:
        """
        Detect anomaly using statistical method and Isolation Forest.
        
        Returns:
            Tuple of (is_anomaly, score)
        """
        if len(historical_values) < 2:
            return False, 0.0
        
        try:
            # Statistical method as primary detection
            mean = np.mean(historical_values)
            std = np.std(historical_values)
            
            if std == 0:
                # No variance - use percentage threshold
                threshold = mean * (self.percentage_threshold / 100)
                is_statistical_anomaly = abs(current_value - mean) > threshold
                score = abs(current_value - mean) / (mean + 1e-6)
            else:
                # Use Z-score with configurable threshold
                z_score = abs(current_value - mean) / std
                is_statistical_anomaly = z_score > self.std_threshold
                score = z_score
            
            # Isolation Forest for additional validation
            contamination_rate = (
                max(self.contamination, 0.25) 
                if len(historical_values) < 10 
                else self.contamination
            )
            
            is_forest_anomaly = False
            forest_score = 0.0
            
            try:
                data = np.array(historical_values + [current_value]).reshape(-1, 1)
                model = IsolationForest(
                    contamination=contamination_rate,
                    random_state=42,
                    n_estimators=100
                )
                model.fit(data[:-1])
                prediction = model.predict([data[-1]])
                is_forest_anomaly = prediction[0] == -1
                forest_score = model.score_samples([data[-1]])[0]
            except Exception as e:
                logger.warning(f"Isolation Forest failed: {e}, using statistical method only")
            
            # Anomaly if either method detects it
            is_anomaly = is_statistical_anomaly or is_forest_anomaly
            
            # Use the more negative score (more anomalous)
            final_score = min(score, forest_score) if is_forest_anomaly else score
            
            return is_anomaly, final_score
            
        except Exception as e:
            logger.error(f"Error in _detect_with_isolation_forest: {e}", exc_info=True)
            return False, 0.0
    
    def _group_by_category(self, transactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Group transactions by category and sum amounts."""
        grouped = defaultdict(float)
        for t in transactions:
            if t.get('Type') == 2:  # Only expenses
                try:
                    category_id = str(t.get('CategoryId', ''))
                    amount = float(t.get('Amount', 0))
                    if amount > 0:  # Validate positive amount
                        grouped[category_id] += amount
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error processing transaction: {e}")
                    continue
        return dict(grouped)
    
    def _prepare_historical_data(
        self, 
        transactions: List[Dict[str, Any]], 
        period_days: int
    ) -> Dict[str, List[float]]:
        """
        Prepare historical data grouped by category and time period.
        
        For weekly periods (7 days), groups by week.
        For other periods, groups by day.
        """
        period_amounts = defaultdict(lambda: defaultdict(float))
        
        for t in transactions:
            if t.get('Type') != 2:  # Only expenses
                continue
            
            date = t.get('Date')
            if not isinstance(date, datetime):
                continue
            
            # Remove timezone if present
            if date.tzinfo is not None:
                date = date.replace(tzinfo=None)
            
            try:
                category_id = str(t.get('CategoryId', ''))
                amount = float(t.get('Amount', 0))
                
                if amount <= 0:
                    continue
                
                # Group by week for 7-day periods, by day for others
                if period_days == 7:
                    # Start of week (Monday)
                    period_start = (date - timedelta(days=date.weekday())).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                else:
                    period_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
                
                period_key = period_start.isoformat()
                period_amounts[category_id][period_key] += amount
                
            except (ValueError, TypeError) as e:
                logger.debug(f"Error processing transaction for historical data: {e}")
                continue
        
        # Convert to list of values per category
        result = {}
        for category_id, periods in period_amounts.items():
            if len(periods) >= self.min_historical_periods:
                result[category_id] = list(periods.values())
        
        return result
    
    def _get_category_name(self, transactions: List[Dict[str, Any]], category_id: str) -> str:
        """Extract category name from transactions."""
        for t in transactions:
            if str(t.get('CategoryId', '')) == category_id:
                category = t.get('Category')
                if category and isinstance(category, dict):
                    return category.get('Name', 'Unknown')
                elif hasattr(category, 'Name'):
                    return category.Name
        return 'Unknown'
    
    def _generate_recommendation(
        self,
        category_name: str,
        budget_limit: float,
        projected_spending: float,
        overage_percentage: float,
        period_days: int
    ) -> str:
        """
        Generate actionable recommendation for overspending.
        
        Args:
            category_name: Name of the category
            budget_limit: Budget limit for the period
            projected_spending: Projected spending amount
            overage_percentage: Percentage over budget
            period_days: Number of days in the budget period
        
        Returns:
            Recommendation string
        """
        overage_amount = projected_spending - budget_limit
        daily_budget = budget_limit / period_days if period_days > 0 else budget_limit
        daily_spending = projected_spending / period_days if period_days > 0 else projected_spending
        daily_reduction_needed = daily_spending - daily_budget
        
        # Generate category-specific recommendations
        category_lower = category_name.lower()
        
        if 'dining' in category_lower or 'restaurant' in category_lower or 'food' in category_lower:
            return (
                f"Consider reducing dining out by ${daily_reduction_needed:.2f} per day. "
                f"Try meal prepping, cooking at home more often, or choosing more budget-friendly restaurants. "
                f"You're currently {overage_percentage:.0f}% over your ${budget_limit:.2f} budget."
            )
        elif 'entertainment' in category_lower or 'recreation' in category_lower:
            return (
                f"Reduce entertainment spending by ${daily_reduction_needed:.2f} per day. "
                f"Look for free or low-cost activities, use streaming services you already have, "
                f"or limit outings. You're {overage_percentage:.0f}% over budget."
            )
        elif 'shopping' in category_lower or 'retail' in category_lower:
            return (
                f"Cut shopping expenses by ${daily_reduction_needed:.2f} per day. "
                f"Create a 24-hour waiting period before non-essential purchases, "
                f"use wishlists, and look for sales. You're {overage_percentage:.0f}% over budget."
            )
        elif 'transport' in category_lower or 'gas' in category_lower or 'fuel' in category_lower:
            return (
                f"Reduce transportation costs by ${daily_reduction_needed:.2f} per day. "
                f"Consider carpooling, using public transport, or combining errands. "
                f"You're {overage_percentage:.0f}% over your ${budget_limit:.2f} budget."
            )
        else:
            return (
                f"Reduce spending on {category_name} by ${daily_reduction_needed:.2f} per day "
                f"to stay within your ${budget_limit:.2f} budget. "
                f"You're currently {overage_percentage:.0f}% over budget."
            )
    
    def format_message_english(self, anomaly: Dict[str, Any]) -> str:
        """
        Format a user-friendly message for an anomaly.
        
        Args:
            anomaly: Dictionary containing anomaly details
        
        Returns:
            Formatted message string
        """
        period_text = "week" if anomaly['period_days'] == 7 else f"{anomaly['period_days']} days"
        category = anomaly['category_name']
        percentage = abs(anomaly['percentage_change'])
        
        messages = []
        
        # Add historical comparison message if significant
        if abs(percentage) > self.percentage_threshold:
            if anomaly['is_increase']:
                messages.append(f"This {period_text} you spent {percentage:.0f}% more on {category} than usual.")
            else:
                messages.append(f"This {period_text} you spent {percentage:.0f}% less on {category} than usual.")
        
        # Add budget message if over budget
        if anomaly.get('budget_percentage_used') and anomaly['budget_percentage_used'] > 100:
            budget_pct = anomaly['budget_percentage_used']
            messages.append(
                f"You've exceeded your {category} budget by {budget_pct - 100:.0f}% "
                f"(${anomaly.get('budget_overage', 0):.2f} over)."
            )
        elif anomaly.get('budget_percentage_used') and anomaly['budget_percentage_used'] > 80:
            budget_pct = anomaly['budget_percentage_used']
            messages.append(
                f"You've used {budget_pct:.0f}% of your {category} budget. "
                f"Consider reducing spending to stay within budget."
            )
        
        return " ".join(messages) if messages else f"Unusual spending pattern detected in {category}."
