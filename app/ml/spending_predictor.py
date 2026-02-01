"""
Spending Period Predictor for ExpenseAlly.

Analyzes historical spending patterns to predict high-spending periods,
identify seasonal trends, and detect recurring events that lead to increased spending.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
from calendar import month_name

from app.core.config import settings

logger = logging.getLogger(__name__)


class SpendingPeriodPredictor:
    """
    Predicts high-spending periods by analyzing historical patterns.
    
    Features:
    1. Monthly spending pattern analysis
    2. Seasonal trend detection
    3. Recurring event identification
    4. Future period predictions
    """
    
    # Common high-spending events/holidays
    # Format: (start_month, start_day, end_month, end_day) for multi-month
    # Format: (month, start_day, end_day) for single-month
    HIGH_SPENDING_EVENTS = {
        'Christmas': (12, 20, 12, 31),  # Dec 20 - Dec 31
        'New Year': (12, 28, 1, 1),  # Dec 28 - Jan 1 (cross-year)
        'Valentine\'s Day': (2, 10, 2, 16),  # Feb 10 - Feb 16
        'Easter': None,  # Variable date, handled separately
        'Summer Vacation': (6, 1, 8, 31),  # June 1 - August 31
        'Back to School': (8, 15, 9, 15),  # Aug 15 - Sep 15
        'Black Friday': (11, 20, 11, 30),  # Nov 20 - Nov 30
        'Holiday Season': (11, 1, 12, 31),  # Nov 1 - Dec 31
    }
    
    def __init__(self):
        """Initialize the spending predictor."""
        self.min_months_history = 6  # Need at least 6 months of data
        self.high_spending_threshold = 1.2  # 20% above average
        self.seasonal_threshold = 0.15  # 15% variation for seasonal pattern
        
        logger.info("Initialized SpendingPeriodPredictor")
    
    def predict_high_spending_periods(
        self,
        transactions: List[Dict[str, Any]],
        forecast_months: int = 6,
        include_income: bool = True
    ) -> Dict[str, Any]:
        """
        Predict high-spending periods based on historical data.
        
        Args:
            transactions: List of transaction dictionaries with keys:
                         - 'Date': datetime object
                         - 'Amount': float
                         - 'Type': int (1 for income, 2 for expenses)
                         - 'CategoryId': str
                         - 'Category': dict with 'Name' key
            forecast_months: Number of months to forecast ahead (default 6)
            include_income: Whether to analyze income for saving suggestions (default True)
        
        Returns:
            Dictionary containing:
            - predictions: List of predicted high-spending periods
            - seasonal_patterns: Detected seasonal trends
            - recurring_events: Identified recurring high-spending events
            - low_spending_periods: Periods with lower spending (saving opportunities)
            - income_patterns: Income patterns for saving suggestions
            - saving_suggestions: Recommendations for when to save
            - status: Analysis status
        """
        try:
            if not transactions:
                logger.warning("Empty transactions list provided")
                return self._empty_response("insufficient_data")
            
            # Separate expenses and income
            expenses = [t for t in transactions if t.get('Type') == 2]
            income = [t for t in transactions if t.get('Type') == 1] if include_income else []
            
            if len(expenses) < 10:
                logger.info(f"Insufficient transactions: {len(expenses)}")
                return self._empty_response("insufficient_data")
            
            # Group spending by month
            monthly_spending = self._group_by_month(expenses)
            
            logger.info(f"Grouped {len(expenses)} expense transactions into {len(monthly_spending)} unique months")
            
            if len(monthly_spending) < self.min_months_history:
                logger.warning(
                    f"Insufficient months of data: {len(monthly_spending)} < {self.min_months_history}. "
                    f"Months found: {sorted(monthly_spending.keys())}"
                )
                return self._empty_response("insufficient_data")
            
            # Calculate statistics
            spending_values = list(monthly_spending.values())
            avg_monthly = np.mean(spending_values)
            std_monthly = np.std(spending_values)
            
            # Identify historical high-spending months (limit to top 5 for pattern detection)
            high_spending_months = self._identify_high_spending_months(
                monthly_spending, avg_monthly, std_monthly
            )[:5]  # Limit to top 5 for pattern detection
            
            # Detect seasonal patterns
            seasonal_patterns = self._detect_seasonal_patterns(monthly_spending)
            
            # Identify recurring events (filtered to forecast period)
            all_recurring_events = self._identify_recurring_events(
                monthly_spending, expenses
            )
            # Filter to only events within forecast period
            now = datetime.utcnow()
            current_year = now.year
            current_month = now.month
            forecast_end_month = current_month + forecast_months
            forecast_end_year = current_year
            while forecast_end_month > 12:
                forecast_end_month -= 12
                forecast_end_year += 1
            
            recurring_events = [
                event for event in all_recurring_events
                if event.get('year') and event.get('month') and
                (event['year'] > current_year or (event['year'] == current_year and event['month'] >= current_month)) and
                (event['year'] < forecast_end_year or (event['year'] == forecast_end_year and event['month'] <= forecast_end_month))
            ][:3]  # Limit to top 3
            
            # Predict future high-spending periods
            predictions = self._predict_future_periods(
                monthly_spending,
                high_spending_months,
                seasonal_patterns,
                forecast_months
            )
            
            # Detect low-spending periods (saving opportunities) - filter to forecast period
            all_low_spending_periods = self._identify_low_spending_periods(
                monthly_spending, avg_monthly, std_monthly
            )
            # Filter to only future months within forecast period
            now = datetime.utcnow()
            current_year = now.year
            current_month = now.month
            forecast_end_month = current_month + forecast_months
            forecast_end_year = current_year
            while forecast_end_month > 12:
                forecast_end_month -= 12
                forecast_end_year += 1
            
            low_spending_periods = [
                lp for lp in all_low_spending_periods
                if (lp['year'] > current_year or (lp['year'] == current_year and lp['month'] >= current_month))
                and (lp['year'] < forecast_end_year or (lp['year'] == forecast_end_year and lp['month'] <= forecast_end_month))
            ][:3]  # Limit to top 3
            
            # Analyze income patterns if available
            income_patterns = None
            if include_income and income:
                income_patterns = self._analyze_income_patterns(income, monthly_spending, forecast_months)
            
            # Generate saving suggestions (filtered to forecast period)
            saving_suggestions = self._generate_saving_suggestions(
                low_spending_periods,
                seasonal_patterns,
                income_patterns,
                predictions,
                monthly_spending,
                forecast_months
            )
            
            # Generate insights (filtered to forecast period)
            insights = self._generate_insights(
                monthly_spending,
                seasonal_patterns,
                recurring_events,
                predictions,
                low_spending_periods,
                income_patterns,
                saving_suggestions,
                forecast_months
            )
            
            result = {
                'predictions': predictions,
                'seasonal_patterns': seasonal_patterns,
                'recurring_events': recurring_events,
                'historical_high_months': high_spending_months,
                'low_spending_periods': low_spending_periods,
                'average_monthly_spending': float(avg_monthly),
                'insights': insights,
                'status': 'sufficient_data'
            }
            
            if income_patterns:
                result['income_patterns'] = income_patterns
            if saving_suggestions:
                result['saving_suggestions'] = saving_suggestions
            
            return result
            
        except Exception as e:
            logger.error(f"Error in spending prediction: {e}", exc_info=True)
            return self._empty_response("error")
    
    def _group_by_month(
        self, 
        transactions: List[Dict[str, Any]]
    ) -> Dict[Tuple[int, int], float]:
        """
        Group transactions by year-month and sum amounts.
        
        Returns:
            Dict with (year, month) keys and total spending values
        """
        monthly = defaultdict(float)
        skipped_count = 0
        
        for t in transactions:
            try:
                date = t.get('Date')
                if date is None:
                    skipped_count += 1
                    continue
                
                # Convert to datetime if it's not already
                if not isinstance(date, datetime):
                    # Try to extract year/month from other date-like objects (SQLAlchemy DateTime, etc.)
                    if hasattr(date, 'year') and hasattr(date, 'month'):
                        try:
                            # Create a datetime from year/month/day
                            day = getattr(date, 'day', 1)
                            hour = getattr(date, 'hour', 0)
                            minute = getattr(date, 'minute', 0)
                            second = getattr(date, 'second', 0)
                            date = datetime(date.year, date.month, day, hour, minute, second)
                        except (ValueError, TypeError, AttributeError) as e:
                            logger.debug(f"Could not convert date object: {type(date)}, error: {e}")
                            skipped_count += 1
                            continue
                    elif isinstance(date, str):
                        # Try ISO format first
                        try:
                            date_str = date.replace('Z', '+00:00')
                            date = datetime.fromisoformat(date_str)
                        except (ValueError, AttributeError):
                            logger.debug(f"Could not parse date string with fromisoformat: {date}")
                            skipped_count += 1
                            continue
                    else:
                        logger.debug(f"Date is not a datetime and cannot be converted: {type(date)}")
                        skipped_count += 1
                        continue
                
                # Remove timezone if present
                if date.tzinfo is not None:
                    date = date.replace(tzinfo=None)
                
                amount = float(t.get('Amount', 0))
                if amount <= 0:
                    continue
                
                key = (date.year, date.month)
                monthly[key] += amount
                
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Error processing transaction for monthly grouping: {e}, transaction: {t.get('Id', 'unknown')}")
                skipped_count += 1
                continue
        
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} transactions during monthly grouping due to date issues")
        
        logger.info(f"Grouped transactions into {len(monthly)} unique months")
        return dict(monthly)
    
    def _identify_high_spending_months(
        self,
        monthly_spending: Dict[Tuple[int, int], float],
        avg_monthly: float,
        std_monthly: float
    ) -> List[Dict[str, Any]]:
        """Identify months with significantly higher spending (historical only, for pattern detection)."""
        high_months = []
        threshold = avg_monthly + (std_monthly * 0.5)  # Above average + 0.5 std dev
        now = datetime.utcnow()
        current_year = now.year
        current_month = now.month
        
        for (year, month), amount in monthly_spending.items():
            # Only include historical months (past months)
            if year < current_year or (year == current_year and month < current_month):
                if amount >= threshold:
                    percentage_above_avg = ((amount - avg_monthly) / avg_monthly * 100) if avg_monthly > 0 else 0
                    
                    high_months.append({
                        'year': year,
                        'month': month,
                        'month_name': month_name[month],
                        'amount': float(amount),
                        'percentage_above_average': round(percentage_above_avg, 1),
                        'is_seasonal': self._is_seasonal_month(month, monthly_spending)
                    })
        
        # Sort by amount descending
        high_months.sort(key=lambda x: x['amount'], reverse=True)
        return high_months
    
    def _is_seasonal_month(
        self,
        month: int,
        monthly_spending: Dict[Tuple[int, int], float]
    ) -> bool:
        """Check if a month consistently shows high spending across years."""
        month_amounts = [amount for (y, m), amount in monthly_spending.items() if m == month]
        
        if len(month_amounts) < 2:
            return False
        
        avg_for_month = np.mean(month_amounts)
        overall_avg = np.mean(list(monthly_spending.values()))
        
        return avg_for_month >= overall_avg * (1 + self.seasonal_threshold)
    
    def _detect_seasonal_patterns(
        self,
        monthly_spending: Dict[Tuple[int, int], float]
    ) -> List[Dict[str, Any]]:
        """Detect seasonal spending patterns."""
        patterns = []
        
        # Group by month across all years
        by_month = defaultdict(list)
        for (year, month), amount in monthly_spending.items():
            by_month[month].append(amount)
        
        overall_avg = np.mean(list(monthly_spending.values()))
        
        for month in range(1, 13):
            if month not in by_month:
                continue
            
            month_amounts = by_month[month]
            if len(month_amounts) < 2:
                continue
            
            avg_for_month = np.mean(month_amounts)
            std_for_month = np.std(month_amounts) if len(month_amounts) > 1 else 0
            
            # Check if consistently high or low
            if avg_for_month >= overall_avg * (1 + self.seasonal_threshold):
                patterns.append({
                    'month': month,
                    'month_name': month_name[month],
                    'average_spending': float(avg_for_month),
                    'pattern_type': 'high',
                    'consistency': 'high' if std_for_month < avg_for_month * 0.2 else 'moderate',
                    'percentage_above_average': round(((avg_for_month - overall_avg) / overall_avg * 100), 1)
                })
            elif avg_for_month <= overall_avg * (1 - self.seasonal_threshold):
                patterns.append({
                    'month': month,
                    'month_name': month_name[month],
                    'average_spending': float(avg_for_month),
                    'pattern_type': 'low',
                    'consistency': 'high' if std_for_month < avg_for_month * 0.2 else 'moderate',
                    'percentage_below_average': round(((overall_avg - avg_for_month) / overall_avg * 100), 1)
                })
        
        return sorted(patterns, key=lambda x: x.get('percentage_above_average', 0) or -x.get('percentage_below_average', 0), reverse=True)
    
    def _identify_recurring_events(
        self,
        monthly_spending: Dict[Tuple[int, int], float],
        transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify recurring events that cause high spending."""
        events = []
        now = datetime.utcnow()
        current_year = now.year
        current_month = now.month
        
        # Check for known events (all now use 4-value format)
        for event_name, event_data in self.HIGH_SPENDING_EVENTS.items():
            if event_data is None:
                continue
            
            if len(event_data) == 4:
                start_month, start_day, end_month, end_day = event_data
                event_months = self._check_multi_month_event(
                    monthly_spending, transactions, start_month, start_day,
                    end_month, end_day, event_name
                )
                # Filter to only include future occurrences
                future_events = []
                for event in event_months:
                    event_year = event.get('year', current_year)
                    event_month = event.get('month') or event.get('start_month')
                    if event_year > current_year or (event_year == current_year and event_month >= current_month):
                        future_events.append(event)
                if future_events:
                    events.extend(future_events)
        
        # Check for user-specific recurring patterns (only future)
        user_patterns = self._detect_user_specific_patterns(monthly_spending)
        future_patterns = []
        for pattern in user_patterns:
            month = pattern.get('month')
            if month:
                # Find next occurrence
                next_year = current_year
                if month < current_month:
                    next_year = current_year + 1
                pattern['year'] = next_year
                pattern['month'] = month
                # Only include if there's a meaningful increase (at least 10%)
                if pattern.get('average_increase', 0) >= 10.0:
                    future_patterns.append(pattern)
        events.extend(future_patterns)
        
        # Return top 3 recurring events (more concise)
        return sorted(events, key=lambda x: x.get('average_increase', 0), reverse=True)[:3]
    
    def _check_event_in_months(
        self,
        monthly_spending: Dict[Tuple[int, int], float],
        transactions: List[Dict[str, Any]],
        month: int,
        start_day: int,
        end_day: int,
        event_name: str
    ) -> List[Dict[str, Any]]:
        """Check if an event period shows high spending."""
        event_data = []
        
        for (year, m), amount in monthly_spending.items():
            if m == month:
                # Get average for this month across all years
                month_amounts = [amt for (y, mon), amt in monthly_spending.items() if mon == month]
                avg_month = np.mean(month_amounts)
                overall_avg = np.mean(list(monthly_spending.values()))
                
                if amount >= overall_avg * self.high_spending_threshold:
                    event_data.append({
                        'event_name': event_name,
                        'year': year,
                        'month': month,
                        'month_name': month_name[month],
                        'amount': float(amount),
                        'average_increase': round(((amount - overall_avg) / overall_avg * 100), 1),
                        'occurrences': len([(y, m) for (y, m) in monthly_spending.keys() if m == month])
                    })
        
        return event_data[:3]  # Return top 3 occurrences
    
    def _check_multi_month_event(
        self,
        monthly_spending: Dict[Tuple[int, int], float],
        transactions: List[Dict[str, Any]],
        start_month: int,
        start_day: int,
        end_month: int,
        end_day: int,
        event_name: str
    ) -> List[Dict[str, Any]]:
        """Check multi-month events."""
        event_data = []
        overall_avg = np.mean(list(monthly_spending.values()))
        
        # Group by year for multi-month periods
        years = set(y for (y, m) in monthly_spending.keys())
        
        for year in years:
            period_total = 0
            months_in_period = []
            
            # Calculate spending for the period
            for month in range(start_month, end_month + 1):
                if (year, month) in monthly_spending:
                    period_total += monthly_spending[(year, month)]
                    months_in_period.append(month)
            
            if period_total >= overall_avg * len(months_in_period) * self.high_spending_threshold:
                avg_monthly_for_period = period_total / len(months_in_period) if months_in_period else 0
                event_data.append({
                    'event_name': event_name,
                    'year': year,
                    'start_month': start_month,
                    'end_month': end_month,
                    'period_total': float(period_total),
                    'average_monthly': float(avg_monthly_for_period),
                    'average_increase': round(((avg_monthly_for_period - overall_avg) / overall_avg * 100), 1),
                    'occurrences': len([y for y in years if any((y, m) in monthly_spending for m in range(start_month, end_month + 1))])
                })
        
        return event_data[:2]  # Return top 2 occurrences
    
    def _detect_user_specific_patterns(
        self,
        monthly_spending: Dict[Tuple[int, int], float]
    ) -> List[Dict[str, Any]]:
        """Detect user-specific recurring patterns."""
        patterns = []
        overall_avg = np.mean(list(monthly_spending.values()))
        
        # Look for months that consistently show high spending
        by_month = defaultdict(list)
        for (year, month), amount in monthly_spending.items():
            by_month[month].append((year, amount))
        
        for month, occurrences in by_month.items():
            if len(occurrences) >= 2:  # Need at least 2 occurrences
                amounts = [amt for _, amt in occurrences]
                avg_for_month = np.mean(amounts)
                
                if avg_for_month >= overall_avg * self.high_spending_threshold:
                    patterns.append({
                        'event_name': f'Recurring High Spending in {month_name[month]}',
                        'month': month,
                        'month_name': month_name[month],
                        'average_spending': float(avg_for_month),
                        'average_increase': round(((avg_for_month - overall_avg) / overall_avg * 100), 1),
                        'occurrences': len(occurrences),
                        'consistency': 'high' if np.std(amounts) < avg_for_month * 0.25 else 'moderate'
                    })
        
        return patterns
    
    def _predict_future_periods(
        self,
        monthly_spending: Dict[Tuple[int, int], float],
        high_spending_months: List[Dict[str, Any]],
        seasonal_patterns: List[Dict[str, Any]],
        forecast_months: int
    ) -> List[Dict[str, Any]]:
        """Predict future high-spending periods."""
        predictions = []
        now = datetime.utcnow()
        current_year = now.year
        current_month = now.month
        
        overall_avg = np.mean(list(monthly_spending.values()))
        
        # Create a map of month to historical high spending
        month_high_spending = {}
        for hm in high_spending_months:
            month = hm['month']
            if month not in month_high_spending:
                month_high_spending[month] = []
            month_high_spending[month].append(hm['amount'])
        
        # Predict for next N months
        for i in range(1, forecast_months + 1):
            target_month = current_month + i
            target_year = current_year
            
            # Handle year rollover
            while target_month > 12:
                target_month -= 12
                target_year += 1
            
            # Check if this month has historical high spending
            is_predicted_high = False
            confidence = 'low'
            predicted_amount = overall_avg
            reasoning = []
            
            # Check seasonal patterns
            seasonal = next((p for p in seasonal_patterns if p['month'] == target_month and p['pattern_type'] == 'high'), None)
            if seasonal:
                is_predicted_high = True
                confidence = seasonal.get('consistency', 'moderate')
                predicted_amount = seasonal['average_spending']
                reasoning.append(f"Historical seasonal pattern: {seasonal['percentage_above_average']}% above average")
            
            # Check month-specific high spending history
            if target_month in month_high_spending:
                historical_amounts = month_high_spending[target_month]
                if len(historical_amounts) >= 2:
                    is_predicted_high = True
                    if confidence == 'low':
                        confidence = 'moderate'
                    predicted_historical = np.mean(historical_amounts)
                    if predicted_historical > predicted_amount:
                        predicted_amount = predicted_historical
                    reasoning.append(f"Recurring high spending in {month_name[target_month]}")
            
            # Check for known events (including multi-month events like New Year)
            for event_name, event_data in self.HIGH_SPENDING_EVENTS.items():
                if event_data is None:
                    continue
                
                # All events now use 4-value format: (start_month, start_day, end_month, end_day)
                if len(event_data) == 4:
                    start_month, start_day, end_month, end_day = event_data
                    
                    # Check if target month is in the range
                    if start_month <= end_month:
                        # Same year range (e.g., Summer Vacation: June-August, Christmas: Dec 20-31)
                        if start_month <= target_month <= end_month:
                            is_predicted_high = True
                            if confidence == 'low':
                                confidence = 'moderate'
                            reasoning.append(f"Known high-spending period: {event_name}")
                    else:
                        # Cross-year range (e.g., New Year: Dec 28 - Jan 1)
                        # Check if we're in December of current year or January of next year
                        if target_month == start_month:  # December
                            is_predicted_high = True
                            if confidence == 'low':
                                confidence = 'moderate'
                            reasoning.append(f"Known high-spending event: {event_name}")
                        elif target_month == end_month and target_year > current_year:  # January of next year
                            is_predicted_high = True
                            if confidence == 'low':
                                confidence = 'moderate'
                            reasoning.append(f"Known high-spending event: {event_name}")
            
            if is_predicted_high:
                expected_increase = ((predicted_amount - overall_avg) / overall_avg * 100) if overall_avg > 0 else 0
                # Include if there's any increase OR if there's a known event (even if 0% increase)
                # This ensures events like Valentine's Day are shown even if historical data shows no increase
                if expected_increase > 0 or reasoning:  # Show if increase OR has reasoning (known event)
                    predictions.append({
                        'year': target_year,
                        'month': target_month,
                        'month_name': month_name[target_month],
                        'predicted_amount': round(float(predicted_amount), 2),
                        'average_amount': round(float(overall_avg), 2),
                        'expected_increase': round(expected_increase, 1),
                        'confidence': confidence,
                        'reasoning': reasoning
                    })
        
        # Filter to only include future months (at least 1 month ahead)
        future_predictions = [
            p for p in predictions 
            if p['year'] > current_year or (p['year'] == current_year and p['month'] > current_month)
        ]
        
        logger.info(f"Generated {len(predictions)} predictions, {len(future_predictions)} are in the future")
        
        # Return top 5 predictions sorted by expected increase
        return sorted(future_predictions, key=lambda x: x['expected_increase'], reverse=True)[:5]
    
    def _identify_low_spending_periods(
        self,
        monthly_spending: Dict[Tuple[int, int], float],
        avg_monthly: float,
        std_monthly: float
    ) -> List[Dict[str, Any]]:
        """Identify months with significantly lower spending (saving opportunities)."""
        low_months = []
        threshold = avg_monthly - (std_monthly * 0.5)  # Below average - 0.5 std dev
        
        for (year, month), amount in monthly_spending.items():
            if amount <= threshold and amount > 0:
                percentage_below_avg = ((avg_monthly - amount) / avg_monthly * 100) if avg_monthly > 0 else 0
                potential_savings = avg_monthly - amount
                
                low_months.append({
                    'year': year,
                    'month': month,
                    'month_name': month_name[month],
                    'amount': float(amount),
                    'percentage_below_average': round(percentage_below_avg, 1),
                    'potential_savings': round(float(potential_savings), 2),
                    'is_seasonal': self._is_seasonal_low_month(month, monthly_spending)
                })
        
        # Sort by potential savings descending
        low_months.sort(key=lambda x: x['potential_savings'], reverse=True)
        return low_months
    
    def _is_seasonal_low_month(
        self,
        month: int,
        monthly_spending: Dict[Tuple[int, int], float]
    ) -> bool:
        """Check if a month consistently shows low spending across years."""
        month_amounts = [amount for (y, m), amount in monthly_spending.items() if m == month]
        
        if len(month_amounts) < 2:
            return False
        
        avg_for_month = np.mean(month_amounts)
        overall_avg = np.mean(list(monthly_spending.values()))
        
        return avg_for_month <= overall_avg * (1 - self.seasonal_threshold)
    
    def _analyze_income_patterns(
        self,
        income: List[Dict[str, Any]],
        monthly_spending: Dict[Tuple[int, int], float],
        forecast_months: int = 6
    ) -> Dict[str, Any]:
        """Analyze income patterns to identify high-income periods within forecast period."""
        monthly_income = self._group_by_month(income)
        
        if not monthly_income:
            return None
        
        income_values = list(monthly_income.values())
        avg_income = np.mean(income_values)
        avg_spending = np.mean(list(monthly_spending.values())) if monthly_spending else 0
        
        # Calculate forecast period
        now = datetime.utcnow()
        current_year = now.year
        current_month = now.month
        forecast_end_month = current_month + forecast_months
        forecast_end_year = current_year
        while forecast_end_month > 12:
            forecast_end_month -= 12
            forecast_end_year += 1
        
        # Find months with above-average income (only within forecast period)
        high_income_months = []
        for (year, month), amount in monthly_income.items():
            # Check if this month is in the forecast period
            if (year > current_year or (year == current_year and month >= current_month)) and \
               (year < forecast_end_year or (year == forecast_end_year and month <= forecast_end_month)):
                if amount >= avg_income * 1.1:  # 10% above average
                    excess_income = amount - avg_spending
                    percentage_above_avg = ((amount - avg_income) / avg_income * 100) if avg_income > 0 else 0
                    
                    high_income_months.append({
                        'year': year,
                        'month': month,
                        'month_name': month_name[month],
                        'income': float(amount),
                        'spending': float(monthly_spending.get((year, month), 0)),
                        'excess_income': round(float(excess_income), 2),
                        'percentage_above_average': round(percentage_above_avg, 1)
                    })
        
        high_income_months.sort(key=lambda x: x['excess_income'], reverse=True)
        
        return {
            'average_monthly_income': float(avg_income),
            'average_monthly_spending': float(avg_spending),
            'average_surplus': float(avg_income - avg_spending) if avg_income > avg_spending else 0,
            'high_income_months': high_income_months[:6]  # Top 6 within forecast period
        }
    
    def _generate_saving_suggestions(
        self,
        low_spending_periods: List[Dict[str, Any]],
        seasonal_patterns: List[Dict[str, Any]],
        income_patterns: Optional[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        monthly_spending: Dict[Tuple[int, int], float],
        forecast_months: int = 6
    ) -> List[Dict[str, Any]]:
        """Generate saving suggestions based on low-spending periods and income (filtered to forecast period)."""
        suggestions = []
        now = datetime.utcnow()
        current_year = now.year
        current_month = now.month
        
        # Calculate forecast end
        forecast_end_month = current_month + forecast_months
        forecast_end_year = current_year
        while forecast_end_month > 12:
            forecast_end_month -= 12
            forecast_end_year += 1
        
        # Suggest saving during low-spending periods (only within forecast period)
        low_seasonal = [p for p in seasonal_patterns if p['pattern_type'] == 'low']
        for pattern in low_seasonal[:3]:  # Top 3 low-spending months
            month = pattern['month']
            target_month = month
            target_year = current_year
            
            # Adjust year if month has passed
            if month < current_month:
                target_year = current_year + 1
            
            # Check if this is within forecast period
            if (target_year > current_year or (target_year == current_year and target_month >= current_month)) and \
               (target_year < forecast_end_year or (target_year == forecast_end_year and target_month <= forecast_end_month)):
                avg_savings = pattern.get('percentage_below_average', 0)
                if avg_savings:
                    suggestions.append({
                        'type': 'low_spending_period',
                        'month': target_month,
                        'month_name': pattern['month_name'],
                        'year': target_year,
                        'suggestion': f"Consider saving extra in {pattern['month_name']} {target_year}. "
                                    f"You typically spend {abs(avg_savings):.0f}% less during this month, "
                                    f"making it an ideal time to build your savings.",
                        'potential_savings': pattern.get('average_spending', 0),
                        'priority': 'medium'
                    })
        
        # Suggest saving during high-income periods (only within forecast period)
        if income_patterns and income_patterns.get('high_income_months'):
            for high_income in income_patterns['high_income_months'][:3]:  # Top 3
                month = high_income['month']
                year = high_income['year']
                target_month = month
                target_year = year
                
                # Check if within forecast period (already filtered in _analyze_income_patterns, but double-check)
                if (target_year > current_year or (target_year == current_year and target_month >= current_month)) and \
                   (target_year < forecast_end_year or (target_year == forecast_end_year and target_month <= forecast_end_month)):
                    excess = high_income.get('excess_income', 0)
                    if excess > 0:
                        suggestions.append({
                            'type': 'high_income_period',
                            'month': target_month,
                            'month_name': high_income['month_name'],
                            'year': target_year,
                            'suggestion': f"Save extra in {high_income['month_name']} {target_year}. "
                                        f"Your income is typically {high_income['percentage_above_average']:.0f}% above average "
                                        f"this month, with potential savings of ${excess:.2f}.",
                            'potential_savings': excess,
                            'priority': 'high'
                        })
        
        # Suggest saving before predicted high-spending periods (only within forecast period)
        for prediction in predictions[:2]:  # Top 2 predictions
            pred_month = prediction['month']
            pred_year = prediction['year']
            
            # Suggest saving in the month before
            save_month = pred_month - 1
            save_year = pred_year
            if save_month < 1:
                save_month = 12
                save_year = pred_year - 1
            
            # Only if within forecast period
            if (save_year > current_year or (save_year == current_year and save_month >= current_month)) and \
               (save_year < forecast_end_year or (save_year == forecast_end_year and save_month <= forecast_end_month)):
                expected_increase = prediction.get('expected_increase', 0)
                if expected_increase > 20:  # Only for significant increases
                    suggestions.append({
                        'type': 'prepare_for_high_spending',
                        'month': save_month,
                        'month_name': month_name[save_month],
                        'year': save_year,
                        'suggestion': f"Start saving in {month_name[save_month]} {save_year} to prepare for "
                                    f"high spending in {prediction['month_name']} {pred_year}. "
                                    f"Expected spending increase: {expected_increase:.0f}%.",
                        'potential_savings': prediction.get('predicted_amount', 0) * 0.2,  # Suggest saving 20% of predicted
                        'priority': 'high',
                        'related_period': f"{prediction['month_name']} {pred_year}"
                    })
        
        # Sort by priority and potential savings
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        suggestions.sort(key=lambda x: (priority_order.get(x.get('priority', 'low'), 1), x.get('potential_savings', 0)), reverse=True)
        
        return suggestions[:3]  # Return top 3 suggestions (more concise)
    
    def _generate_insights(
        self,
        monthly_spending: Dict[Tuple[int, int], float],
        seasonal_patterns: List[Dict[str, Any]],
        recurring_events: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        low_spending_periods: List[Dict[str, Any]] = None,
        income_patterns: Optional[Dict[str, Any]] = None,
        saving_suggestions: List[Dict[str, Any]] = None,
        forecast_months: int = 6
    ) -> List[str]:
        """Generate human-readable insights filtered to the forecast period."""
        insights = []
        now = datetime.utcnow()
        current_year = now.year
        current_month = now.month
        
        # Calculate forecast end date
        forecast_end_month = current_month + forecast_months
        forecast_end_year = current_year
        while forecast_end_month > 12:
            forecast_end_month -= 12
            forecast_end_year += 1
        
        if not monthly_spending:
            return ["Insufficient data to generate insights"]
        
        overall_avg = np.mean(list(monthly_spending.values()))
        
        # Filter seasonal patterns to forecast period
        forecast_seasonal = []
        for pattern in seasonal_patterns:
            month = pattern['month']
            # Check if this month appears in the forecast period
            for i in range(forecast_months):
                check_month = current_month + i + 1
                check_year = current_year
                while check_month > 12:
                    check_month -= 12
                    check_year += 1
                if check_month == month:
                    forecast_seasonal.append(pattern)
                    break
        
        # Top seasonal pattern (prefer in forecast period, but show general pattern if none)
        high_seasonal = [p for p in forecast_seasonal if p['pattern_type'] == 'high']
        if high_seasonal:
            top = high_seasonal[0]
            # Find next occurrence in forecast period
            next_occurrence_year = current_year
            if top['month'] < current_month:
                next_occurrence_year = current_year + 1
            insights.append(
                f"You typically spend {top['percentage_above_average']}% more in {top['month_name']} "
                f"compared to your average monthly spending."
            )
        elif seasonal_patterns:
            # Show general seasonal pattern even if not in exact forecast period
            high_seasonal_general = [p for p in seasonal_patterns if p['pattern_type'] == 'high']
            if high_seasonal_general:
                top = high_seasonal_general[0]
                insights.append(
                    f"You typically spend {top['percentage_above_average']}% more in {top['month_name']} "
                    f"compared to your average monthly spending."
                )
        
        # New Year detection (only if in forecast period)
        # Check if December or January is in the forecast period
        december_in_forecast = any(
            (current_month + i) % 12 == 11 or (current_month + i) % 12 == 0
            for i in range(1, forecast_months + 1)
        )
        if december_in_forecast:
            new_year_events = [e for e in recurring_events if 'New Year' in e.get('event_name', '')]
            if new_year_events:
                insights.append(
                    f"New Year period shows increased spending. Consider budgeting extra for December-January "
                    f"to cover holiday expenses."
                )
        
        # Top recurring event (prefer in forecast period, but show general if none)
        if recurring_events:
            event_shown = False
            # Check if any recurring event month is in forecast period
            for event in recurring_events:
                event_month = event.get('month')
                if event_month:
                    # Check if this month appears in forecast
                    for i in range(forecast_months):
                        check_month = current_month + i + 1
                        check_year = current_year
                        while check_month > 12:
                            check_month -= 12
                            check_year += 1
                        if check_month == event_month:
                            if 'New Year' not in event.get('event_name', ''):
                                insights.append(
                                    f"Your spending increases by an average of {event.get('average_increase', 0)}% "
                                    f"during {event.get('event_name', 'certain periods')}."
                                )
                                event_shown = True
                            break
                    if event_shown:
                        break
            
            # If no event in forecast period, show top recurring event anyway
            if not event_shown and recurring_events:
                top_event = recurring_events[0]
                if 'New Year' not in top_event.get('event_name', ''):
                    insights.append(
                        f"Your spending increases by an average of {top_event.get('average_increase', 0)}% "
                        f"during {top_event.get('event_name', 'certain periods')}."
                    )
        
        # Upcoming predictions (already filtered to forecast period)
        if predictions:
            for pred in predictions[:2]:  # Show top 2 predictions
                expected_increase = pred.get('expected_increase', 0)
                reasoning_text = ""
                event_name = None
                
                if pred.get('reasoning'):
                    # Extract event names from reasoning
                    event_reasons = [r for r in pred['reasoning'] if 'Known high-spending' in r or 'event' in r.lower() or 'period' in r.lower()]
                    if event_reasons:
                        event_name = event_reasons[0].replace('Known high-spending period: ', '').replace('Known high-spending event: ', '')
                        reasoning_text = f" due to {event_name}"
                
                # Format the insight based on whether there's an event or just a pattern
                if event_name:
                    # If there's a known event, always mention it
                    if expected_increase >= 5.0:
                        insights.append(
                            f"{pred['month_name']} {pred['year']} is predicted to be a high-spending period "
                            f"({expected_increase}% above average) due to {event_name}."
                        )
                    elif expected_increase > 0:
                        insights.append(
                            f"{pred['month_name']} {pred['year']} may see higher spending "
                            f"({expected_increase}% above average) due to {event_name}."
                        )
                    else:
                        # Even if no increase, mention the event
                        insights.append(
                            f"{pred['month_name']} {pred['year']} typically sees increased spending due to {event_name}."
                        )
                elif expected_increase >= 5.0:
                    insights.append(
                        f"{pred['month_name']} {pred['year']} is predicted to be a high-spending period "
                        f"({expected_increase}% above average)."
                    )
                elif expected_increase > 0:
                    insights.append(
                        f"{pred['month_name']} {pred['year']} may see higher spending "
                        f"({expected_increase}% above average)."
                    )
        
        # Show recurring events in forecast period
        if recurring_events:
            for event in recurring_events[:2]:  # Show top 2 events
                event_month = event.get('month')
                event_year = event.get('year', current_year)
                if event_month:
                    # Check if in forecast period
                    if (event_year > current_year or (event_year == current_year and event_month >= current_month)) and \
                       (event_year < forecast_end_year or (event_year == forecast_end_year and event_month <= forecast_end_month)):
                        event_name = event.get('event_name', 'certain periods')
                        increase = event.get('average_increase', 0)
                        if increase > 0:
                            insights.append(
                                f"{event_name} typically increases your spending by {increase}% "
                                f"(occurs in {month_name[event_month]})."
                            )
        
        # Show saving suggestions (prioritize high priority)
        if saving_suggestions:
            # Show high priority first
            high_priority = [s for s in saving_suggestions if s.get('priority') == 'high']
            if high_priority:
                insights.append(high_priority[0]['suggestion'])
            else:
                # Show top suggestion if no high priority
                insights.append(saving_suggestions[0]['suggestion'])
        
        # Show low spending periods as saving opportunities
        if low_spending_periods:
            top_low = low_spending_periods[0]
            insights.append(
                f"Consider saving extra in {top_low['month_name']} {top_low['year']}. "
                f"You typically spend {top_low['percentage_below_average']:.0f}% less during this month, "
                f"potentially saving ${top_low['potential_savings']:.2f}."
            )
        
        # Fallback: Show general information only if no specific insights
        if not insights:
            # Show average spending info
            insights.append(
                f"Your average monthly spending is ${overall_avg:.2f}. "
                f"Monitor your spending over the next {forecast_months} month{'s' if forecast_months > 1 else ''} to maintain this average."
            )
            
            # Show if there are any seasonal patterns (even if not in forecast)
            if seasonal_patterns:
                top_seasonal = seasonal_patterns[0]
                insights.append(
                    f"You typically spend {abs(top_seasonal.get('percentage_above_average', 0) or top_seasonal.get('percentage_below_average', 0)):.0f}% "
                    f"{'more' if top_seasonal.get('pattern_type') == 'high' else 'less'} in {top_seasonal['month_name']}."
                )
        
        return insights[:4]  # Return top 4 insights (more informative)
    
    def _empty_response(self, status: str) -> Dict[str, Any]:
        """Return empty response structure."""
        return {
            'predictions': [],
            'seasonal_patterns': [],
            'recurring_events': [],
            'historical_high_months': [],
            'low_spending_periods': [],
            'average_monthly_spending': 0.0,
            'insights': [],
            'status': status
        }

