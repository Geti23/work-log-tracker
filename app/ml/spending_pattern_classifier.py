"""
Spending Pattern Classifier for ExpenseAlly.

Identifies a user's spending personality based on their
transaction behavior using a lightweight statistical approach.

The goal is not prediction, but self-awareness and financial insight.
"""

from typing import List, Dict, Any
import numpy as np


class SpendingPatternClassifier:
    """
    Classifies spending behavior into friendly, human-readable personalities.
    """

    def classify(
        self,
        transactions: List[Dict[str, Any]],
        period_days: int = 7
    ) -> Dict[str, Any]:
        """
        Classify spending pattern for a given period.

        Args:
            transactions: List of expense transactions
            period_days: Time window for analysis

        Returns:
            Classification result with personality, confidence, and explanation
        """
        if not transactions:
            return self._result(
                "No Activity Yet 💤",
                0.0,
                "We don’t see any spending activity yet. Add a few transactions to unlock your spending personality."
            )

        amounts = np.array([
            float(t["Amount"])
            for t in transactions
            if t.get("Amount") and float(t["Amount"]) > 0
        ])

        if len(amounts) < 3:
            return self._result(
                "Just Getting Started 🌱",
                0.3,
                "You’ve started tracking expenses, but we need a bit more data to understand your habits."
            )

        total_spent = amounts.sum()
        avg_spend = amounts.mean()
        std_spend = amounts.std()
        coefficient_of_variation = std_spend / avg_spend if avg_spend > 0 else 0

        # --- Classification rules ---
        if coefficient_of_variation > 1.2 and total_spent > avg_spend * len(amounts):
            return self._result(
                "Impulse Buyer 🚀",
                0.85,
                "Your spending shows sudden spikes and big purchases. You tend to buy on impulse rather than planning ahead."
            )

        if coefficient_of_variation < 0.4:
            return self._result(
                "Balanced Planner ⚖️",
                0.9,
                "Your spending is steady and well-distributed. You manage expenses consistently and avoid extremes."
            )

        if total_spent < avg_spend * len(amounts) * 0.7:
            return self._result(
                "Smart Saver 🐢",
                0.8,
                "You keep expenses low and under control. Careful decisions and long-term thinking define your style."
            )

        if 0.4 <= coefficient_of_variation <= 1.2:
            return self._result(
                "Unpredictable Spender 🎢",
                0.65,
                "Your spending varies from time to time. Some weeks are calm, others more spontaneous."
            )

        return self._result(
            "Still Exploring 🤔",
            0.5,
            "Your spending doesn’t clearly match a specific pattern yet. Keep tracking to reveal your personality."
        )

    @staticmethod
    def _result(pattern: str, confidence: float, explanation: str) -> Dict[str, Any]:
        """
        Format classification result.
        """
        return {
            "pattern": pattern,
            "confidence": round(confidence, 2),
            "explanation": explanation
        }
