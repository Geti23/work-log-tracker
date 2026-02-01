"""
Metrics and observability for anomaly detection.
"""

import time
from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AnomalyDetectionMetrics:
    """Track metrics for anomaly detection operations."""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_detections = 0
        self.failed_detections = 0
        self.anomalies_detected = 0
        self.total_processing_time = 0.0
        self.users_analyzed = set()
    
    def record_detection(
        self,
        user_id: str,
        anomalies_count: int,
        processing_time: float,
        status: str,
        success: bool = True
    ):
        """Record a detection operation."""
        self.total_requests += 1
        self.users_analyzed.add(user_id)
        self.total_processing_time += processing_time
        
        if success:
            self.successful_detections += 1
            self.anomalies_detected += anomalies_count
        else:
            self.failed_detections += 1
        
        logger.info(
            f"Metrics: requests={self.total_requests}, "
            f"success={self.successful_detections}, "
            f"failures={self.failed_detections}, "
            f"anomalies={self.anomalies_detected}, "
            f"avg_time={self.get_average_processing_time():.2f}s"
        )
    
    def get_average_processing_time(self) -> float:
        """Get average processing time in seconds."""
        if self.total_requests == 0:
            return 0.0
        return self.total_processing_time / self.total_requests
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            'total_requests': self.total_requests,
            'successful_detections': self.successful_detections,
            'failed_detections': self.failed_detections,
            'anomalies_detected': self.anomalies_detected,
            'unique_users': len(self.users_analyzed),
            'average_processing_time_seconds': self.get_average_processing_time(),
            'success_rate': (
                self.successful_detections / self.total_requests 
                if self.total_requests > 0 else 0.0
            )
        }


# Global metrics instance
metrics = AnomalyDetectionMetrics()

