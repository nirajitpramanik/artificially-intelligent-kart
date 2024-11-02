import datetime
from .models import Order, Product
from django.db.models import Count, Avg
from django.utils import timezone
from datetime import timedelta
from django.contrib.auth.models import User
from typing import Tuple, List, Optional
import math
from statistics import stdev, mean, median
from collections import defaultdict

class OrderPredictor:
    def __init__(self, max_depth: int = 3, evaluation_window: int = 90):
        self.max_depth = max_depth
        self.evaluation_window = evaluation_window
        self.min_pattern_confidence = 0.3
        self.min_orders_for_pattern = 2  # Require at least 2 orders to establish a pattern

    def analyze_quantity_patterns(self, orders: List[Order]) -> Tuple[float, float]:
        """Analyze quantity patterns and trends."""
        if not orders or len(orders) < self.min_orders_for_pattern:
            return 0.0, 0.0
        
        quantities = [order.quantity for order in orders]
        
        # Calculate quantity trend
        quantity_trend = (quantities[-1] - quantities[0]) / len(quantities)
        
        # Calculate quantity consistency
        try:
            quantity_std = stdev(quantities)
            quantity_mean = mean(quantities)
            quantity_consistency = 1.0 / (1.0 + (quantity_std / quantity_mean))
        except:
            quantity_consistency = 0.0
            
        # Boost score for high quantities
        quantity_boost = math.log10(max(quantities) + 1) / 2
        
        return quantity_trend * (1 + quantity_boost), quantity_consistency

    def analyze_interval_patterns(self, orders: List[Order]) -> Tuple[float, timedelta, float]:
        """Analyze time intervals between orders, handling cases with a single interval."""
        if not orders or len(orders) < self.min_orders_for_pattern:
            return 0.0, timedelta(days=30), 0.0  # Default to 30 days if not enough data

        intervals = [
            (orders[i + 1].order_date - orders[i].order_date).days
            for i in range(len(orders) - 1)
        ]

        # Initialize defaults for consistency
        interval_mean = mean(intervals) if intervals else 0.0
        predicted_interval = timedelta(days=30)  # Default to 30 days
        interval_consistency = 1.0 if len(intervals) == 1 else 0.0  # Default consistency

        # Calculate interval consistency and predicted interval
        try:
            if len(intervals) > 1:
                interval_std = stdev(intervals)
                interval_consistency = 1.0 / (1.0 + (interval_std / interval_mean))

                # Use average interval as predicted interval if consistency is high enough
                if interval_consistency > 0.5:  # Adjust threshold if needed
                    predicted_interval = timedelta(days=interval_mean)
                else:
                    predicted_interval = timedelta(days=median(intervals))
            else:
                # Only one interval, assume consistent interval with the single value
                predicted_interval = timedelta(days=interval_mean)

        except:
            # Fall back to 30-day interval if calculations fail
            interval_consistency = 0.0

        return interval_consistency, predicted_interval, interval_mean


    def calculate_volume_score(self, quantities: List[int]) -> float:
        """Calculate score based on order volumes."""
        if not quantities or len(quantities) < self.min_orders_for_pattern:
            return 0.0
            
        total_volume = sum(quantities)
        avg_volume = total_volume / len(quantities)
        recent_volume = quantities[-1]
        
        # Enhanced volume scoring
        volume_trend = recent_volume / avg_volume if avg_volume > 0 else 1.0
        volume_magnitude = math.log10(total_volume + 1) / math.log10(100)  # Normalized log scale
        
        return min(100.0, (volume_trend * 50.0 + volume_magnitude * 50.0))

    def calculate_pattern_strength(self, orders: List[Order]) -> float:
        """Calculate the strength of the ordering pattern."""
        if len(orders) < self.min_orders_for_pattern:
            return 0.0
            
        # Analyze consecutive order consistency
        intervals = []
        quantities = []
        for i in range(len(orders) - 1):
            interval = (orders[i + 1].order_date - orders[i].order_date).days
            intervals.append(interval)
            quantities.append(orders[i].quantity)
        quantities.append(orders[-1].quantity)
        
        try:
            interval_consistency = 1.0 / (1.0 + stdev(intervals) / mean(intervals))
            quantity_consistency = 1.0 / (1.0 + stdev(quantities) / mean(quantities))
            return (interval_consistency + quantity_consistency) / 2
        except:
            return 0.0

    def calculate_heuristic(self, product: Product, user: User, reference_date: datetime) -> Tuple[float, Optional[datetime.datetime]]:
        """
        Enhanced heuristic calculation with stronger emphasis on established patterns
        and higher volumes
        """
        recent_period = timezone.now() - timedelta(days=self.evaluation_window)
        orders = list(Order.objects.filter(
            user=user,
            product=product,
            order_date__gte=recent_period
        ).order_by('order_date'))
        
        if len(orders) < self.min_orders_for_pattern:
            return 0.0, None
            
        # Analyze patterns
        quantity_trend, quantity_consistency = self.analyze_quantity_patterns(orders)
        quantities = [order.quantity for order in orders]
        volume_score = self.calculate_volume_score(quantities)
        pattern_strength = self.calculate_pattern_strength(orders)
        
        # Analyze intervals
        interval_consistency, predicted_interval, avg_interval = self.analyze_interval_patterns(orders)
        
        # Calculate predicted date
        last_order_date = orders[-1].order_date
        predicted_date = last_order_date + predicted_interval
        
        # Combine scores with adjusted weights
        weights = {
            'volume': 0.35,      
            'pattern': 0.30,     
            'interval': 0.20,    
            'quantity': 0.15     
        }
        
        final_score = (
            weights['volume'] * volume_score +
            weights['pattern'] * (pattern_strength * 100) +
            weights['interval'] * (interval_consistency * 100) +
            weights['quantity'] * (quantity_consistency * 100)
        )
        
        # Penalty for single orders
        if len(orders) < self.min_orders_for_pattern:
            final_score *= 0.1
            
        return final_score, predicted_date

    def predict_next_order(self, user: User) -> Tuple[Optional[Product], Optional[datetime.datetime]]:
        """Generate prediction using the enhanced heuristic function."""
        recent_period = timezone.now() - timedelta(days=self.evaluation_window)
        products = Product.objects.filter(
            order__user=user,
            order__order_date__gte=recent_period
        ).distinct()
        
        best_score = -1
        best_product = None
        best_date = None
        
        for product in products:
            score, predicted_date = self.calculate_heuristic(product, user, timezone.now())
            
            if score > best_score:
                best_score = score
                best_product = product
                best_date = predicted_date
        
        return best_product, best_date

def predict_next_order(user: User) -> Tuple[Optional[Product], Optional[datetime.datetime]]:
    """Main prediction function that initializes and runs the prediction algorithm"""
    predictor = OrderPredictor()
    return predictor.predict_next_order(user)