from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.core.cache import cache
from django.db.models import Sum, F, Count, Q
from django.utils import timezone
from django.contrib import messages
from datetime import date, timedelta
from .utils import predict_next_order
from .models import Order, Product
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_order_statistics(user) -> Dict[str, Any]:
    """Calculate order statistics for the dashboard."""
    now = timezone.now()
    thirty_days_ago = now - timedelta(days=30)
    
    # Get basic statistics
    total_orders = Order.objects.filter(user=user).count()
    recent_orders = Order.objects.filter(
        user=user,
        order_date__gte=thirty_days_ago
    ).count()
    
    # Calculate total spent
    orders = Order.objects.filter(user=user).select_related('product')
    total_spent = sum(order.quantity * order.product.price for order in orders)
    
    # Get frequently ordered products
    frequent_products = (Order.objects.filter(user=user)
                        .values('product__name')
                        .annotate(order_count=Count('id'))
                        .order_by('-order_count')[:3])
    
    return {
        'total_orders': total_orders,
        'total_spent': total_spent,
        'recent_orders': recent_orders,
        'frequent_products': frequent_products
    }

def get_prediction_data(user) -> Dict[str, Any]:
    """Get prediction data with caching."""
    cache_key = f'prediction_data_{user.id}'
    prediction_data = cache.get(cache_key)
    
    if prediction_data is None:
        try:
            predicted_product, predicted_date = predict_next_order(user)
            
            if predicted_product and predicted_date:
                prediction_data = {
                    'product': predicted_product,
                    'date': predicted_date,
                    'confidence': 'High' if predicted_date - timezone.now() < timedelta(days=7) else 'Medium',
                    'previous_price': predicted_product.price,
                }
            else:
                prediction_data = None
                
            # Cache for 1 hour
            cache.set(cache_key, prediction_data, 3600)
        except Exception as e:
            logger.error(f"Error getting prediction data: {str(e)}")
            prediction_data = None
            
    return prediction_data

@login_required
def dashboard(request):
    """Enhanced dashboard view with predictions and statistics."""
    try:
        user = request.user
        
        # Get orders with efficient querying
        orders = (Order.objects.filter(user=user)
                 .select_related('product')  # Prevent N+1 queries
                 .order_by('-order_date'))
        
        # Calculate order totals and status
        for order in orders:
            order.total_price = order.product.price * order.quantity
            order.status = "Fulfilled" if order.order_date.date() <= date.today() else "Processing"
        
        # Get order statistics
        statistics = get_order_statistics(user)
        
        # Get prediction data
        prediction_data = get_prediction_data(user)
        
        context = {
            'orders': orders,
            'statistics': statistics,
            'prediction_data': prediction_data,
            'last_order_date': orders.first().order_date if orders.exists() else None,
            'total_items': orders.count(),
        }
        
        return render(request, 'dashboard.html', context)
        
    except Exception as e:
        logger.error(f"Dashboard error for user {request.user.id}: {str(e)}")
        messages.error(request, "There was an error loading your dashboard. Please try again later.")
        return render(request, 'dashboard.html', {'error': True})

@login_required
def logout_view(request):
    """Handle user logout."""
    try:
        # Clear user-specific cache
        cache.delete(f'prediction_data_{request.user.id}')
        logout(request)
        messages.success(request, "You have been successfully logged out.")
        return redirect('login')
    except Exception as e:
        logger.error(f"Logout error for user {request.user.id}: {str(e)}")
        messages.error(request, "There was an error during logout. Please try again.")
        return redirect('dashboard')

def login_view(request):
    """Handle user login with enhanced security and validation."""
    if request.user.is_authenticated:
        return redirect('dashboard')
        
    if request.method == "POST":
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        
        if not username or not password:
            messages.error(request, "Please provide both username and password.")
            return render(request, 'login.html')
            
        try:
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {user.username}!")
                
                # Get or create prediction data in background
                get_prediction_data(user)
                
                return redirect('dashboard')
            else:
                messages.error(request, "Invalid username or password.")
        except Exception as e:
            logger.error(f"Login error for username {username}: {str(e)}")
            messages.error(request, "An error occurred during login. Please try again.")
            
    return render(request, 'login.html')