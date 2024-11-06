from django.shortcuts import render

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .utils import predict_next_order
from .models import Order, Product
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

from datetime import date

@login_required
def dashboard(request):
    user = request.user
    orders = Order.objects.filter(user=user).order_by('-order_date')

    for order in orders:
        order.total_price = order.product.price * order.quantity 

        if order.order_date.date() <= date.today():  
            order.status = "Fulfilled"
        else:
            order.status = "Processing"

    predicted_product, predicted_date = predict_next_order(user)

    return render(request, 'dashboard.html', {
        'orders': orders,
        'predicted_product': predicted_product,
        'predicted_date': predicted_date,
    })

def login_view(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            return render(request, 'login.html', {'form': form, 'error': 'Invalid credentials'})

    return render(request, 'login.html')

@login_required
def logout_view(request):
    logout(request)
    return redirect('login')
