from django.shortcuts import render

# Create your views here.
# recognition/views.py
from django.shortcuts import render, redirect
from django.http import HttpResponse
import subprocess

def index(request):
    return render(request, 'recognition/index.html')

def register(request):
    if request.method == 'POST':
        name = request.POST['name']
        subprocess.run(['python3', 'recognition/capture_faces.py', name])
        return redirect('index')
    return render(request, 'recognition/register.html')

def authenticate_view(request):
    if request.method == 'POST':
        subprocess.run(['python3', 'recognition/authenticate.py'])
        return redirect('index')
    return render(request, 'recognition/authenticate.html')
