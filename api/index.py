import sys
import os

# Add parent directory to sys.path so modules in root can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Entry point handler for Vercel Serverless Function
handler = app
