import sys
import os

# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the user's site-packages directory to the path
sys.path.append(os.path.expanduser('~/.local/lib/python3.10/site-packages'))

# Add your application directory to the system path
sys.path.append(CURRENT_DIR)

# Import your Flask app
from messenger_webhook import app as application
