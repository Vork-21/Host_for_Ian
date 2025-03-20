import sys
import os

# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to your virtual environment Python interpreter
INTERP = os.path.join(os.environ['HOME'], os.path.basename(CURRENT_DIR), 'venv', 'bin', 'python3')

# If we're not already using the virtual environment Python, restart with it
if sys.executable != INTERP:
    os.execl(INTERP, INTERP, *sys.argv)

# Add your application directory to the system path
sys.path.append(CURRENT_DIR)

# Import your Flask app
from messenger_webhook import app as application