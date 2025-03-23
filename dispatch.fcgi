#!/usr/bin/env python3

import sys, os

# Add virtual environment site packages to path
venv_path = os.path.expanduser('~/chatbot-venv')
site_packages = os.path.join(venv_path, 'lib', 'python3.10', 'site-packages')
sys.path.insert(0, site_packages)

# Add the application directory to the path
sys.path.insert(0, os.path.dirname(__file__))

# Import your Flask app
from messenger_webhook import app

# Run with FastCGI
from flup.server.fcgi import WSGIServer
if __name__ == '__main__':
    WSGIServer(app).run()
