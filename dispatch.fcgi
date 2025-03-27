#!/usr/bin/env python3

import sys, os
import logging

# Set up basic logging at the very beginning
log_file = os.path.expanduser('~/fcgi_error.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file,
    filemode='a'
)
logging.debug("=== dispatch.fcgi starting ===")

try:
    logging.debug("Setting up Python paths")
    # Add virtual environment site packages to path
    venv_path = os.path.expanduser('~/chatbot-venv')
    site_packages = os.path.join(venv_path, 'lib', 'python3.10', 'site-packages')
    logging.debug(f"Adding site-packages from: {site_packages}")
    sys.path.insert(0, site_packages)

    # Add the application directory to the path
    app_dir = os.path.dirname(os.path.abspath(__file__))
    logging.debug(f"Adding application directory: {app_dir}")
    sys.path.insert(0, app_dir)

    # Log environment variables for debugging
    logging.debug(f"Current working directory: {os.getcwd()}")
    logging.debug(f"PYTHONPATH: {sys.path}")
    
    # Log environment variables (without sensitive values)
    env_vars = [key for key in os.environ.keys()]
    logging.debug(f"Environment variables available: {env_vars}")
    
    # Import your Flask app
    logging.debug("Importing Flask app from messenger_webhook")
    from messenger_webhook import app
    logging.debug("Successfully imported Flask app")

    # Run with FastCGI
    logging.debug("Importing WSGIServer from flup")
    from flup.server.fcgi import WSGIServer
    logging.debug("Starting WSGIServer")
    
    if __name__ == '__main__':
        logging.debug("About to run WSGIServer...")
        WSGIServer(app).run()
except Exception as e:
    logging.exception(f"Fatal error in dispatch.fcgi: {e}")
    # Print a basic error message as FCGI response
    print("Status: 500 Internal Server Error")
    print("Content-Type: text/html\n")
    print("<html><body><h1>500 Internal Server Error</h1><p>An application error occurred.</p></body></html>")
    sys.exit(1)