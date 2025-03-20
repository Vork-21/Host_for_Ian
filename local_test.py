#!/usr/bin/env python3
"""
Local server runner for CP Chatbot Messenger integration.
This starts the Flask server for webhook testing with ngrok.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("local_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LocalServer")

def check_env_setup():
    """Check if environment variables are set up correctly"""
    load_dotenv()
    
    required_vars = [
        'ANTHROPIC_API_KEY',
        'PAGE_ACCESS_TOKEN',
        'APP_SECRET',
        'MESSENGER_VERIFY_TOKEN'
    ]
    
    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing.append(var)
    
    if missing:
        print("❌ The following required environment variables are missing or empty:")
        for var in missing:
            print(f"  - {var}")
        choice = input("Continue anyway? (y/n): ").strip().lower()
        if choice not in ['y', 'yes']:
            return False
    
    return True

def main():
    """Main function to run the local server"""
    if not check_env_setup():
        sys.exit(1)
    
    try:
        # Import the Flask app - we do this here to allow environment
        # variables to be loaded first
        print("🚀 Starting local Flask server for webhook testing...")
        print("This server can receive webhook events from Facebook Messenger.")
        print("To expose it to the internet, run ngrok in a separate terminal:")
        print("\nngrok http 5000\n")
        
        # Give instructions for using ngrok
        print("⚠️ IMPORTANT: After starting ngrok, use the HTTPS URL it provides")
        print("   as your webhook URL in the Facebook Developer Console.")
        print("   Example: https://abc123.ngrok.io/webhook")
        print("   Your verify token is:", os.getenv('MESSENGER_VERIFY_TOKEN'))
        print("\n🔌 Press Ctrl+C to stop the server\n")
        
        # Import and run the Flask app
        from messenger_webhook import app
        
        # Get the port from environment variable or use 5000 by default
        port = int(os.getenv('PORT', 5000))
        
        # Run the Flask development server
        app.run(host='0.0.0.0', port=port, debug=True)
        
    except ImportError as e:
        logger.error(f"Failed to import Flask app: {e}")
        print(f"❌ Error: {e}")
        print("Make sure messenger_webhook.py is in the current directory and all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting local server: {e}")
        print(f"❌ An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()