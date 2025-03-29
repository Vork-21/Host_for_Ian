import os
import json
import logging
import smtplib
import threading
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging - file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_notifications.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AgentNotifications")

# Add direct console output for critical debugging (works even if logging fails)
def debug_print(message):
    """Print directly to stderr to ensure output regardless of logging configuration"""
    print(f"DEBUG: {message}", file=sys.stderr)
    sys.stderr.flush()  # Ensure output is written immediately

class EmailNotifier:
    """Send notifications via email."""
    
    def __init__(self):
        # Get email configuration from environment variables
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.dreamhost.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.from_email = os.getenv('FROM_EMAIL', self.smtp_username)
        
        # Validate configuration with direct console output
        if not all([self.smtp_server, self.smtp_port, self.smtp_username, self.smtp_password]):
            msg = "Email notifications enabled but configuration incomplete"
            logger.warning(msg)
            debug_print(msg)
        else:
            debug_print(f"Email configuration loaded: Server={self.smtp_server}, Port={self.smtp_port}, User={self.smtp_username}")
    
    def send_notification(self, agents: List[Dict[str, Any]], subject: str, message: str) -> None:
        """Send email notifications to agents with improved error handling."""
        if not all([self.smtp_server, self.smtp_port, self.smtp_username, self.smtp_password]):
            msg = "Cannot send email notification: configuration incomplete"
            logger.error(msg)
            debug_print(msg)
            return
        
        # Extract email addresses for agents
        email_addresses = [agent.get('email') for agent in agents if agent.get('email')]
        
        if not email_addresses:
            msg = "No agent email addresses available"
            logger.warning(msg)
            debug_print(msg)
            return
        
        debug_print(f"Preparing to send notification to {len(email_addresses)} agents")
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['Subject'] = subject
            
            # Use BCC for privacy
            msg['To'] = self.from_email
            msg['Bcc'] = ', '.join(email_addresses)
            
            # Add message body
            msg.attach(MIMEText(message, 'plain'))
            
            debug_print(f"Connecting to SMTP server: {self.smtp_server}:{self.smtp_port}")
            
            # Connect to SMTP server with enhanced error handling and debugging
            try:
                # Create SMTP connection
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.set_debuglevel(2)  # Verbose output to help diagnose issues
                
                debug_print("Starting TLS handshake")
                server.starttls()
                server.ehlo_or_helo_if_needed()  # Ensure proper handshake after TLS
                
                debug_print(f"Authenticating as {self.smtp_username}")
                server.login(self.smtp_username, self.smtp_password)
                
                debug_print("Sending email message")
                server.send_message(msg)
                server.quit()
                
                msg_sent = f"Email notification sent to {len(email_addresses)} agents"
                logger.info(msg_sent)
                debug_print(msg_sent)
                
            except smtplib.SMTPAuthenticationError as e:
                error_msg = f"SMTP Authentication Error: {e}"
                logger.error(error_msg)
                debug_print(error_msg)
                debug_print("Check your username and password")
                raise
                
            except smtplib.SMTPConnectError as e:
                error_msg = f"SMTP Connection Error: {e}"
                logger.error(error_msg)
                debug_print(error_msg)
                debug_print(f"Unable to connect to {self.smtp_server}:{self.smtp_port}")
                raise
                
            except smtplib.SMTPException as e:
                error_msg = f"SMTP Error: {e}"
                logger.error(error_msg)
                debug_print(error_msg)
                raise
                
        except Exception as e:
            error_msg = f"Unexpected error sending email: {str(e)}"
            logger.error(error_msg)
            debug_print(error_msg)
            raise

class NotificationManager:
    """
    Manages notifications to human agents when a conversation requires their attention.
    """
    
    def __init__(self):
        # Load agent contact information
        self.agents = self._load_agents()
        self.email_notifier = EmailNotifier()
        init_msg = f"NotificationManager initialized with {len(self.agents)} agents"
        logger.info(init_msg)
        debug_print(init_msg)
    
    def _load_agents(self) -> List[Dict[str, Any]]:
        """Load agent contact information from JSON file or environment variable."""
        try:
            # Try to load from file first
            agents_file = os.getenv('AGENTS_FILE', 'agents.json')
            if os.path.exists(agents_file):
                with open(agents_file, 'r') as f:
                    agents = json.load(f)
                    debug_print(f"Loaded {len(agents)} agents from {agents_file}")
                    return agents
            
            # Fall back to parsing from environment variable
            agents_json = os.getenv('AGENTS_JSON', '[]')
            agents = json.loads(agents_json)
            debug_print(f"Loaded {len(agents)} agents from environment variable")
            return agents
        
        except Exception as e:
            error_msg = f"Error loading agents: {e}"
            logger.error(error_msg)
            debug_print(error_msg)
            return []
    
    def _get_available_agents(self, priority: str = 'normal') -> List[Dict[str, Any]]:
        available = [agent for agent in self.agents if agent.get('active', True)]
        debug_print(f"Found {len(available)} available agents")
        return available
    
    def notify_new_case(self, case_data: Dict[str, Any]) -> bool:
        """
        Notify appropriate agents about a new case that needs attention.
        Returns True if notifications were sent successfully.
        """
        # Extract case information
        sender_id = case_data.get('sender_id', 'Unknown')
        ref_id = case_data.get('ref_id', 'Unknown')
        priority = case_data.get('ranking', 'normal').lower()
        age = case_data.get('age', 'Unknown')
        state = case_data.get('state', 'Unknown')
        
        debug_print(f"Processing notification for case {ref_id} with {priority} priority")
        
        # Get available agents for this priority
        agents = self._get_available_agents(priority)
        
        if not agents:
            msg = f"No available agents to notify for case {ref_id}"
            logger.warning(msg)
            debug_print(msg)
            return False
        
        # Build notification message
        subject = f"New CP Case: #{ref_id} ({priority.upper()} Priority)"
        message = f"""
        NEW CP CASE REQUIRES ATTENTION

        Reference: #{ref_id}
        Priority: {priority.upper()}
        Age: {age}
        State: {state}

        This case has been automatically pre-qualified and requires your attention.
        Please log in to Facebook Business Suite to respond:
        https://business.facebook.com/latest/inbox
        """
        
        # Send notification via email
        try:
            debug_print(f"Attempting to send email notification for case {ref_id}")
            self.email_notifier.send_notification(agents, subject, message)
            success_msg = f"Email notifications sent for case {ref_id} to {len(agents)} agents"
            logger.info(success_msg)
            debug_print(success_msg)
            return True
        except Exception as e:
            error_msg = f"Error sending email notification: {e}"
            logger.error(error_msg)
            debug_print(error_msg)
            return False
    
    def notify_in_background(self, case_data: Dict[str, Any]) -> None:
        """Send notifications in a background thread to avoid blocking."""
        debug_print(f"Starting background notification thread for case {case_data.get('ref_id', 'Unknown')}")
        thread = threading.Thread(target=self.notify_new_case, args=(case_data,))
        thread.daemon = True
        thread.start()

# Create a singleton instance
notification_manager = NotificationManager()

# Function to use in messenger_webhook.py
def notify_agents_about_case(case_data):
    """Utility function to notify agents about a new case."""
    debug_print(f"notify_agents_about_case called with case ID: {case_data.get('ref_id', 'Unknown')}")
    notification_manager.notify_in_background(case_data)

# Simple test function to verify SMTP settings directly
def test_smtp_connection():
    """Test SMTP connection directly to verify settings"""
    debug_print("Running SMTP connection test")
    
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.dreamhost.com')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    smtp_username = os.getenv('SMTP_USERNAME', '')
    smtp_password = os.getenv('SMTP_PASSWORD', '')
    from_email = os.getenv('FROM_EMAIL', smtp_username)
    
    debug_print(f"Using settings: Server={smtp_server}, Port={smtp_port}, User={smtp_username}")
    
    try:
        # Create a simple test message
        msg = MIMEText("Test message from CP Chatbot")
        msg['Subject'] = "SMTP Test"
        msg['From'] = from_email
        msg['To'] = from_email
        
        # Connect to server with debugging
        debug_print(f"Connecting to {smtp_server}:{smtp_port}")
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.set_debuglevel(2)
        debug_print("Starting TLS")
        server.starttls()
        debug_print(f"Authenticating as {smtp_username}")
        server.login(smtp_username, smtp_password)
        debug_print("Sending test message")
        server.sendmail(from_email, from_email, msg.as_string())
        server.quit()
        debug_print("SMTP test successful!")
        return True
    except Exception as e:
        debug_print(f"SMTP test failed: {str(e)}")
        return False

# Run a test if this file is executed directly
if __name__ == "__main__":
    debug_print("Testing agent notification system")
    test_smtp_connection()