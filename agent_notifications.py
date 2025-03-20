import os
import json
import logging
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_notifications.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AgentNotifications")

class EmailNotifier:
    """Send notifications via email."""
    
    def __init__(self):
        # Get email configuration from environment variables
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.from_email = os.getenv('FROM_EMAIL', self.smtp_username)
        
        # Validate configuration
        if not all([self.smtp_server, self.smtp_port, self.smtp_username, self.smtp_password]):
            logger.warning("Email notifications enabled but configuration incomplete")
    
    def send_notification(self, agents: List[Dict[str, Any]], subject: str, message: str) -> None:
        """Send email notifications to agents."""
        if not all([self.smtp_server, self.smtp_port, self.smtp_username, self.smtp_password]):
            logger.error("Cannot send email notification: configuration incomplete")
            return
        
        # Extract email addresses for agents
        email_addresses = [agent.get('email') for agent in agents if agent.get('email')]
        
        if not email_addresses:
            logger.warning("No agent email addresses available")
            return
        
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
            
            # Connect to SMTP server
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email notification sent to {len(email_addresses)} agents")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            raise

class NotificationManager:
    """
    Manages notifications to human agents when a conversation requires their attention.
    """
    
    def __init__(self):
        # Load agent contact information
        self.agents = self._load_agents()
        self.email_notifier = EmailNotifier()
        logger.info(f"NotificationManager initialized with {len(self.agents)} agents")
    
    def _load_agents(self) -> List[Dict[str, Any]]:
        """Load agent contact information from JSON file or environment variable."""
        try:
            # Try to load from file first
            agents_file = os.getenv('AGENTS_FILE', 'agents.json')
            if os.path.exists(agents_file):
                with open(agents_file, 'r') as f:
                    return json.load(f)
            
            # Fall back to parsing from environment variable
            agents_json = os.getenv('AGENTS_JSON', '[]')
            return json.loads(agents_json)
        
        except Exception as e:
            logger.error(f"Error loading agents: {e}")
            return []
    
    def _get_available_agents(self, priority: str = 'normal') -> List[Dict[str, Any]]:
        return [agent for agent in self.agents if agent.get('active', True)]
    
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
        
        # Get available agents for this priority
        agents = self._get_available_agents(priority)
        
        if not agents:
            logger.warning(f"No available agents to notify for case {ref_id}")
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
            self.email_notifier.send_notification(agents, subject, message)
            logger.info(f"Email notifications sent for case {ref_id} to {len(agents)} agents")
            return True
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    def notify_in_background(self, case_data: Dict[str, Any]) -> None:
        """Send notifications in a background thread to avoid blocking."""
        thread = threading.Thread(target=self.notify_new_case, args=(case_data,))
        thread.daemon = True
        thread.start()

# Create a singleton instance
notification_manager = NotificationManager()

# Function to use in messenger_webhook.py
def notify_agents_about_case(case_data):
    """Utility function to notify agents about a new case."""
    notification_manager.notify_in_background(case_data)