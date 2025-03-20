import json
import os
import logging
import time
import hmac
import hashlib
import requests
import threading
import random
import string
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from typing import Dict, Any, Optional
from datetime import datetime

# Import your existing CP chatbot modules
from Chat_Deploy import ClaudeChat, ConversationManager, ModelConfiguration

# Import agent notification system
from agent_notifications import notify_agents_about_case

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("messenger_webhook.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MessengerWebhook")

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Initialize the Claude chatbot components
api_key = os.getenv('ANTHROPIC_API_KEY')
model_config = ModelConfiguration()
examples_dir = os.getenv('EXAMPLES_DIR', './examples')
criteria_file = os.getenv('CRITERIA_FILE', './criteria.json')

# Messenger configuration
VERIFY_TOKEN = os.getenv('MESSENGER_VERIFY_TOKEN')
PAGE_ACCESS_TOKEN = os.getenv('PAGE_ACCESS_TOKEN')
APP_SECRET = os.getenv('APP_SECRET')

# Active conversations
active_conversations = {}

class MessengerSession:
    """
    Manages a single conversation session with a Messenger user.
    """
    def __init__(self, sender_id: str):
        self.sender_id = sender_id
        self.claude_client = ClaudeChat(api_key)
        
        # Initialize conversation manager with legal rules
        self.claude_client.initialize_rules(criteria_file)
        
        # Conversation state
        self.conversation_active = True
        self.last_activity = time.time()
        self.handled_by_agent = False
        self.agent_id = None
        
        logger.info(f"New session created for sender {sender_id}")
    
    def process_message(self, message_text: str) -> None:
        """Process a message from the user and generate a response."""
        if not self.conversation_active:
            return
            
        # Update activity timestamp
        self.last_activity = time.time()
        
        # Check if this conversation is now handled by an agent
        if self.handled_by_agent:
            # Store the message in DB for agent to see
            self._store_message_for_agent(message_text)
            # Send acknowledgment to user
            self._send_message("Your message has been received. An agent will respond shortly.")
            return
        
        # Process the message using Claude's conversation manager
        try:
            # Rather than using the terminal interface, we feed the message directly
            response_data = self.claude_client.conversation_manager.analyze_response(message_text)
            
            # Handle errors
            if 'error' in response_data:
                self._send_message(response_data['error'])
                return
                
            # Handle case ineligibility
            if 'eligible' in response_data and not response_data['eligible']:
                self._send_message(response_data['reason'])
                self._transition_to_agent("Case ineligible - needs human review")
                return
                
            # Handle ending the chat (e.g., after lawyer question)
            if response_data.get('end_chat'):
                farewell_message = response_data.get('farewell_message', "Thank you for your time.")
                self._send_message(farewell_message)
                self.conversation_active = False
                return
            
            # Get the next question to ask
            next_question, is_control = self.claude_client.conversation_manager.get_next_question()
            
            # Handle control messages
            if is_control:
                if self.claude_client.conversation_manager.empty_response_count >= 3:
                    self._send_message(next_question)
                    # We'll wait for their response in the next message
                    return
            
            # If we've completed all questions, transition to an agent
            if self.claude_client.conversation_manager.current_phase == 'complete':
                self._send_message(next_question)  # Send the scheduling message
                
                # Save the case data
                self.claude_client.conversation_manager.save_case_data()
                
                # Transition to agent with case ranking
                ranking = self.claude_client.conversation_manager.case_data.get('ranking', 'normal')
                points = self.claude_client.conversation_manager.case_data.get('points', 50)
                self._transition_to_agent(f"Case completed - {ranking} priority ({points} points)")
                return
            
            # Otherwise, send the next question
            self._send_message(next_question)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self._send_message("I'm having trouble processing your message. Let me connect you with a live representative.")
            self._transition_to_agent("Error processing message")
    
    def _send_message(self, message_text: str) -> bool:
        """Send a message to the user via Messenger."""
        try:
            url = f"https://graph.facebook.com/v18.0/me/messages"
            payload = {
                "recipient": {"id": self.sender_id},
                "message": {"text": message_text}
            }
            params = {"access_token": PAGE_ACCESS_TOKEN}
            response = requests.post(url, json=payload, params=params)
            
            if response.status_code != 200:
                logger.error(f"Failed to send message: {response.text}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def _store_message_for_agent(self, message_text: str) -> None:
        """Store a message in the database for an agent to view."""
        # In a production environment, this would save to your database
        # For now, we'll just log it
        logger.info(f"Message for agent from {self.sender_id}: {message_text}")
        
        # TODO: Implement database storage for agent dashboard
    
    def _transition_to_agent(self, reason: str) -> None:
        """Transition this conversation to be handled by a human agent via Facebook Inbox."""
        self.handled_by_agent = True
        
        # Log the transition for monitoring purposes
        logger.info(f"Conversation {self.sender_id} transitioned to agent. Reason: {reason}")
        
        # Save case data to your database for agents to reference
        try:
            self.claude_client.conversation_manager.save_case_data()
            
            # Get case details for the notification
            case_data = self.claude_client.conversation_manager.case_data
            age = case_data.get('age', 'Unknown')
            state = case_data.get('state', 'Unknown')
            ranking = case_data.get('ranking', 'normal')
            points = case_data.get('points', 0)
            
            # Create a case reference ID for tracking
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            ref_id = f"{timestamp[-6:]}_{random_suffix}"
            
            # Prepare case data for agent notification
            notification_data = {
                'sender_id': self.sender_id,
                'ref_id': ref_id,
                'age': age,
                'state': state,
                'ranking': ranking,
                'points': points,
                'timestamp': timestamp,
                'reason': reason
            }
            
            # Notify the user with case reference
            self._send_message(f"Thank you for providing your information. A team member will review your case (Ref: #{ref_id}) and continue this conversation shortly.")
            
            # Send notifications to appropriate agents
            notify_agents_about_case(notification_data)
            
        except Exception as e:
            logger.error(f"Error during agent transition: {e}")
            self._send_message("I'll connect you with a representative who can help you further. They'll review your information and respond shortly.")
    
    def send_welcome_message(self) -> None:
        """Send the initial welcome message to start the conversation."""
        initial_question = self.claude_client.conversation_manager.phases['initial']['question']
        self._send_message(initial_question)

def verify_fb_signature(request_data, signature_header):
    """Verify that the request is properly signed by Facebook."""
    if not APP_SECRET:
        logger.warning("APP_SECRET not configured, skipping signature verification")
        return True
        
    if not signature_header:
        logger.warning("No signature header provided")
        return False
    
    # The signature header format is "sha1=<signature>"
    elements = signature_header.split('=')
    if len(elements) != 2:
        logger.warning(f"Invalid signature format: {signature_header}")
        return False
        
    signature = elements[1]
    
    # Calculate expected signature
    expected_signature = hmac.new(
        bytes(APP_SECRET, 'utf-8'),
        msg=request_data,
        digestmod=hashlib.sha1
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)

def cleanup_inactive_sessions():
    """
    Periodically clean up inactive sessions to prevent memory leaks.
    """
    while True:
        try:
            current_time = time.time()
            inactive_threshold = 3600  # 1 hour
            
            to_remove = []
            for sender_id, session in active_conversations.items():
                if current_time - session.last_activity > inactive_threshold:
                    to_remove.append(sender_id)
            
            for sender_id in to_remove:
                logger.info(f"Removing inactive session for {sender_id}")
                del active_conversations[sender_id]
                
            # Sleep for 15 minutes before next cleanup
            time.sleep(900)
        except Exception as e:
            logger.error(f"Error in cleanup thread: {e}")
            time.sleep(900)  # Still sleep on error

# Start the cleanup thread when the module loads
cleanup_thread = threading.Thread(target=cleanup_inactive_sessions, daemon=True)
cleanup_thread.start()

@app.route('/webhook', methods=['GET'])
def verify_webhook():
    """
    Handle the initial webhook verification from Facebook.
    """
    mode = request.args.get('hub.mode', '')
    token = request.args.get('hub.verify_token', '')
    challenge = request.args.get('hub.challenge', '')
    
    if mode == 'subscribe' and token == VERIFY_TOKEN:
        logger.info("Webhook verified")
        return challenge
        
    logger.warning(f"Webhook verification failed. Mode: {mode}, Token: {token}")
    return 'Verification Failed', 403

@app.route('/webhook', methods=['POST'])
def webhook():
    """
    Handle incoming webhook events from Facebook Messenger.
    """
    # Verify the request signature
    signature = request.headers.get('X-Hub-Signature', '')
    if not verify_fb_signature(request.get_data(), signature):
        logger.warning("Invalid request signature")
        return 'Invalid signature', 403
    
    data = request.json
    
    # Check if this is a page subscription
    if data.get('object') != 'page':
        return 'Not a page subscription', 404
    
    # Process each entry (there may be multiple)
    for entry in data.get('entry', []):
        # Process each messaging event
        for messaging_event in entry.get('messaging', []):
            sender_id = messaging_event.get('sender', {}).get('id')
            
            if not sender_id:
                continue
                
            # Check for message event
            if 'message' in messaging_event:
                message_text = messaging_event.get('message', {}).get('text')
                
                if not message_text:
                    # Handle non-text messages (e.g., attachments)
                    continue
                
                # Get or create a session for this sender
                if sender_id not in active_conversations:
                    active_conversations[sender_id] = MessengerSession(sender_id)
                    # Send welcome message to start conversation
                    active_conversations[sender_id].send_welcome_message()
                else:
                    # Process the message in an existing conversation
                    active_conversations[sender_id].process_message(message_text)
            
            # Handle postback events (e.g., button clicks)
            elif 'postback' in messaging_event:
                payload = messaging_event.get('postback', {}).get('payload')
                
                if not payload:
                    continue
                
                # Get or create a session for this sender
                if sender_id not in active_conversations:
                    active_conversations[sender_id] = MessengerSession(sender_id)
                
                # Process the postback as if it were a text message
                active_conversations[sender_id].process_message(payload)
    
    return 'Success', 200

@app.route('/', methods=['GET'])
def home():
    """Simple endpoint to verify the server is running."""
    return 'CP Chatbot Messenger Webhook is running!'

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'ok',
        'active_conversations': len(active_conversations)
    })

if __name__ == '__main__':
    # Check if required environment variables are set
    if not VERIFY_TOKEN:
        logger.error("MESSENGER_VERIFY_TOKEN not set. Please update your .env file.")
        exit(1)
        
    if not PAGE_ACCESS_TOKEN:
        logger.error("PAGE_ACCESS_TOKEN not set. Please update your .env file.")
        exit(1)
    
    # Start the Flask development server
    # In production, use a proper WSGI server like Gunicorn
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)