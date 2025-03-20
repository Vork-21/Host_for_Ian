# CP Chatbot for Facebook Messenger

A Facebook Messenger chatbot powered by Claude AI that conducts initial screening for potential Cerebral Palsy (CP) cases. This system automates the intake process, asks relevant medical questions, evaluates eligibility, ranks potential cases, and seamlessly transitions to human agents when appropriate.

## System Overview

This system includes:

1. **AI-Powered Intake Process** - Utilizes Claude AI to understand and process natural language responses from potential clients.
2. **Eligibility Screening** - Checks case details against legal criteria including state-specific Statute of Limitations.
3. **Case Ranking** - Scores and prioritizes cases based on medical factors.
4. **Agent Notification System** - Alerts human agents via email when cases require intervention.
5. **Facebook Messenger Integration** - Connects the system to Facebook Messenger for client communication.
6. **Flexible Deployment** - Supports local development with ngrok and production deployment via Dreamhost.

## Component Descriptions

### Core Files

- **messenger_webhook.py**: Main Flask application that handles webhook events from Facebook Messenger.
- **Chat_Deploy.py**: Core chatbot logic with Claude AI integration, conversation management, and case evaluation.
- **agent_notifications.py**: System for notifying human agents about cases that need attention.
- **local_test.py**: Local development server with ngrok integration for webhook testing.
- **passenger_wsgi.py**: WSGI configuration for Dreamhost deployment.
- **setup.sh**: Setup script that configures the environment and dependencies.

### Configuration Files

- **criteria.json**: Contains legal rules, state-specific SOL (Statute of Limitations) information, and exclusion criteria.
- **agents.json**: Contact information for human agents.
- **.env**: Environment variables for API keys, tokens, and application settings (created by setup.sh).

## Installation

### Prerequisites

- Python 3.8 or higher
- Facebook Developer account with a configured Messenger app
- Anthropic API key for Claude AI
- SMTP server access for agent notifications

### Setup Steps

1. Clone this repository:
```bash
git clone https://github.com/yourusername/cp-chatbot.git
cd cp-chatbot
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

3. Edit the `.env` file with your API keys and configuration:
```
# Anthropic API Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
MODEL_VERSION=claude-3-5-sonnet-20241022

# Facebook Messenger Configuration
MESSENGER_VERIFY_TOKEN=your_verification_token
PAGE_ACCESS_TOKEN=your_facebook_page_access_token
APP_SECRET=your_facebook_app_secret

# Email Notifications
ENABLE_EMAIL_NOTIFICATIONS=true
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=your_email@example.com
SMTP_PASSWORD=your_app_password
FROM_EMAIL=your_email@example.com
```

4. Update `agents.json` with your team's contact information.

5. Review and update `criteria.json` with your specific legal criteria.

6. Install dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Local Development

1. Start the local development server:
```bash
python local_test.py
```

2. Use ngrok to expose your local server to the internet:
```bash
ngrok http 5000
```

3. Configure your Facebook webhook with the ngrok HTTPS URL (e.g., `https://your-ngrok-subdomain.ngrok.io/webhook`) and the verification token from your `.env` file.

## Dreamhost Deployment

1. Set up a Dreamhost account and create a new domain or subdomain.

2. Enable Python passenger for your domain in the Dreamhost panel.

3. Upload all files to your Dreamhost directory structure.

4. Create a Python virtual environment on Dreamhost:
```bash
cd ~/yourdomain.com
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

5. Configure your Facebook webhook to point to your Dreamhost domain (e.g., `https://yourdomain.com/webhook`).

6. Create a `tmp` directory in your Dreamhost domain's root folder to support passenger restarts:
```bash
mkdir -p tmp
```

7. To restart the application after changes, touch the `tmp/restart.txt` file:
```bash
touch tmp/restart.txt
```

## How the System Works

### Initial Intake Flow

1. User messages the Facebook page
2. Chatbot responds with initial CP screening question
3. If user indicates their child may have CP, the chatbot proceeds with structured questions:
   - Child's age
   - Pregnancy details and birth complications
   - NICU stay information
   - Medical interventions (brain scans, HIE therapy)
   - Developmental milestones
   - Previous legal consultations
   - State of birth

### Case Evaluation Logic

1. Checks age against state-specific Statute of Limitations
2. Automatically disqualifies cases from excluded states
3. Awards points based on medical factors:
   - Prematurity (gestational age < 36 weeks)
   - Difficult delivery
   - NICU stay and duration
   - HIE/cooling therapy
   - Brain imaging
   - Developmental delays

### Transition to Human Agents

The chatbot transitions to a human agent when:
- The user completes all screening questions
- The case is determined to be ineligible (for explanation)
- The chatbot encounters an error processing responses
- After 3 empty responses from the user

### Agent Notification

When a case transitions to a human agent:
1. Case data is saved with a unique reference ID
2. The system evaluates the case priority (low, normal, high, very high)
3. Available agents are notified via email with case details
4. The user receives a confirmation message with their reference ID

## Advanced Configuration

### Customizing Criteria

Edit `criteria.json` to modify:
- State-specific Statute of Limitations rules
- Excluded states list
- Other eligibility criteria

### Modifying Questions

The conversation flow is defined in the `ConversationManager` class in `Chat_Deploy.py`. Modify the `phases` dictionary to change questions or add new phases.

### Claude AI Model

The system uses Claude 3.5 Sonnet by default. You can change the model by updating the `MODEL_VERSION` in your `.env` file.

## Troubleshooting

### Facebook Webhook Issues

- Verify your `PAGE_ACCESS_TOKEN` and `APP_SECRET` are correct
- Ensure your webhook URL is accessible from the internet
- Check logs in `messenger_webhook.log` for detailed error information

### AI Response Issues

- Verify your `ANTHROPIC_API_KEY` is valid
- Check logs in `cp_chatbot.log` for Claude API error details
- Ensure your model version is correct in the `.env` file

### Agent Notification Issues

- Verify SMTP settings in your `.env` file
- Check logs in `agent_notifications.log` for email sending errors
- Ensure `agents.json` contains valid email addresses

## License

[Your License Information]

## Support

For support, please contact [Your Contact Information]
