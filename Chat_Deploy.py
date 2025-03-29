import json
import os
import re
import logging
import time
import sys
import signal
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime
import anthropic
from dotenv import load_dotenv
from numbers import Number

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cp_chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CPChatbot")


# At the top of the file, add:
try:
    import anthropic
    logger.info(f"Anthropic SDK version: {anthropic.__version__}")
except Exception as e:
    logger.error(f"Error importing anthropic: {e}")

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
MAX_RESPONSE_LENGTH = 5000  # characters
DEFAULT_TIMEOUT = 30  # seconds for API calls
DATA_DIRECTORY = "case_data"

class SafeAnthropicClient:
    """Wrapper for Anthropic client to handle initialization issues."""
    
    def __init__(self, api_key):
        try:
            # Import version to log it
            import anthropic
            logger.info(f"Using Anthropic SDK version: {anthropic.__version__}")
            
            # Directly import Client with fully qualified name
            from anthropic import Client
            
            # Create client with only the API key, no extra parameters
            self.client = Client(api_key=api_key)
            self.initialized = True
            logger.info("Anthropic client initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize Anthropic client: {e}")
            self.initialized = False
    
    def messages_create(self, *args, **kwargs):
        """Wrapper for client.messages.create to handle initialization failures."""
        if not hasattr(self, 'initialized') or not self.initialized:
            logger.error("Cannot create message: Anthropic client not initialized")
            # Return an error message instead of a dummy response
            from types import SimpleNamespace
            error_response = SimpleNamespace()
            error_content = SimpleNamespace()
            error_content.text = "Sorry, I'm having trouble connecting to my services right now. Please try again later."
            error_content.type = "text"
            error_response.content = [error_content]
            return error_response
        
        try:
            return self.client.messages.create(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error creating message: {e}")
            # Return an error message
            from types import SimpleNamespace
            error_response = SimpleNamespace()
            error_content = SimpleNamespace()
            error_content.text = "Sorry, I'm having trouble processing your request right now. Please try again later."
            error_content.type = "text"
            error_response.content = [error_content]
            return error_response

class ClaudeNLU:
    """
    Natural Language Understanding using Claude AI to interpret complex user responses
    in the CP case intake process.
    """
    def __init__(self, anthropic_client, model_version):
        self.client = anthropic_client
        self.model = model_version
        logger.info(f"Claude NLU initialized with model: {model_version}")
        
    def interpret_age(self, user_input: str) -> Optional[float]:
        """
        Extract age from natural language input.
        
        Example inputs:
        - "He is seven years old"
        - "She just turned 4 last month"
        - "About 5 and a half"
        - "18 months old"
        """
        if not user_input or len(user_input.strip()) == 0:
            return None
            
        prompt = """
        Extract the child's age in years from this text. Respond with ONLY a number.
        If the age includes partial years (like "5 and a half"), convert to a decimal (5.5).
        If the age is given in months (like "18 months"), convert to years (1.5).
        If you can't determine the age, respond with "unknown".
        """
        
        response = self._query_claude(prompt, user_input)
        
        try:
            # Try to convert the response to a float
            age = float(response.strip().replace(',', '.'))  # Handle comma as decimal separator
            return age
        except ValueError:
            # If Claude couldn't determine the age, return None
            return None
    
    def interpret_pregnancy_details(self, user_input: str) -> dict:
        """
        Extract gestational age and whether delivery was difficult.
        
        Example inputs:
        - "The delivery was not easy and the baby was born at thirty 4 weeks"
        - "I carried him for about 32 weeks and then had an emergency C-section"
        - "Full term but with some complications during delivery"
        """
        if not user_input:
            return {"weeks": None, "difficult_delivery": False}
            
        prompt = """
        Extract two pieces of information from this text about a child's birth:
        1. The number of weeks pregnant (gestational age) when the child was born
        2. Whether there was a difficult delivery

        Respond with ONLY a JSON object with two keys:
        - "weeks": number (or null if not mentioned)
        - "difficult_delivery": boolean (true if any indication of difficult/complicated/not easy delivery)

        Example response: {"weeks": 34, "difficult_delivery": true}
        """
        
        response = self._query_claude(prompt, user_input)
        
        try:
            # Parse the JSON response
            data = json.loads(response.strip())
            return data
        except json.JSONDecodeError:
            # Fallback if Claude doesn't return valid JSON
            logger.warning(f"Failed to parse JSON from Claude: {response}")
            # Try to extract information through regex
            import re
            
            # Extract weeks with regex
            weeks = None
            weeks_patterns = [
                r'(\d+)\s*(?:weeks|week|wks|wk)',  # Match "34 weeks", "34 week", etc.
                r'(\d+)\s*w',                      # Match "34w"
                r'(\d+)w',                         # Match "34w"
                r'(\w+)[- ](\d+)\s*(?:weeks|week|wks|wk)', # Match "thirty-4 weeks"
                r'(\w+)[- ](\d+)',                 # Match "thirty-4"
                r'full(?:\s*[ -]?term)?',          # Match "full term" (implying ~40 weeks)
                r'(?:term|mature)',                # Match "term" or "mature" (implying ~40 weeks)
            ]
            
            for pattern in weeks_patterns:
                match = re.search(pattern, user_input.lower())
                if match:
                    # Handle "full term" or "term" matches (set to 40 weeks)
                    if pattern in [r'full(?:\s*[ -]?term)?', r'(?:term|mature)']:
                        weeks = 40
                        break
                        
                    # If there are two groups, it might be something like "thirty-4"
                    if len(match.groups()) > 1 and match.group(2):
                        # Try to convert text number to digit
                        text_num = self._text_to_num(match.group(1))
                        if text_num:
                            try:
                                digit = int(match.group(2))
                                weeks = text_num * 10 + digit
                                break
                            except ValueError:
                                pass
                    else:
                        try:
                            weeks = int(match.group(1))
                            break
                        except ValueError:
                            pass
            
            # Check for difficult delivery
            difficult = False
            difficult_indicators = [
                'difficult', 'not easy', 'hard', 'complications', 'emergency', 
                'c-section', 'csection', 'c section', 'cesarean', 'forceps', 
                'vacuum', 'distress', 'oxygen', 'resuscitate', 'nicu', 
                'intensive care', 'problem', 'complication', 'issue',
                'prolonged', 'stuck', 'trauma', 'injury', 'monitor', 'fetal',
                'induced', 'induction', 'premature', 'preemie', 'breech'
            ]
            
            for indicator in difficult_indicators:
                if indicator in user_input.lower():
                    difficult = True
                    break
                    
            return {"weeks": weeks, "difficult_delivery": difficult}
    
    def interpret_yes_no(self, user_input: str, context: str = "") -> bool:
        """
        Determine if the response is affirmative or negative.
        
        Example inputs:
        - "Yes we did stay in the NICU"
        - "No, we didn't have to"
        - "We spent about 3 days there" (context-dependent)
        """
        if not user_input:
            return False
            
        # Quick check for simple yes/no
        user_input_lower = user_input.lower().strip()
        if user_input_lower in ['yes', 'yeah', 'yep', 'yup', 'sure', 'definitely', 'absolutely', 'correct']:
            return True
            
        if user_input_lower in ['no', 'nope', 'not', 'never', 'negative']:
            return False
            
        # More complex analysis needed
        prompt = f"""
        Determine if this response is affirmative (yes) or negative (no).
        Context about the question: {context}
        
        Respond with ONLY "yes" or "no".
        When in doubt and the response indicates any affirmative element, respond with "yes".
        """
        
        response = self._query_claude(prompt, user_input)
        
        # Return True if the response is "yes"
        return response.strip().lower() == "yes"
    
    def interpret_duration(self, user_input: str) -> int:
        """
        Extract a duration in days from natural language.
        
        Example inputs:
        - "About 2 weeks"
        - "3 days"
        - "A week and a half"
        - "2 months 3 days"
        """
        if not user_input:
            return 0
            
        prompt = """
        Extract the duration mentioned in this text and convert it to total days.
        Respond with ONLY the number of days as an integer.
        
        For example:
        - "2 weeks" → 14
        - "3 days" → 3
        - "a week and a half" → 10
        - "2 months and 5 days" → 65
        - "a couple of days" → 2
        - "a few weeks" → 21
        
        If you can't determine a specific duration, respond with "0".
        """
        
        response = self._query_claude(prompt, user_input)
        
        try:
            # Try to convert the response to an integer
            days = int(response.strip())
            return days
        except ValueError:
            # If Claude couldn't determine the duration, try regex parsing
            logger.warning(f"Failed to parse duration from Claude: {response}")
            
            # Attempt to parse common duration patterns
            user_input_lower = user_input.lower()
            total_days = 0
            
            # Check for months
            month_match = re.search(r'(\d+)\s*(?:month|mo)s?', user_input_lower)
            if month_match:
                try:
                    months = int(month_match.group(1))
                    total_days += months * 30  # Approximate
                except ValueError:
                    pass
                    
            # Check for weeks
            week_match = re.search(r'(\d+)\s*(?:week|wk)s?', user_input_lower)
            if week_match:
                try:
                    weeks = int(week_match.group(1))
                    total_days += weeks * 7
                except ValueError:
                    pass
                    
            # Check for days
            day_match = re.search(r'(\d+)\s*(?:day|d)s?', user_input_lower)
            if day_match:
                try:
                    days = int(day_match.group(1))
                    total_days += days
                except ValueError:
                    pass
                    
            # Check for common phrases
            if 'couple' in user_input_lower or 'few' in user_input_lower:
                if 'day' in user_input_lower:
                    total_days += 2 if 'couple' in user_input_lower else 3
                elif 'week' in user_input_lower:
                    total_days += 14 if 'couple' in user_input_lower else 21
                elif 'month' in user_input_lower:
                    total_days += 60 if 'couple' in user_input_lower else 90
                    
            return total_days
    
    def interpret_state(self, user_input: str) -> Optional[str]:
        """
        Extract the U.S. state name from natural language.
        
        Example inputs:
        - "Born in Texas"
        - "We were living in NY at the time"
        - "At a hospital in Pennsylvania"
        """
        if not user_input:
            return None
            
        prompt = """
        Extract the U.S. state mentioned in this text.
        Respond with ONLY the full state name with proper capitalization.
        Convert state abbreviations to full names (e.g., "NY" → "New York").
        
        If you can't determine a specific state, respond with "unknown".
        """
        
        response = self._query_claude(prompt, user_input)
        
        # Clean up the response
        state = response.strip()
        
        # Return None if Claude couldn't determine the state
        if state.lower() == "unknown":
            # Try to match state with regex patterns
            return self._legacy_state_parsing(user_input)
            
        return state
    
    def _legacy_state_parsing(self, text: str) -> Optional[str]:
        """Fallback method to extract state using regex patterns."""
        # Dictionary mapping state abbreviations to full names
        state_abbrev = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
            'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
            'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
            'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
            'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
            'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
            'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
            'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
            'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
            'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
            'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
        }
        
        # List of all state names for regex matching
        state_names = list(state_abbrev.values())
        
        # First, check for state abbreviations (must be uppercase, with word boundaries)
        for abbrev, full_name in state_abbrev.items():
            pattern = r'\b' + abbrev + r'\b'
            if re.search(pattern, text):
                return full_name
                
        # Then check for full state names (case insensitive)
        for state in state_names:
            pattern = r'\b' + re.escape(state) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                return state
                
        # Check for close matches (for typos and misspellings)
        text_lower = text.lower()
        for state in state_names:
            # Check if state name appears with minor differences
            if state.lower() in text_lower or text_lower in state.lower():
                return state
                
        # Fuzzy matching for typos
        best_match = None
        best_score = 0.7  # Threshold for similarity
        
        for state in state_names:
            score = self._similar_text(state.lower(), text_lower)
            if score > best_score:
                best_score = score
                best_match = state
                
        return best_match
        
    def _similar_text(self, a: str, b: str) -> float:
        """Calculate similarity between two strings."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()
    
    def _query_claude(self, system_prompt: str, user_input: str) -> str:
        """
        Helper method to query Claude with a system prompt and user input.
        Returns the text content of Claude's response.
        Includes retry logic for transient errors.
        """
        # Limit input length to prevent token issues
        if len(user_input) > MAX_RESPONSE_LENGTH:
            logger.warning(f"Truncating oversized input of {len(user_input)} chars")
            user_input = user_input[:MAX_RESPONSE_LENGTH] + "..."
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.messages_create(
                    model=self.model,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=150,
                    temperature=0.1,  # Low temperature for more deterministic responses
                    timeout=DEFAULT_TIMEOUT
                )
                
                # Extract and return the response content
                return response.content[0].text.strip()
            except Exception as e:
                # Handle all exceptions more generally
                wait_time = (attempt + 1) * RETRY_DELAY
                logger.warning(f"API error: {e}. Waiting {wait_time}s before retry {attempt+1}/{MAX_RETRIES}")
                time.sleep(wait_time)
                    
        logger.error(f"Failed to get response from Claude after {MAX_RETRIES} attempts")
        return ""
    
    def _text_to_num(self, text: str) -> Optional[int]:
        """
        Convert textual number to integer.
        """
        text_map = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 
            'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 
            'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 
            'eighty': 80, 'ninety': 90
        }
        
        return text_map.get(text.lower(), None)

class AgeParser:
    """
    Enhanced age parsing and validation for CP case eligibility
    """
    
    # Expanded patterns for age detection
    AGE_PATTERNS = [
        # Standard formats
        r'(\d+(?:\.\d+)?)\s*(?:year|yr|y)s?\s*old',
        r'(\d+(?:\.\d+)?)\s*(?:year|yr|y)s?',
        # Just the number
        r'(?:is|turned|age)\s*(\d+(?:\.\d+)?)',
        r'^(\d+(?:\.\d+)?)$',
        # Textual numbers (limited to common young ages)
        r'(?:is|turned|age)\s*(one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:years?\s*old)?',
        # Months format
        r'(\d+)\s*months?\s*old',
        # Age with fractions
        r'(\d+)\s*(?:and a half|and 1/2)',
        r'(\d+)\s*(?:and three quarters|and 3/4)',
        r'(\d+)\s*(?:and a quarter|and 1/4)',
        # Special cases for young children
        r'almost\s*(\d+)',
        r'just turned\s*(\d+)',
        r'about to turn\s*(\d+)',
    ]
    
    # Mapping for text numbers
    TEXT_TO_NUM = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    # Fraction modifiers
    FRACTION_MODIFIERS = {
        'and a half': 0.5,
        'and 1/2': 0.5,
        'and a quarter': 0.25,
        'and 1/4': 0.25,
        'and three quarters': 0.75,
        'and 3/4': 0.75
    }
    
    @classmethod
    def parse_age(cls, age_input: str) -> Optional[float]:
        """
        Parse age from various input formats.
        Returns age as float or None if unable to parse.
        """
        if not age_input:
            return None
            
        age_input = age_input.lower().strip()
        
        # Check for months format first
        months_match = re.search(r'(\d+)\s*months?\s*old', age_input)
        if months_match:
            try:
                months = float(months_match.group(1))
                # Convert months to years (rounded to 1 decimal place)
                return round(months / 12.0, 1)
            except ValueError:
                pass
        
        # Check for fractional expressions
        for pattern, modifier in cls.FRACTION_MODIFIERS.items():
            if pattern in age_input:
                base_match = re.search(r'(\d+)\s*' + re.escape(pattern), age_input)
                if base_match:
                    try:
                        base = float(base_match.group(1))
                        return base + modifier
                    except ValueError:
                        continue
        
        # Try each regular pattern
        for pattern in cls.AGE_PATTERNS:
            match = re.search(pattern, age_input)
            if match:
                age_str = match.group(1)
                
                # Handle textual numbers
                if age_str in cls.TEXT_TO_NUM:
                    return float(cls.TEXT_TO_NUM[age_str])
                
                # Handle special prefixes
                if pattern.startswith(r'almost\s*'):
                    try:
                        return float(age_str) - 0.1
                    except ValueError:
                        continue
                
                if pattern.startswith(r'about to turn\s*'):
                    try:
                        return float(age_str) - 0.1
                    except ValueError:
                        continue
                
                try:
                    age = float(age_str)
                    # Validate reasonable age range
                    if 0 <= age <= 25:
                        return age
                except ValueError:
                    continue
                    
        return None

    @classmethod
    def is_age_valid(cls, age: float) -> bool:
        """Validate if age is within reasonable range for CP cases."""
        return 0 <= age <= 25
        
    @classmethod
    def normalize_age(cls, age: Optional[float]) -> Optional[float]:
        """Normalize age value to handle edge cases."""
        if age is None:
            return None
            
        # Round to 1 decimal place for consistency
        age = round(age, 1)
        
        # Ensure within valid range
        if age < 0:
            return 0.0
        if age > 25:
            return 25.0
            
        return age

class SOLParser:
    """
    Enhanced Statute of Limitations parsing and comparison
    """
    
    @classmethod
    def parse_sol_age(cls, sol_string: str) -> Optional[float]:
        """
        Convert SOL string to numerical age limit.
        Handles both "Nth birthday" and "N years" formats.
        """
        if not sol_string:
            return None
            
        # Handle "Nth birthday" format
        birthday_match = re.search(r'(\d+)(?:st|nd|rd|th)\s*birthday', sol_string)
        if birthday_match:
            return float(birthday_match.group(1))
            
        # Handle "N years" format
        years_match = re.search(r'(\d+)\s*years?', sol_string)
        if years_match:
            return float(years_match.group(1))
            
        # Handle just number
        num_match = re.search(r'(\d+)', sol_string)
        if num_match:
            return float(num_match.group(1))
            
        return None

    @classmethod
    def is_within_sol(cls, current_age: float, sol_string: str) -> bool:
        """
        Check if current age is within the SOL limit.
        For "Nth birthday" format, must be under N years old.
        For "N years" format, must be under N years old.
        """
        sol_age = cls.parse_sol_age(sol_string)
        if sol_age is None:
            return False
            
        # Always check if current age is less than SOL age
        return current_age < sol_age

class ModelConfiguration:
    """
    Manages the configuration and validation of the Claude model version.
    """
    def __init__(self):
        load_dotenv()
        # Use a more recent default model if available
        self.model_version = os.getenv('MODEL_VERSION', 'claude-3-5-sonnet-20241022')
        
        if not self._validate_model_version(self.model_version):
            logger.warning(f"Invalid model version format: {self.model_version}")
            logger.warning("Falling back to default version: claude-3-5-sonnet-20241022")
            self.model_version = 'claude-3-5-sonnet-20241022'
    
    def _validate_model_version(self, version: str) -> bool:
        return bool(re.match(r'claude-\d+-\d?-?[a-zA-Z]+-\d{8}', version))

class ConversationManager:
    """
    Manages the conversation flow and state with added ranking system and Claude NLU.
    """
    def __init__(self, legal_rules: Dict, claude_client=None, model_version=None):
        self.legal_rules = legal_rules
        self.current_phase = 'age'  # Start directly at age phase instead of initial
        self.empty_response_count = 0
        self.case_data = {
            'age': None,
            'state': None,
            'weeks_pregnant': 0,  # Initialize to 0 instead of None
            'difficult_delivery': False,  # Track delivery difficulty
            'points': 50,  # Starting with a base score of 50
            'ranking': 'normal'  # Default ranking (low, normal, high, very high)
        }
        
        # Add this new field to track implied answers to future questions
        self.implied_answers = {
            'nicu': None,
            'nicu_duration': None,
            'hie_therapy': None,
            'brain_scan': None,
            'milestones': None,
            'lawyer': None,
            'state': None
        }
        
        # Initialize Claude NLU if client is provided
        self.nlu = None
        if claude_client and model_version:
            try:
                self.nlu = ClaudeNLU(claude_client, model_version)
                logger.info("Claude NLU initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Claude NLU: {e}")
        
        self.phases = {
            'age': {
                'complete': False,
                'question': "How old is your child with CP?",
                'value': None
            },
            'pregnancy': {
                'complete': False,
                'question': "How many weeks pregnant were you when your child was born? Did your child have a difficult delivery?",
                'value': None
            },
            'nicu': {
                'complete': False,
                'question': "Did your child go to the NICU after birth?",
                'value': None
            },
            'nicu_duration': {
                'complete': False,
                'question': "How long was your child in the NICU for after birth?",
                'value': None
            },
            'hie_therapy': {  # New phase for HIE therapy question
                'complete': False,
                'question': "Did your child receive head cooling or HIE therapy while in the NICU?",
                'value': None
            },
            'brain_scan': {
                'complete': False,
                'question': "Did your child receive an MRI or Brain Scan while in the NICU?",
                'value': None
            },
            'milestones': {
                'complete': False,
                'question': "Is your child missing any milestones and or having any delays?",
                'value': None
            },
            'lawyer': {
                'complete': False,
                'question': "This sounds like it definitely needs to be looked into further. Have you had your case reviewed by a lawyer yet?",
                'value': None
            },
            'state': {
                'complete': False,
                'question': "In what State was your child born?",
                'value': None
            }
        }
        
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIRECTORY, exist_ok=True)

    def update_points(self, points_change: int, reason: str) -> None:
        """
        Updates the case points based on various factors.
        Points system determines overall case ranking.
        """
        # Ensure points_change is an integer
        try:
            points_change = int(points_change)
        except (ValueError, TypeError):
            logger.warning(f"Invalid points change value: {points_change}. Using 0.")
            points_change = 0
            
        self.case_data['points'] += points_change
        logger.info(f"Points {'+' if points_change >= 0 else ''}{points_change} ({reason}). New total: {self.case_data['points']}")
        
        # Ensure points don't go negative
        if self.case_data['points'] < 0:
            self.case_data['points'] = 0
            logger.info("Points adjusted to minimum of 0")
        
        # Update ranking based on point thresholds
        if self.case_data['points'] >= 80:
            new_rank = 'very high'
        elif self.case_data['points'] >= 65:
            new_rank = 'high'
        elif self.case_data['points'] >= 40:
            new_rank = 'normal'
        else:
            new_rank = 'low'
            
        # Update ranking if it changed
        if new_rank != self.case_data['ranking']:
            self.case_data['ranking'] = new_rank
            logger.info(f"Case ranking updated to: {new_rank}")

    def save_case_data(self) -> Tuple[bool, Optional[str]]:
        """
        Saves the case data and rankings to a JSON file.
        Returns a tuple of (success, error_message)
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prepare data to save with defaults for missing values
            data = {
                'timestamp': timestamp,
                'ranking': self.case_data.get('ranking', 'normal'),
                'points': self.case_data.get('points', 50),
                'age': self.case_data.get('age'),
                'state': self.case_data.get('state'),
                'weeks_pregnant': self.case_data.get('weeks_pregnant', 0),
                'difficult_delivery': self.case_data.get('difficult_delivery', False),
            }
            
            # Add all phase values
            for phase_name, phase_data in self.phases.items():
                if phase_name != 'initial' and phase_data.get('value') is not None:
                    if isinstance(phase_data['value'], bool):
                        data[phase_name] = 'yes' if phase_data['value'] else 'no'
                    else:
                        data[phase_name] = phase_data['value']
            
            # Generate a unique filename with timestamp
            filename = os.path.join(DATA_DIRECTORY, f"case_{timestamp}.json")
            
            # Save as JSON
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            # Also append to aggregate file for all cases
            aggregate_file = os.path.join(DATA_DIRECTORY, "all_cases.json")
            
            if os.path.exists(aggregate_file):
                try:
                    with open(aggregate_file, 'r') as f:
                        existing_data = json.load(f)
                        
                    # Ensure existing data is a list
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
                        
                    existing_data.append(data)
                        
                    with open(aggregate_file, 'w') as f:
                        json.dump(existing_data, f, indent=2)
                except Exception as e:
                    logger.error(f"Error updating aggregate file: {e}")
                    # Fall back to writing a new file if reading fails
                    with open(aggregate_file, 'w') as f:
                        json.dump([data], f, indent=2)
            else:
                with open(aggregate_file, 'w') as f:
                    json.dump([data], f, indent=2)
                    
            logger.info(f"Case data saved to {filename} and {aggregate_file}")
            return True, None
            
        except PermissionError:
            error_msg = f"Permission denied when saving case data"
            logger.error(error_msg)
            return False, error_msg
        except IOError as e:
            error_msg = f"I/O error when saving case data: {e}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error saving case data: {e}"
            logger.error(error_msg)
            return False, error_msg

    def analyze_age_response(self, message: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Analyze age response and return parsed age and any error message.
        Now with Claude NLU capabilities for complex responses.
        """
        if not message or not message.strip():
            return None, "Please provide your child's age."
            
        # First try to use Claude NLU if available
        if self.nlu:
            try:
                parsed_age = self.nlu.interpret_age(message)
                if parsed_age is not None:
                    parsed_age = AgeParser.normalize_age(parsed_age)  # Normalize
                    if parsed_age is not None and AgeParser.is_age_valid(parsed_age):
                        logger.info(f"Claude NLU parsed age: {parsed_age}")
                        return parsed_age, None
            except Exception as e:
                logger.error(f"Claude NLU age parsing error: {e}. Falling back to regex parsing.")
        
        # Fallback to traditional parsing if NLU fails or is unavailable
        parsed_age = AgeParser.parse_age(message)
        parsed_age = AgeParser.normalize_age(parsed_age)  # Normalize
        
        if parsed_age is None:
            return None, "I couldn't understand the age provided. Please provide the age in years, like '5' or '5 years old'."
        
        if not AgeParser.is_age_valid(parsed_age):
            return None, "Please provide a valid age between 0 and 25 years."
        
        return parsed_age, None

    def check_age_eligibility(self, age: float, state: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Check if age meets eligibility criteria for given state.
        """
        if age is None:
            return False, "Unable to determine age eligibility without age information."
            
        # Basic age check (maximum age across all states)
        if age >= 21:
            return False, "We apologize, but based on your child's age, we cannot proceed with your case."
        
        # State-specific check if state is provided
        if state and state in self.legal_rules.get('stateSOL', {}):
            state_sol = self.legal_rules['stateSOL'][state].get('minorSOL')
            if state_sol and not SOLParser.is_within_sol(age, state_sol):
                return False, f"We apologize, but based on your child's age and {state}'s requirements, we cannot proceed with your case."
        
        return True, None

    def check_eligibility(self) -> Tuple[bool, Optional[str]]:
        """
        Evaluates current case information against eligibility criteria.
        Only state/SOL issues should disqualify a case completely.
        """
        # Check both age and state if available
        if self.case_data.get('age') is not None and self.case_data.get('state') is not None:
            # First check excluded states list - this is a true disqualifier
            excluded_states = self.legal_rules.get('globalExclusions', {}).get('excludedStates', {}).get('list', [])
            if self.case_data['state'] in excluded_states:
                return False, f"We apologize, but we are currently not accepting cases from {self.case_data['state']}."
            
            # Check age eligibility with state context - SOL is a true disqualifier
            is_eligible, reason = self.check_age_eligibility(self.case_data['age'], self.case_data['state'])
            if not is_eligible:
                return False, reason
        
        # Check age alone if that's all we have - SOL is a true disqualifier
        elif self.case_data.get('age') is not None:
            is_eligible, reason = self.check_age_eligibility(self.case_data['age'])
            if not is_eligible:
                return False, reason
        
        return True, None

    def _analyze_for_implied_answers(self, message: str) -> None:
        """
        Analyzes a message for information that implies answers to other questions.
        Updates the implied_answers dictionary with any found information.
        """
        if not message:
            return
            
        message_lower = message.lower()
        
        # Check for NICU mentions
        nicu_indicators = ['nicu', 'intensive care', 'incubator', 'special care']
        nicu_negative = ['didn\'t go', 'did not go', 'no nicu', 'wasn\'t in', 'never went']
        
        if any(indicator in message_lower for indicator in nicu_indicators):
            # Check if it's a negative mention
            if any(neg in message_lower for neg in nicu_negative):
                self.implied_answers['nicu'] = False
            else:
                self.implied_answers['nicu'] = True
                
                # If NICU is mentioned, check for duration
                try:
                    if self.nlu:
                        duration_days = self.nlu.interpret_duration(message)
                        if duration_days > 0:
                            self.implied_answers['nicu_duration'] = duration_days
                    else:
                        duration_days = self._legacy_duration_parsing(message)
                        if duration_days > 0:
                            self.implied_answers['nicu_duration'] = duration_days
                except Exception as e:
                    logger.error(f"Error parsing implied NICU duration: {e}")
        
        # Check for HIE/cooling therapy mentions
        cooling_indicators = ['cooling', 'hypothermia', 'hie therapy', 'head cool', 'cooling blanket', 'therapeutic hypothermia']
        if any(indicator in message_lower for indicator in cooling_indicators):
            self.implied_answers['hie_therapy'] = True
        
        # Check for brain scan/MRI mentions
        scan_indicators = ['mri', 'brain scan', 'head scan', 'cat scan', 'ct scan', 'ultrasound', 'sonogram', 'imaging']
        if any(indicator in message_lower for indicator in scan_indicators):
            self.implied_answers['brain_scan'] = True
        
        # Check for developmental delay mentions
        delay_indicators = ['delay', 'behind', 'missing milestone', 'developmental', 'not meeting', 'therapy', 'pt', 'ot', 'speech', 'physical therapy']
        if any(indicator in message_lower for indicator in delay_indicators):
            self.implied_answers['milestones'] = True
        
        # Check for lawyer mentions
        lawyer_indicators = ['lawyer', 'attorney', 'legal', 'law firm', 'lawsuit', 'case review', 'litigation']
        lawyer_negative = ['no lawyer', 'haven\'t seen', 'didn\'t consult', 'not yet', 'looking for']
        if any(indicator in message_lower for indicator in lawyer_indicators):
            # Check if it's a negative mention
            if any(neg in message_lower for neg in lawyer_negative):
                self.implied_answers['lawyer'] = False
            else:
                self.implied_answers['lawyer'] = True
        
        # Try to extract state information
        state_indicators = [
            'born in', 'state of', 'from', 'in'
        ]
        for indicator in state_indicators:
            if indicator in message_lower:
                # Try to extract state after indicator
                parts = message_lower.split(indicator)
                if len(parts) > 1:
                    potential_state = parts[1].strip()
                    # Use NLU to interpret state if available
                    if self.nlu:
                        try:
                            state = self.nlu.interpret_state(potential_state)
                            if state:
                                self.implied_answers['state'] = state
                                break
                        except Exception as e:
                            logger.error(f"Error parsing implied state: {e}")

    def analyze_response(self, message: str) -> Dict:
        """
        Processes user response and updates conversation state.
        Now with Claude NLU for better natural language understanding.
        """
        # Sanitize input
        if not message:
            message = ""
        message = str(message).strip()
        
        if not message:
            self.empty_response_count += 1
            return {}
            
        self.empty_response_count = 0
        response_data = {}
        
        # Check for back commands or special navigational requests
        back_indicators = ['back', 'previous', 'go back', 'return']
        if any(indicator in message.lower() for indicator in back_indicators):
            return self._handle_back_request()
        
        # Check for help/confusion commands
        help_indicators = ['help', 'confused', "don't understand", "what's this", "explain"]
        if message.lower() in help_indicators or any(indicator in message.lower() for indicator in help_indicators):
            return self._handle_help_request()
        
        # Before processing based on current phase, analyze response for implied answers
        self._analyze_for_implied_answers(message)
        
        # Process based on current phase
        try:
                    
            elif self.current_phase == 'age':
                # Age parsing uses the analyze_age_response method which already incorporates NLU
                age, error_message = self.analyze_age_response(message)
                
                if error_message:
                    return {'error': error_message}
                    
                self.phases['age']['value'] = age
                self.phases['age']['complete'] = True
                self.case_data['age'] = age
                
                # Check eligibility based on age
                is_eligible, reason = self.check_eligibility()
                if not is_eligible:
                    return {'eligible': False, 'reason': reason}
                    
                self.current_phase = 'pregnancy'
                response_data['age'] = age
                    
            elif self.current_phase == 'pregnancy':
                self.phases['pregnancy']['value'] = message
                self.phases['pregnancy']['complete'] = True
                
                # Use Claude NLU to extract pregnancy details if available
                if self.nlu:
                    try:
                        pregnancy_details = self.nlu.interpret_pregnancy_details(message)
                        logger.info(f"Claude NLU pregnancy details: {pregnancy_details}")
                        
                        # Update weeks pregnant if available
                        if pregnancy_details.get('weeks') is not None:
                            weeks = pregnancy_details['weeks']
                            self.case_data['weeks_pregnant'] = weeks
                            
                            # Update points based on gestational age
                            if weeks < 30:
                                self.update_points(15, "Very premature birth (< 30 weeks)")
                            elif weeks < 36:
                                self.update_points(10, "Premature birth (< 36 weeks)")
                            else:
                                self.update_points(-5, "Full term birth (≥ 36 weeks)")
                        
                        # Update difficult delivery flag
                        difficult_delivery = pregnancy_details.get('difficult_delivery', False)
                        self.case_data['difficult_delivery'] = difficult_delivery
                        
                        # Set sympathetic message for difficult delivery
                        if difficult_delivery:
                            self.update_points(15, "Difficult delivery reported")
                            response_data['sympathy_message'] = "I'm sorry to hear that your delivery was difficult."
                        else:
                            self.update_points(-10, "No difficult delivery reported")
                            
                    except Exception as e:
                        logger.error(f"Claude NLU pregnancy parsing error: {e}. Falling back to regex parsing.")
                        # Fallback to regex parsing
                        self._legacy_pregnancy_parsing(message)
                else:
                    # No NLU available, use regex parsing
                    self._legacy_pregnancy_parsing(message)
                
                # Check if NICU was mentioned in pregnancy response
                if self.implied_answers['nicu'] is not None:
                    logger.info(f"Skipping NICU question as it was answered in pregnancy response: {self.implied_answers['nicu']}")
                    self.phases['nicu']['value'] = self.implied_answers['nicu']
                    self.phases['nicu']['complete'] = True
                    
                    if not self.implied_answers['nicu']:
                        # Lower points if no NICU stay
                        self.update_points(-15, "No NICU stay (implied)")
                        # Skip to appropriate next phase
                        if self.case_data.get('weeks_pregnant') is not None and self.case_data.get('weeks_pregnant') >= 36:
                            self.current_phase = 'hie_therapy'
                        else:
                            self.current_phase = 'milestones'
                    else:
                        # Increase points for NICU stay
                        self.update_points(10, "NICU stay required (implied)")
                        
                        # Check if NICU duration was also mentioned
                        if self.implied_answers['nicu_duration'] is not None:
                            self.phases['nicu_duration']['value'] = self.implied_answers['nicu_duration']
                            self.phases['nicu_duration']['complete'] = True
                            
                            # Award points based on implied NICU duration
                            duration_days = self.implied_answers['nicu_duration']
                            if isinstance(duration_days, (int, float)) and duration_days > 0:
                                if duration_days > 30:
                                    self.update_points(15, "Extended NICU stay (>30 days) (implied)")
                                elif duration_days > 14:
                                    self.update_points(10, "Moderate NICU stay (>14 days) (implied)")
                                elif duration_days > 7:
                                    self.update_points(5, "Short NICU stay (>7 days) (implied)")
                                else:
                                    self.update_points(3, "Brief NICU stay (implied)")
                            
                            # Skip directly to appropriate next phase
                            if self.case_data.get('weeks_pregnant') is not None and self.case_data.get('weeks_pregnant') >= 36:
                                self.current_phase = 'hie_therapy'
                            else:
                                self.current_phase = 'brain_scan'
                        else:
                            self.current_phase = 'nicu_duration'
                    
                else:
                    self.current_phase = 'nicu'
                    
            elif self.current_phase == 'nicu':
                # Use Claude NLU to determine if NICU stay if available
                is_nicu = False
                
                if self.nlu:
                    try:
                        is_nicu = self.nlu.interpret_yes_no(message, "Did the child go to NICU after birth")
                        logger.info(f"Claude NLU NICU interpretation: {'yes' if is_nicu else 'no'}")
                    except Exception as e:
                        logger.error(f"Claude NLU NICU parsing error: {e}. Falling back to pattern matching.")
                        is_nicu = self._is_affirmative(message)
                else:
                    is_nicu = self._is_affirmative(message)
                    
                self.phases['nicu']['value'] = is_nicu
                self.phases['nicu']['complete'] = True
                
                if not is_nicu:
                    # Lower points if no NICU stay
                    self.update_points(-15, "No NICU stay")
                    
                    # Important: Check if full term (≥36 weeks) - we still need to ask HIE for full-term
                    if self.case_data.get('weeks_pregnant') is not None and self.case_data.get('weeks_pregnant') >= 36:
                        self.current_phase = 'hie_therapy'
                    else:
                        # Skip the brain scan question since they didn't go to NICU
                        self.current_phase = 'milestones'
                else:
                    # Increase points for NICU stay
                    self.update_points(10, "NICU stay required")
                    self.current_phase = 'nicu_duration'
                    
            elif self.current_phase == 'nicu_duration':
                self.phases['nicu_duration']['value'] = message
                self.phases['nicu_duration']['complete'] = True
                
                # Use Claude NLU to extract duration if available
                duration_days = 0
                
                if self.nlu:
                    try:
                        duration_days = self.nlu.interpret_duration(message)
                        logger.info(f"Claude NLU NICU duration: {duration_days} days")
                    except Exception as e:
                        logger.error(f"Claude NLU duration parsing error: {e}. Falling back to regex parsing.")
                        # Fallback to regex parsing
                        duration_days = self._legacy_duration_parsing(message)
                else:
                    # No NLU available, use regex parsing
                    duration_days = self._legacy_duration_parsing(message)
                    
                # Award points based on NICU duration
                if duration_days > 30:
                    self.update_points(15, "Extended NICU stay (>30 days)")
                elif duration_days > 14:
                    self.update_points(10, "Moderate NICU stay (>14 days)")
                elif duration_days > 7:
                    self.update_points(5, "Short NICU stay (>7 days)")
                elif duration_days > 0:
                    self.update_points(3, "Brief NICU stay")
                
                # Check if HIE therapy was mentioned in NICU duration response
                if self.implied_answers['hie_therapy'] is not None:
                    logger.info(f"Skipping HIE therapy question as it was answered in NICU duration response: {self.implied_answers['hie_therapy']}")
                    self.phases['hie_therapy']['value'] = self.implied_answers['hie_therapy']
                    self.phases['hie_therapy']['complete'] = True
                    
                    if self.implied_answers['hie_therapy']:
                        # Very high points for HIE therapy (strongest indicator)
                        self.update_points(40, "Received HIE/head cooling therapy (implied)")
                    
                    self.current_phase = 'brain_scan'
                # Check if brain scan was mentioned in NICU duration response
                elif self.implied_answers['brain_scan'] is not None:
                    logger.info(f"Skipping brain scan question as it was answered in NICU duration response: {self.implied_answers['brain_scan']}")
                    self.phases['brain_scan']['value'] = self.implied_answers['brain_scan']
                    self.phases['brain_scan']['complete'] = True
                    
                    if self.implied_answers['brain_scan']:
                        # Higher points for brain scan evidence
                        self.update_points(20, "Brain scan/MRI was performed (implied)")
                    else:
                        # Lower points if no scan
                        self.update_points(-10, "No brain scan/MRI performed (implied)")
                    
                    self.current_phase = 'milestones'
                # Always ask HIE for full term babies regardless of NICU stay
                elif self.case_data.get('weeks_pregnant') is not None and self.case_data.get('weeks_pregnant') >= 36:
                    self.current_phase = 'hie_therapy'
                else:
                    self.current_phase = 'brain_scan'
                    
            elif self.current_phase == 'hie_therapy':
                # Use Claude NLU to determine if received HIE therapy if available
                received_hie = False
                
                if self.nlu:
                    try:
                        received_hie = self.nlu.interpret_yes_no(message, "Did the child receive head cooling or HIE therapy")
                        logger.info(f"Claude NLU HIE therapy interpretation: {'yes' if received_hie else 'no'}")
                    except Exception as e:
                        logger.error(f"Claude NLU HIE therapy parsing error: {e}. Falling back to pattern matching.")
                        received_hie = self._is_affirmative(message)
                else:
                    received_hie = self._is_affirmative(message)
                    
                self.phases['hie_therapy']['value'] = received_hie
                self.phases['hie_therapy']['complete'] = True
                
                if received_hie:
                    # Very high points for HIE therapy (strongest indicator)
                    self.update_points(40, "Received HIE/head cooling therapy")
                
                # Check if brain scan was mentioned in HIE therapy response
                if self.implied_answers['brain_scan'] is not None:
                    logger.info(f"Skipping brain scan question as it was answered in HIE therapy response: {self.implied_answers['brain_scan']}")
                    self.phases['brain_scan']['value'] = self.implied_answers['brain_scan']
                    self.phases['brain_scan']['complete'] = True
                    
                    if self.implied_answers['brain_scan']:
                        # Higher points for brain scan evidence
                        self.update_points(20, "Brain scan/MRI was performed (implied)")
                    else:
                        # Lower points if no scan
                        self.update_points(-10, "No brain scan/MRI performed (implied)")
                    
                    self.current_phase = 'milestones'
                else:
                    self.current_phase = 'brain_scan'
                    
            elif self.current_phase == 'brain_scan':
                # Use Claude NLU to determine if received brain scan if available
                received_scan = False
                
                if self.nlu:
                    try:
                        received_scan = self.nlu.interpret_yes_no(message, "Did the child receive an MRI or brain scan while in the NICU")
                        logger.info(f"Claude NLU brain scan interpretation: {'yes' if received_scan else 'no'}")
                    except Exception as e:
                        logger.error(f"Claude NLU brain scan parsing error: {e}. Falling back to pattern matching.")
                        received_scan = self._is_affirmative(message)
                else:
                    received_scan = self._is_affirmative(message)
                    
                self.phases['brain_scan']['value'] = received_scan
                self.phases['brain_scan']['complete'] = True
                
                if received_scan:
                    # Higher points for brain scan evidence
                    self.update_points(20, "Brain scan/MRI was performed")
                else:
                    # Lower points if no scan
                    self.update_points(-10, "No brain scan/MRI performed")
                    
                # Check if developmental delays were mentioned in brain scan response
                if self.implied_answers['milestones'] is not None:
                    logger.info(f"Skipping milestones question as it was answered in brain scan response: {self.implied_answers['milestones']}")
                    self.phases['milestones']['value'] = self.implied_answers['milestones']
                    self.phases['milestones']['complete'] = True
                    
                    if self.implied_answers['milestones']:
                        # Higher points for developmental issues
                        self.update_points(15, "Developmental delays reported (implied)")
                    else:
                        # Lower points if no issues
                        self.update_points(-5, "No developmental delays reported (implied)")
                    
                    self.current_phase = 'lawyer'
                else:
                    self.current_phase = 'milestones'
                    
            elif self.current_phase == 'milestones':
                # Use Claude NLU to determine if has developmental delays if available
                has_delays = False
                
                if self.nlu:
                    try:
                        has_delays = self.nlu.interpret_yes_no(message, "Is the child missing developmental milestones or has delays")
                        logger.info(f"Claude NLU milestones interpretation: {'yes' if has_delays else 'no'}")
                    except Exception as e:
                        logger.error(f"Claude NLU milestones parsing error: {e}. Falling back to pattern matching.")
                        has_delays = self._is_affirmative(message) or any(positive in message.lower() for positive in ['delay', 'behind', 'missing', 'not meeting'])
                else:
                    has_delays = self._is_affirmative(message) or any(positive in message.lower() for positive in ['delay', 'behind', 'missing', 'not meeting'])
                    
                self.phases['milestones']['value'] = message
                self.phases['milestones']['complete'] = True
                
                if has_delays:
                    # Higher points for developmental issues
                    self.update_points(15, "Developmental delays reported")
                else:
                    # Lower points if no issues
                    self.update_points(-5, "No developmental delays reported")
                    
                # Check if lawyer consultation was mentioned in milestones response
                if self.implied_answers['lawyer'] is not None:
                    logger.info(f"Skipping lawyer question as it was answered in milestones response: {self.implied_answers['lawyer']}")
                    self.phases['lawyer']['value'] = self.implied_answers['lawyer']
                    self.phases['lawyer']['complete'] = True
                    
                    if self.implied_answers['lawyer']:
                        # Adjust points slightly down for previous consultation
                        self.update_points(-5, "Previous legal consultation (implied)")
                        # Set a flag to end the chat with farewell message
                        response_data['end_chat'] = True
                        response_data['farewell_message'] = "We're glad to hear you're already getting your case reviewed and getting the help you need. We wish you and your family the best."
                        
                        # Save case data before ending
                        success, error = self.save_case_data()
                        if not success:
                            logger.warning(f"Could not save case data: {error}")
                            
                        # Return immediately to prevent going to the next phase
                        return response_data
                    else:
                        # Adjust points slightly up for new case
                        self.update_points(5, "No previous legal consultation (implied)")
                        
                        # Check if state was mentioned in lawyer response
                        if self.implied_answers['state'] is not None:
                            logger.info(f"Using implied state from previous response: {self.implied_answers['state']}")
                            self.phases['state']['value'] = self.implied_answers['state']
                            self.phases['state']['complete'] = True
                            self.case_data['state'] = self.implied_answers['state']
                            
                            # Check eligibility based on state and age
                            is_eligible, reason = self.check_eligibility()
                            if not is_eligible:
                                return {'eligible': False, 'reason': reason}
                                
                            self.current_phase = 'complete'
                            
                            # Save answers and ranking to file
                            success, error = self.save_case_data()
                            if not success:
                                logger.warning(f"Could not save case data: {error}")
                        else:
                            self.current_phase = 'state'
                else:
                    self.current_phase = 'lawyer'
                    
            elif self.current_phase == 'lawyer':
                # Use Claude NLU to determine if previous legal consultation if available
                prev_consultation = False
                
                if self.nlu:
                    try:
                        prev_consultation = self.nlu.interpret_yes_no(message, "Has the family previously consulted a lawyer about this case")
                        logger.info(f"Claude NLU lawyer consultation interpretation: {'yes' if prev_consultation else 'no'}")
                    except Exception as e:
                        logger.error(f"Claude NLU lawyer parsing error: {e}. Falling back to pattern matching.")
                        prev_consultation = self._is_affirmative(message)
                else:
                    prev_consultation = self._is_affirmative(message)
                    
                self.phases['lawyer']['value'] = prev_consultation  # Store boolean value 
                self.phases['lawyer']['complete'] = True
                
                if prev_consultation:
                    # Adjust points slightly down for previous consultation
                    self.update_points(-5, "Previous legal consultation")
                    # Set a flag to end the chat with farewell message
                    response_data['end_chat'] = True
                    response_data['farewell_message'] = "We're glad to hear you're already getting your case reviewed and getting the help you need. We wish you and your family the best."
                    
                    # Save case data before ending
                    success, error = self.save_case_data()
                    if not success:
                        logger.warning(f"Could not save case data: {error}")
                        
                    # Return immediately to prevent going to the next phase
                    return response_data
                else:
                    # Adjust points slightly up for new case
                    self.update_points(5, "No previous legal consultation")
                    
                    # Check if state was mentioned in lawyer response
                    if self.implied_answers['state'] is not None:
                        logger.info(f"Using implied state from previous response: {self.implied_answers['state']}")
                        self.phases['state']['value'] = self.implied_answers['state']
                        self.phases['state']['complete'] = True
                        self.case_data['state'] = self.implied_answers['state']
                        
                        # Check eligibility based on state and age
                        is_eligible, reason = self.check_eligibility()
                        if not is_eligible:
                            return {'eligible': False, 'reason': reason}
                            
                        self.current_phase = 'complete'
                        
                        # Save answers and ranking to file
                        success, error = self.save_case_data()
                        if not success:
                            logger.warning(f"Could not save case data: {error}")
                    else:
                        self.current_phase = 'state'
                    
            elif self.current_phase == 'state':
                # Use Claude NLU to extract state if available
                state = None
                
                if self.nlu:
                    try:
                        state = self.nlu.interpret_state(message)
                        logger.info(f"Claude NLU state interpretation: {state}")
                    except Exception as e:
                        logger.error(f"Claude NLU state parsing error: {e}. Falling back to direct input.")
                        state = message.strip()
                else:
                    state = message.strip()
                    
                if not state:
                    state = message.strip()
                    
                self.phases['state']['value'] = state
                self.phases['state']['complete'] = True
                self.case_data['state'] = state
                
                # Check eligibility based on state and age
                is_eligible, reason = self.check_eligibility()
                if not is_eligible:
                    return {'eligible': False, 'reason': reason}
                    
                self.current_phase = 'complete'
                
                # Save answers and ranking to file
                success, error = self.save_case_data()
                if not success:
                    logger.warning(f"Could not save case data: {error}")
                
        except Exception as e:
            logger.error(f"Error processing response for phase {self.current_phase}: {e}")
            response_data['error'] = "I'm having trouble processing your response. Could you please try again?"
            
        return response_data

    def _is_affirmative(self, message: str) -> bool:
        """Determine if a message is affirmative (yes/positive)."""
        if not message:
            return False
            
        message_lower = message.lower().strip()
        
        # Quick check for common yes words
        yes_words = ['yes', 'yeah', 'yep', 'yup', 'sure', 'correct', 'right', 
                    'absolutely', 'definitely', 'indeed', 'affirmative', 'aye', 'y']
        
        # Split message into words to check for exact word matches
        message_words = message_lower.split()
        
        # Check for exact matches (for short responses like "y")
        if message_lower in yes_words:
            return True
        
        # Check if any word is a yes word
        if any(word in yes_words for word in message_words):
            return True
            
        # Check for phrases that start with yes
        if any(message_lower.startswith(word) for word in yes_words):
            return True
            
        # Check for other positive indicators
        positive_phrases = ['i do', 'we did', 'that is right', 'that is correct', 
                          'that\'s right', 'that\'s correct', 'had to', 'did have',
                          'we had', 'they did', 'doctor', 'received', 'cooling', 'blanket',
                          'mri', 'brain scan', 'scan', 'behind', 'delayed', 'delay', 'missing',
                          'not meeting', 'therapy', 'treatment', 'cool', 'attorney', 'spoke', 'spoken']
        if any(phrase in message_lower for phrase in positive_phrases):
            return True
            
        # Check for uncertainty phrases (only treat as affirmative if they contain positive indicators)
        uncertainty_phrases = ['i think', 'maybe', 'possibly', 'probably', 'might have', 'could have', 'not sure']
        
        # Only consider uncertainty as affirmative if there's no negative indicator
        negative_indicators = ['no', 'not', 'never', 'don\'t', 'didn\'t', 'doesn\'t', 'don\'t think']
        if any(phrase in message_lower for phrase in uncertainty_phrases):
            # If has negative indicators, don't treat as affirmative
            if any(neg in message_lower for neg in negative_indicators):
                return False
            # Otherwise treat uncertainty as affirmative
            return True
            
        return False

    def _legacy_pregnancy_parsing(self, message: str) -> None:
        """Legacy method to parse pregnancy details using regex patterns."""
        if not message:
            return
            
        # Parse pregnancy weeks
        weeks_match = re.search(r'(\d+)\s*(?:weeks|week|wks|wk)', message.lower())
        if weeks_match:
            try:
                weeks = int(weeks_match.group(1))
                self.case_data['weeks_pregnant'] = weeks
                
                # Update points based on gestational age
                if weeks < 30:
                    self.update_points(15, "Very premature birth (< 30 weeks)")
                elif weeks < 36:
                    self.update_points(10, "Premature birth (< 36 weeks)")
                else:
                    self.update_points(-5, "Full term birth (≥ 36 weeks)")
            except (ValueError, TypeError):
                logger.warning(f"Could not parse weeks from match: {weeks_match.group(1)}")
        
        # Check for "full term" or "term" mentions (implying ~40 weeks)
        elif re.search(r'full(?:\s*[ -]?term)?|(?:^|\s+)term', message.lower()):
            self.case_data['weeks_pregnant'] = 40
            self.update_points(-5, "Full term birth (implicitly 40 weeks)")
        
        # Check for difficult delivery using pattern matching
        difficult_delivery = any(term in message.lower() for term in [
            'difficult', 'complications', 'emergency', 'c-section', 'csection', 
            'c section', 'cesarean', 'forceps', 'vacuum', 'distress', 'oxygen', 
            'resuscitate', 'yes', 'not easy', 'issues', 'problems', 'hard', 
            'challenging', 'traumatic', 'prolonged', 'stuck', 'breech'
        ])
        
        self.case_data['difficult_delivery'] = difficult_delivery
        
        # Update points based on delivery
        if difficult_delivery:
            self.update_points(15, "Difficult delivery reported")
        else:
            self.update_points(-10, "No difficult delivery reported")

    def _legacy_duration_parsing(self, message: str) -> int:
        """Legacy method to parse NICU duration using regex patterns."""
        if not message:
            return 0
            
        message_lower = message.lower()
        total_days = 0
        
        # Check for months
        month_match = re.search(r'(\d+)\s*(?:months?|mos?)', message_lower)
        if month_match:
            try:
                months = int(month_match.group(1))
                total_days += months * 30  # Approximate
            except (ValueError, TypeError):
                pass
                
        # Check for weeks
        week_match = re.search(r'(\d+)\s*(?:weeks?|wks?)', message_lower)
        if week_match:
            try:
                weeks = int(week_match.group(1))
                total_days += weeks * 7
            except (ValueError, TypeError):
                pass
                
        # Check for days
        day_match = re.search(r'(\d+)\s*(?:days?|d)', message_lower)
        if day_match:
            try:
                days = int(day_match.group(1))
                total_days += days
            except (ValueError, TypeError):
                pass
                
        # Check for common phrases
        if re.search(r'couple(?:\s+of)?\s+days?', message_lower):
            total_days += 2
        elif re.search(r'few(?:\s+of)?\s+days?', message_lower):
            total_days += 3
        elif re.search(r'couple(?:\s+of)?\s+weeks?', message_lower):
            total_days += 14
        elif re.search(r'few(?:\s+of)?\s+weeks?', message_lower):
            total_days += 21
        elif re.search(r'about\s+a\s+week', message_lower):
            total_days += 7
        elif re.search(r'week\s+and\s+(?:a\s+)?half', message_lower):
            total_days += 10
            
        return total_days

    def _handle_back_request(self) -> Dict:
        """Handle request to go back to a previous question."""
        # Get the previous phase
        phases_list = list(self.phases.keys())
        try:
            current_index = phases_list.index(self.current_phase)
        except ValueError:
            logger.error(f"Invalid current phase: {self.current_phase}")
            return {'error': "An error occurred. Let's continue with the current question."}
        
        if current_index <= 0:  # We're at age, can't go back
            return {'error': "We can't go back any further. Let's continue with the current question."}
            
        # Move back one phase
        prev_phase = phases_list[current_index - 1]
        
        # Ensure previous phase exists and has the required fields
        if prev_phase not in self.phases:
            logger.error(f"Previous phase {prev_phase} not found in self.phases")
            return {'error': "An error occurred. Let's continue with the current question."}
        
        # Initialize missing fields if needed
        if 'complete' not in self.phases[prev_phase]:
            logger.error(f"Previous phase {prev_phase} missing 'complete' field")
            self.phases[prev_phase]['complete'] = False
            
        if 'question' not in self.phases[prev_phase]:
            logger.error(f"Previous phase {prev_phase} missing 'question' field")
            # Set a default question based on the phase
            default_questions = {
                'initial': "Hi, If your child has or may have CEREBRAL PALSY please message us YES to see if we can offer FREE help for you and your family today!",
                'age': "How old is your child with CP?",
                'pregnancy': "How many weeks pregnant were you when your child was born? Did your child have a difficult delivery?",
                'nicu': "Did your child go to the NICU after birth?",
                'nicu_duration': "How long was your child in the NICU for after birth?",
                'hie_therapy': "Did your child receive head cooling or HIE therapy while in the NICU?",
                'brain_scan': "Did your child receive an MRI or Brain Scan while in the NICU?",
                'milestones': "Is your child missing any milestones and or having any delays?",
                'lawyer': "This sounds like it definitely needs to be looked into further. Have you had your case reviewed by a lawyer yet?",
                'state': "In what State was your child born?"
            }
            self.phases[prev_phase]['question'] = default_questions.get(prev_phase, f"Let's continue with the {prev_phase} phase.")
            
        if 'value' not in self.phases[prev_phase]:
            logger.error(f"Previous phase {prev_phase} missing 'value' field")
            self.phases[prev_phase]['value'] = None
        
        self.phases[prev_phase]['complete'] = False
        self.current_phase = prev_phase
        
        return {'back': True}

    def _handle_help_request(self) -> Dict:
        """Provide help or explanation for the current phase."""
        help_messages = {
            'age': "I need to know how old your child is. You can provide the age in years, like '5 years old' or just '5'.",
            'pregnancy': "I'm asking about your pregnancy length (in weeks) when your child was born, and if there were any complications during delivery.",
            'nicu': "NICU stands for Neonatal Intensive Care Unit. I'm asking if your child needed to stay in the NICU after birth.",
            'nicu_duration': "I need to know how long your child stayed in the NICU. You can answer in days, weeks, or months.",
            'hie_therapy': "HIE therapy (also called head cooling) is a treatment used for babies who experienced oxygen deprivation during birth. I'm asking if your child received this treatment.",
            'brain_scan': "I'm asking if your child had a brain imaging test (MRI or other scan) while in the NICU.",
            'milestones': "Developmental milestones are skills like rolling over, sitting up, walking, or talking that children typically develop at certain ages. I'm asking if your child is delayed in any of these areas.",
            'lawyer': "I'm asking if you've already consulted with a lawyer about your child's case.",
            'state': "I need to know which US state your child was born in. This helps determine eligibility based on state-specific laws."
        }
        
        # Check if we have a help message for the current phase
        if self.current_phase in help_messages:
            return {'help': help_messages[self.current_phase]}
        else:
            # Fallback help message
            return {'help': "I'm gathering information about your child's case to see if we can help. Please answer the current question as best you can."}

    def get_next_question(self) -> Tuple[str, bool]:
        """
        Determines the next appropriate question or message.
        """
        if self.empty_response_count >= 3:
            return "I notice you haven't responded. Would you like to continue with the consultation? Please type 'yes' to continue or 'quit' to end our conversation.", True
            
        if self.current_phase == 'complete':
            return self._get_scheduling_message(), False
            
        return self.phases[self.current_phase]['question'], False

    def _get_scheduling_message(self) -> str:
        """Generates the standard message for connecting to a live representative."""
        rating = ""
        if self.case_data.get('ranking') in ['high', 'very high']:
            rating = "Based on your answers, your case shows strong potential. "
            
        return (f"Thank you! {rating}We'll connect you with a representative who will "
                "ask you a few more questions and schedule a FREE case review call with one of our affiliate lawyers. "
                "There is no fee or cost to you.")


class ClaudeChat:
    """
    Main chatbot class that coordinates conversation flow and model interactions.
    Now with integrated Claude NLU capabilities.
    """
    def _configure_environment(self):
        """Configure environment variables for Dreamhost compatibility."""
        import os
        import sys
        
        # Disable proxy settings
        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''
        os.environ['http_proxy'] = ''
        os.environ['https_proxy'] = ''
        os.environ['NO_PROXY'] = '*'
        os.environ['no_proxy'] = '*'
        
        # Configure SSL certificate verification if needed
        if 'dreamhost' in os.environ.get('SERVER_SOFTWARE', '').lower():
            os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'
        
        # Ensure python paths are correct for Dreamhost
        dreamhost_paths = [
            '/home/z31b4r1k3v0rk/.local/lib/python3.9/site-packages',
            '/usr/local/lib/python3.9/dist-packages'
        ]
        
        for path in dreamhost_paths:
            if path not in sys.path and os.path.exists(path):
                sys.path.append(path)
                
        logger.info(f"Environment configured for Dreamhost compatibility")


    def __init__(self, api_key: str, max_examples: int = 3):
        try:
            # Configure Dreamhost-specific environment
            self._configure_environment()
            
            model_config = ModelConfiguration()
            
            # Use our safe wrapper instead
            self.client = SafeAnthropicClient(api_key)
            
            self.model = model_config.model_version
            self.max_examples = max_examples
            self.conversation_history = []
            self.legal_rules = None
            self.shutdown_requested = False
            
            # Don't use signal handlers in web context
            if __name__ == "__main__":
                # Only register signal handlers in main thread of main interpreter
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info(f"ClaudeChat initialized with model {self.model}")
        except Exception as e:
            logger.critical(f"Fatal error initializing ClaudeChat: {e}")
            # Create minimal functioning chat even if initialization fails
            self.client = None
            self.model = "claude-3-5-sonnet-20241022"  # Fallback model
            self.max_examples = max_examples
            self.conversation_history = []
            self.legal_rules = None
            self.shutdown_requested = False
            raise
    
    def _is_first_yes(self, message: str) -> bool:
        """Check if the first message is an affirmative response"""
        if not message:
            return False
            
        message_lower = message.lower().strip()
        
        # Check for common yes words and patterns
        yes_patterns = ['yes', 'yeah', 'yep', 'yup', 'sure', 'y', 
                        'correct', 'right', 'absolutely', 'definitely']
        
        # For the first message, we want to be more permissive with what counts as "yes"
        return any(pattern in message_lower for pattern in yes_patterns) or message_lower.startswith('y')
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals gracefully."""
        logger.info("Received shutdown signal. Preparing to exit...")
        self.shutdown_requested = True
        # Give time for current operations to complete
        time.sleep(1)

    def load_examples(self, directory: str) -> List[Dict]:
        """Loads conversation examples from JSON files."""
        examples = []
        try:
            if not os.path.exists(directory):
                logger.warning(f"Examples directory {directory} does not exist")
                return []
                
            for filename in sorted(os.listdir(directory))[:self.max_examples]:
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(directory, filename)) as f:
                            example = json.load(f)
                            examples.append(example)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing example file {filename}: {e}")
                    except Exception as e:
                        logger.error(f"Error loading example file {filename}: {e}")
        except Exception as e:
            logger.warning(f"Could not load examples: {e}")
        return examples

    def initialize_rules(self, criteria_file: str):
        """Initializes legal rules and conversation manager with Claude NLU."""
        try:
            if os.path.exists(criteria_file):
                with open(criteria_file) as f:
                    self.legal_rules = json.load(f)
                logger.info(f"Loaded legal rules from {criteria_file}")
            else:
                logger.warning(f"Criteria file {criteria_file} not found")
                self.legal_rules = {"stateSOL": {}, "globalExclusions": {"excludedStates": {"list": []}}}
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in criteria file: {e}")
            self.legal_rules = {"stateSOL": {}, "globalExclusions": {"excludedStates": {"list": []}}}
        except Exception as e:
            logger.error(f"Failed to load criteria file: {e}")
            self.legal_rules = {"stateSOL": {}, "globalExclusions": {"excludedStates": {"list": []}}}
            
        # Initialize the conversation manager with Claude NLU
        self.conversation_manager = ConversationManager(self.legal_rules, self.client, self.model)

    def process_message(self, user_input: str) -> bool:
        """
        Processes user message and generates response.
        Returns False if conversation should end.
        """
        # Sanitize input
        if user_input is None:
            user_input = ""
        user_input = str(user_input).strip()
        
        # Truncate extremely long messages
        if len(user_input) > MAX_RESPONSE_LENGTH:
            logger.warning(f"Truncating user input from {len(user_input)} to {MAX_RESPONSE_LENGTH} chars")
            user_input = user_input[:MAX_RESPONSE_LENGTH] + "..."
        
        # Add to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Handle special commands
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye', 'end']:
            print("\nAssistant: Thank you for your time. Goodbye!")
            return False
            
        # Check for help/confusion indicators in the message
        if user_input.lower() in ['help', '?', 'commands'] or any(term in user_input.lower() for term in ['explain', 'confused', "don't understand", "what's this"]):
            try:
                # Get phase-specific help
                help_data = self.conversation_manager._handle_help_request()
                if 'help' in help_data:
                    help_message = help_data['help']
                    print(f"\nAssistant: {help_message}")
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": help_message
                    })
                    return True
            except Exception as e:
                logger.error(f"Error getting help: {e}")
                
            # Fallback to generic help
            help_text = ("Available commands:\n"
                        "  help - Show this help message\n"
                        "  back - Go back to the previous question\n"
                        "  quit - End the conversation\n")
            print(f"\nAssistant: {help_text}")
            self.conversation_history.append({
                "role": "assistant",
                "content": help_text
            })
            return True
        
        # Analyze response and check eligibility
        try:
            response_data = self.conversation_manager.analyze_response(user_input)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            print("\nAssistant: I'm having trouble processing your response. Could you please try again?")
            return True
        
        # Handle "help" response
        if 'help' in response_data:
            help_message = response_data['help']
            print(f"\nAssistant: {help_message}")
            self.conversation_history.append({
                "role": "assistant",
                "content": help_message
            })
            return True
            
        # Handle "back" response
        if response_data.get('back', False):
            # Go back to previous question
            next_question, _ = self.conversation_manager.get_next_question()
            print(f"\nAssistant: Let's go back to a previous question. {next_question}")
            self.conversation_history.append({
                "role": "assistant",
                "content": f"Let's go back to a previous question. {next_question}"
            })
            return True
        
        # Check if the conversation should end after lawyer question
        if response_data.get('end_chat'):
            farewell_message = response_data.get('farewell_message', "Thank you for your time.")
            print(f"\nAssistant: {farewell_message}")
            self.conversation_history.append({
                "role": "assistant",
                "content": farewell_message
            })
            return False
        
        # Handle errors in age parsing
        if 'error' in response_data:
            print(f"\nAssistant: {response_data['error']}")
            return True
        
        # Handle ineligibility
        if isinstance(response_data, dict) and not response_data.get('eligible', True):
            print(f"\nAssistant: {response_data['reason']}")
            print("I'll need to transfer you to a live representative who can better assist you.")
            try:
                self.conversation_manager.save_case_data()  # Save data even for ineligible cases
            except Exception as e:
                logger.error(f"Could not save case data: {e}")
            return False
        
        # Get next question
        next_question, is_control = self.conversation_manager.get_next_question()
        
        # Add sympathy message for difficult delivery if applicable
        if response_data.get('sympathy_message'):
            next_question = f"{response_data['sympathy_message']} {next_question}"
        
        # Handle control messages
        if is_control:
            if self.conversation_manager.empty_response_count >= 3:
                print(f"\nAssistant: {next_question}")
                try:
                    user_response = input("\nYou: ").strip().lower()
                    if user_response != 'yes':
                        return False
                    self.conversation_manager.empty_response_count = 0
                    next_question, _ = self.conversation_manager.get_next_question()
                except Exception as e:
                    logger.error(f"Error getting user input: {e}")
                    return False
        
        # Add response to conversation history and print
        print(f"\nAssistant: {next_question}")
        self.conversation_history.append({
            "role": "assistant",
            "content": next_question
        })
        
        # Show current ranking in the terminal (for debugging)
        if hasattr(self, 'conversation_manager') and hasattr(self.conversation_manager, 'case_data'):
            current_rank = self.conversation_manager.case_data.get('ranking', 'normal')
            current_points = self.conversation_manager.case_data.get('points', 50)
            logger.info(f"Current case ranking: {current_rank} ({current_points} points)")
            
        return True

    def chat(self, examples_dir: str, criteria_file: str):
        """
        Runs the main chat loop for CP case intake.
        """
        # Initialize components
        try:
            self.initialize_rules(criteria_file)
            self.example_conversations = self.load_examples(examples_dir)
            
            # Start conversation - DON'T show the initial question
            print("\nChatbot initialized. Type 'quit' to exit or 'help' for commands.")
            print("\nWaiting for first message...")
            
            # Main chat loop
            while not self.shutdown_requested:
                try:
                    user_input = input("\nYou: ").strip()
                    if user_input.lower() == 'quit':
                        break
                        
                    # For the first message, directly go to processing it as a response
                    # The system now starts at the age question directly
                    if len(self.conversation_history) == 0:
                        # Show the age question directly for the first message
                        age_question = self.conversation_manager.phases['age']['question']
                        print(f"\nAssistant: {age_question}")
                        
                        # Add to conversation history
                        self.conversation_history.append({"role": "user", "content": user_input})
                        self.conversation_history.append({"role": "assistant", "content": age_question})
                        continue
                    
                    should_continue = self.process_message(user_input)
                    if not should_continue:
                        break
                        
                except KeyboardInterrupt:
                    # Let signal handler take care of this
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in chat loop: {e}")
                    print("\nAn error occurred. Let's try again.")
                    
            # Save any unsaved data before exiting
            if hasattr(self, 'conversation_manager'):
                try:
                    self.conversation_manager.save_case_data()
                except Exception as e:
                    logger.error(f"Error saving case data on exit: {e}")
                    
            print("\nChat session ended.")
            
        except Exception as e:
            logger.critical(f"Fatal error in chat: {e}")
            print("\nI'm sorry, a critical error occurred and the system needs to restart.")
            print("Any information you've provided has been saved.")

def main():
    """
    Main entry point for the chatbot application.
    """
    try:
        # Debug environment variables
        logger.info("Debugging environment variables:")
        for key, value in os.environ.items():
            # Don't log sensitive values
            if 'key' in key.lower() or 'token' in key.lower() or 'secret' in key.lower() or 'password' in key.lower():
                logger.info(f"ENV: {key}=<REDACTED>")
            else:
                logger.info(f"ENV: {key}={value}")
        
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        
        # Load environment variables
        load_dotenv()
        
        # Get configuration values
        api_key = os.getenv('ANTHROPIC_API_KEY')
        examples_dir = os.getenv('EXAMPLES_DIR', './examples')
        criteria_file = os.getenv('CRITERIA_FILE', './criteria.json')
        max_examples = int(os.getenv('MAX_EXAMPLES', '3'))
        
        # Validate API key
        if not api_key:
            logger.error("Missing required ANTHROPIC_API_KEY environment variable.")
            print("Error: Missing required ANTHROPIC_API_KEY environment variable.")
            print("Please set this value in your .env file or environment variables.")
            return 1
        
        # Initialize and run chatbot
        chat = ClaudeChat(api_key, max_examples)
        chat.chat(examples_dir, criteria_file)
        return 0
    
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}")
        print(f"A critical error occurred: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())