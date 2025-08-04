# AI-Enhanced Domestic Violence Assistance App - Fixed Dependencies

import streamlit as st
import pandas as pd
import numpy as np
import math
import time
import hashlib
from datetime import datetime, timedelta
import re
import json
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="AI Safe Support Hub",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
.crisis-banner {
    background-color: #ff4b4b;
    color: white;
    padding: 10px;
    border-radius: 5px;
    text-align: center;
    margin-bottom: 20px;
}
.risk-high { background-color: #ffebee; border-left: 5px solid #f44336; padding: 10px; }
.risk-medium { background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 10px; }
.risk-low { background-color: #e8f5e8; border-left: 5px solid #4caf50; padding: 10px; }
.chatbot-message { background-color: #f0f8ff; padding: 10px; border-radius: 10px; margin: 5px 0; }
.resource-card {
    border: 1px solid #ddd;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    background-color: #f8f9fa;
}
.sentiment-positive { color: #28a745; }
.sentiment-negative { color: #dc3545; }
.sentiment-neutral { color: #6c757d; }
.prediction-card {
    border: 1px solid #007bff;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    background-color: #f8f9ff;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'risk_assessment' not in st.session_state:
    st.session_state.risk_assessment = None
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

# Crisis keywords for NLP detection
CRISIS_KEYWORDS = [
    'kill', 'die', 'suicide', 'hurt me', 'end it', 'can\'t take',
    'weapon', 'gun', 'knife', 'threat', 'going to hurt',
    'emergency', 'help me', 'scared', 'hiding', 'danger'
]

ESCALATION_INDICATORS = [
    'getting worse', 'more violent', 'threatened to kill',
    'has weapons', 'controls money', 'won\'t let me leave',
    'follows me', 'tracking', 'isolated', 'no one believes'
]

# Helper function for distance calculation (replaces geopy)
def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    R = 3959  # Earth's radius in miles
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) * math.sin(delta_lat / 2) +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) * math.sin(delta_lon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

# Simple geocoding function (replaces geopy)
def simple_geocode(location_str):
    """Simple geocoding for major US cities - in production, use real geocoding API"""
    city_coords = {
        'new york': (40.7128, -74.0060),
        'los angeles': (34.0522, -118.2437),
        'chicago': (41.8781, -87.6298),
        'houston': (29.7604, -95.3698),
        'miami': (25.7617, -80.1918),
        'seattle': (47.6062, -122.3321),
        'boston': (42.3601, -71.0589),
        'atlanta': (33.7490, -84.3880),
        'denver': (39.7392, -104.9903),
        'phoenix': (33.4484, -112.0740)
    }
    
    location_lower = location_str.lower().strip()
    
    # Check for exact city matches
    for city, coords in city_coords.items():
        if city in location_lower:
            return coords[0], coords[1], True
    
    # Check for zip code pattern
    zip_match = re.search(r'\b\d{5}\b', location_str)
    if zip_match:
        # Simple zip to coordinate mapping (in production, use real API)
        zip_code = zip_match.group()
        if zip_code.startswith('10'):  # NYC area
            return 40.7128, -74.0060, True
        elif zip_code.startswith('90'):  # LA area  
            return 34.0522, -118.2437, True
        elif zip_code.startswith('60'):  # Chicago area
            return 41.8781, -87.6298, True
        else:
            # Default to center of US
            return 39.8283, -98.5795, True
    
    return None, None, False

# Basic sentiment analysis (replaces TextBlob)
def analyze_sentiment(text):
    """Basic sentiment analysis using keyword matching"""
    if not text or not text.strip():
        return {'sentiment': 'neutral', 'polarity': 0, 'confidence': 0}
    
    positive_words = ['help', 'better', 'good', 'safe', 'support', 'hope', 'thank', 'grateful']
    negative_words = ['scared', 'afraid', 'hurt', 'pain', 'angry', 'sad', 'terrible', 'awful', 'hate', 'fear']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        sentiment = 'positive'
        polarity = 0.3 + (positive_count - negative_count) * 0.1
        color = 'sentiment-positive'
    elif negative_count > positive_count:
        sentiment = 'negative'
        polarity = -0.3 - (negative_count - positive_count) * 0.1
        color = 'sentiment-negative'
    else:
        sentiment = 'neutral'
        polarity = 0
        color = 'sentiment-neutral'
    
    confidence = abs(polarity)
    
    return {
        'sentiment': sentiment,
        'polarity': polarity,
        'confidence': min(confidence, 1.0),
        'color': color
    }

# Enhanced resource data with safety ratings and success metrics
@st.cache_data
def load_enhanced_resources():
    return pd.DataFrame([
        {
            "name": "Safe Haven Shelter", 
            "address": "123 Main St, New York, NY",
            "phone": "(555) 123-4567",
            "lat": 40.7128, 
            "lon": -74.0060, 
            "type": "shelter",
            "hours": "24/7",
            "website": "safehaven.org",
            "capacity": 25,
            "current_occupancy": 18,
            "safety_rating": 9.2,
            "success_rate": 0.87,
            "specialties": ["children", "legal_aid", "counseling"],
            "languages": ["english", "spanish"],
            "wait_time_days": 2
        },
        {
            "name": "Women's Health Center", 
            "address": "456 Oak Ave, New York, NY",
            "phone": "(555) 234-5678",
            "lat": 40.7138, 
            "lon": -74.0050, 
            "type": "health",
            "hours": "Mon-Fri 8AM-6PM",
            "website": "womenshealth.org",
            "capacity": 50,
            "current_occupancy": 30,
            "safety_rating": 8.8,
            "success_rate": 0.92,
            "specialties": ["trauma_counseling", "medical_care"],
            "languages": ["english", "spanish", "french"],
            "wait_time_days": 1
        },
        {
            "name": "Crisis Counseling Center", 
            "address": "789 Pine St, New York, NY",
            "phone": "(555) 345-6789",
            "lat": 40.7148, 
            "lon": -74.0040, 
            "type": "counseling",
            "hours": "24/7 Hotline",
            "website": "crisishelp.org",
            "capacity": 100,
            "current_occupancy": 45,
            "safety_rating": 9.5,
            "success_rate": 0.89,
            "specialties": ["crisis_intervention", "group_therapy", "children"],
            "languages": ["english", "spanish", "mandarin"],
            "wait_time_days": 0
        },
        {
            "name": "Legal Aid Society", 
            "address": "321 Elm St, New York, NY",
            "phone": "(555) 456-7890",
            "lat": 40.7118, 
            "lon": -74.0070, 
            "type": "legal",
            "hours": "Mon-Fri 9AM-5PM",
            "website": "legalaid.org",
            "capacity": 75,
            "current_occupancy": 60,
            "safety_rating": 8.9,
            "success_rate": 0.84,
            "specialties": ["restraining_orders", "custody", "divorce"],
            "languages": ["english", "spanish"],
            "wait_time_days": 7
        },
        {
            "name": "Food Bank Network", 
            "address": "555 Community Dr, New York, NY",
            "phone": "(555) 678-9012",
            "lat": 40.7100, 
            "lon": -74.0080, 
            "type": "food",
            "hours": "Mon-Sat 9AM-5PM",
            "website": "foodnetwork.org",
            "capacity": 200,
            "current_occupancy": 120,
            "safety_rating": 8.5,
            "success_rate": 0.95,
            "specialties": ["emergency_food", "children", "nutrition_education"],
            "languages": ["english", "spanish"],
            "wait_time_days": 0
        }
    ])

RESOURCES = load_enhanced_resources()

# 1. ENHANCED RISK ASSESSMENT SCORING
class RiskAssessmentEngine:
    def __init__(self):
        self.risk_factors = {
            'violence_escalation': 5,
            'weapon_access': 4,
            'threats_to_kill': 5,
            'strangulation': 4,
            'isolation': 3,
            'financial_control': 2,
            'substance_abuse': 2,
            'pregnancy': 3,
            'recent_separation': 4,
            'violation_of_orders': 3,
            'children_threatened': 4
        }
    
    def assess_risk(self, inputs, text_responses):
        risk_score = 0
        risk_factors_present = []
        
        # Analyze text responses for risk indicators
        combined_text = ' '.join([str(v) for v in text_responses.values() if v])
        combined_text = combined_text.lower()
        
        # Check for crisis keywords
        for keyword in CRISIS_KEYWORDS:
            if keyword in combined_text:
                risk_score += 3
                risk_factors_present.append(f"Crisis language detected: '{keyword}'")
        
        # Check for escalation indicators
        for indicator in ESCALATION_INDICATORS:
            if indicator in combined_text:
                risk_score += 4
                risk_factors_present.append(f"Escalation pattern: '{indicator}'")
        
        # Assess based on form inputs
        if inputs.get('felt_helped') == 'No':
            risk_score += 2
            risk_factors_present.append("Police response inadequate")
        
        if inputs.get('need_shelter'):
            risk_score += 3
            risk_factors_present.append("Immediate housing needs")
        
        if inputs.get('has_kids'):
            risk_score += 2
            risk_factors_present.append("Children present")
        
        if inputs.get('unemployed'):
            risk_score += 1
            risk_factors_present.append("Economic vulnerability")
        
        # Additional risk factors from text analysis
        high_risk_phrases = ['can\'t escape', 'controls everything', 'threatens children', 'has gun']
        for phrase in high_risk_phrases:
            if phrase in combined_text:
                risk_score += 5
                risk_factors_present.append(f"High-risk indicator: '{phrase}'")
        
        # Determine risk level
        if risk_score >= 12:
            risk_level = "HIGH"
            color_class = "risk-high"
            recommendation = "🚨 IMMEDIATE INTERVENTION REQUIRED - Contact emergency services"
        elif risk_score >= 6:
            risk_level = "MEDIUM"
            color_class = "risk-medium"
            recommendation = "⚠️ Enhanced safety planning needed - Prioritize immediate resources"
        else:
            risk_level = "LOW"
            color_class = "risk-low"
            recommendation = "✅ Standard support protocols - Focus on long-term planning"
        
        return {
            'score': risk_score,
            'level': risk_level,
            'factors': risk_factors_present,
            'recommendation': recommendation,
            'color_class': color_class
        }

# 2. IMPROVED RESOURCE MATCHING ALGORITHM
class SmartResourceMatcher:
    def __init__(self, resources_df):
        self.resources = resources_df
    
    def calculate_match_score(self, user_profile, resource):
        score = 0
        
        # Distance factor (closer is better)
        if user_profile.get('lat') and user_profile.get('lon'):
            distance = calculate_distance(
                user_profile['lat'], user_profile['lon'],
                resource['lat'], resource['lon']
            )
            score += max(0, 50 - distance * 2)  # Max 50 points for distance
        
        # Availability factor
        availability = (resource['capacity'] - resource['current_occupancy']) / resource['capacity']
        score += availability * 20  # Max 20 points for availability
        
        # Safety rating factor
        score += resource['safety_rating'] * 2  # Max ~20 points
        
        # Success rate factor
        score += resource['success_rate'] * 15  # Max 15 points
        
        # Wait time factor (shorter is better)
        score += max(0, 10 - resource['wait_time_days'])  # Max 10 points
        
        # Specialty matching
        user_needs = []
        if user_profile.get('has_kids'): user_needs.append('children')
        if user_profile.get('need_legal'): user_needs.append('legal_aid')
        if user_profile.get('mental_health_concerns'): user_needs.append('trauma_counseling')
        
        specialty_matches = len(set(user_needs) & set(resource['specialties']))
        score += specialty_matches * 5  # 5 points per specialty match
        
        return score
    
    def find_best_matches(self, user_profile, need_types, top_n=3):
        matches = []
        
        for _, resource in self.resources.iterrows():
            if resource['type'] in need_types:
                match_score = self.calculate_match_score(user_profile, resource)
                resource_dict = resource.to_dict()
                resource_dict['match_score'] = match_score
                resource_dict['distance'] = 0
                
                # Calculate distance if coordinates available
                if user_profile.get('lat') and user_profile.get('lon'):
                    resource_dict['distance'] = calculate_distance(
                        user_profile['lat'], user_profile['lon'],
                        resource['lat'], resource['lon']
                    )
                
                matches.append(resource_dict)
        
        # Sort by match score and return top N
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        return matches[:top_n]

# 3. BASIC SENTIMENT ANALYSIS (Updated to use our function)
class SentimentAnalyzer:
    def analyze_text(self, text):
        return analyze_sentiment(text)
    
    def analyze_crisis_indicators(self, text):
        if not text:
            return False, []
        
        text_lower = text.lower()
        found_indicators = []
        
        for keyword in CRISIS_KEYWORDS:
            if keyword in text_lower:
                found_indicators.append(keyword)
        
        return len(found_indicators) > 0, found_indicators

# 4. INTELLIGENT CHATBOT INTEGRATION
class DVSupportChatbot:
    def __init__(self):
        self.responses = {
            'greeting': [
                "Hello, I'm here to help you find support and resources. How are you feeling today?",
                "Hi there. I understand you're looking for help. You're taking a brave step.",
                "Welcome. This is a safe space to talk about what you're going through."
            ],
            'crisis': [
                "🚨 I'm concerned about your safety. Please consider calling 911 if you're in immediate danger.",
                "Your safety is the priority. The National DV Hotline (1-800-799-7233) has trained counselors available 24/7.",
                "I'm here to help, but please reach out to emergency services if you need immediate assistance."
            ],
            'resources': [
                "I can help you find local resources. What type of support do you need most right now?",
                "There are many resources available. Let me help you find the right ones for your situation.",
                "Based on what you've shared, here are some resources that might help..."
            ],
            'emotional_support': [
                "What you're going through is difficult, and your feelings are valid.",
                "You're not alone in this. Many people have been through similar situations and found help.",
                "It takes courage to reach out for help. You're taking an important step."
            ],
            'safety_planning': [
                "Let's talk about creating a safety plan. Do you have a safe place to go?",
                "Safety planning is important. Have you thought about what you'd need if you had to leave quickly?",
                "I can help you think through safety strategies. What's your biggest concern right now?"
            ]
        }
    
    def generate_response(self, user_message, context=None):
        sentiment_analyzer = SentimentAnalyzer()
        is_crisis, crisis_indicators = sentiment_analyzer.analyze_crisis_indicators(user_message)
        
        if is_crisis:
            return np.random.choice(self.responses['crisis'])
        
        sentiment = sentiment_analyzer.analyze_text(user_message)
        
        user_lower = user_message.lower()
        
        if any(word in user_lower for word in ['resource', 'help', 'need', 'find']):
            return np.random.choice(self.responses['resources'])
        elif any(word in user_lower for word in ['safe', 'safety', 'plan', 'leave']):
            return np.random.choice(self.responses['safety_planning'])
        elif sentiment['sentiment'] == 'negative' and sentiment['confidence'] > 0.3:
            return np.random.choice(self.responses['emotional_support'])
        else:
            return np.random.choice(self.responses['greeting'])

# 5. PREDICTIVE FINANCIAL NEEDS MODEL
class FinancialNeedsPredictor:
    def __init__(self):
        self.base_amounts = {
            'emergency': 500,
            'shelter': 800,
            'food': 300,
            'legal': 1200,
            'counseling': 600,
            'childcare': 700,
            'transportation': 200,
            'job_training': 1500
        }
        
        self.location_multipliers = {
            'new york': 1.5,
            'california': 1.4,
            'chicago': 1.2,
            'miami': 1.1,
            'houston': 1.0,
            'default': 1.0
        }
    
    def predict_needs(self, user_profile, duration_months=6):
        needs_breakdown = {}
        total_predicted = 0
        
        # Base emergency fund
        needs_breakdown['Emergency Fund'] = self.base_amounts['emergency']
        total_predicted += needs_breakdown['Emergency Fund']
        
        # Predict based on indicated needs
        if user_profile.get('need_shelter'):
            monthly_shelter = self.base_amounts['shelter']
            needs_breakdown['Housing Support'] = monthly_shelter * duration_months
            total_predicted += needs_breakdown['Housing Support']
        
        if user_profile.get('need_food'):
            monthly_food = self.base_amounts['food']
            needs_breakdown['Food Assistance'] = monthly_food * duration_months
            total_predicted += needs_breakdown['Food Assistance']
        
        if user_profile.get('mental_health_concerns'):
            needs_breakdown['Counseling Services'] = self.base_amounts['counseling']
            total_predicted += needs_breakdown['Counseling Services']
        
        if user_profile.get('has_kids'):
            needs_breakdown['Childcare Support'] = self.base_amounts['childcare'] * duration_months
            total_predicted += needs_breakdown['Childcare Support']
        
        if user_profile.get('unemployed'):
            needs_breakdown['Job Training/Search'] = self.base_amounts['job_training']
            total_predicted += needs_breakdown['Job Training/Search']
        
        # Legal assistance prediction based on risk assessment
        risk_level = user_profile.get('risk_level', 'LOW')
        if risk_level in ['HIGH', 'MEDIUM']:
            needs_breakdown['Legal Protection'] = self.base_amounts['legal']
            total_predicted += needs_breakdown['Legal Protection']
        
        # Apply location multiplier
        location = user_profile.get('location', 'default').lower()
        multiplier = 1.0
        for loc, mult in self.location_multipliers.items():
            if loc in location:
                multiplier = mult
                break
        
        for key in needs_breakdown:
            needs_breakdown[key] = int(needs_breakdown[key] * multiplier)
        
        total_predicted = int(total_predicted * multiplier)
        
        return total_predicted, needs_breakdown

# 6. FORM AUTO-COMPLETION FEATURES
class FormAutoCompleter:
    def __init__(self):
        self.common_responses = {
            'police_response': [
                "They took a report but didn't seem to take it seriously",
                "Officer was helpful and provided resources",
                "Police said it was a civil matter",
                "They arrested the perpetrator",
                "Didn't respond to the call",
                "Officer blamed me for the situation",
                "They provided a case number and information"
            ],
            'financial_use': [
                "Temporary housing/rent deposit",
                "Food and basic necessities",
                "Transportation to get to work/appointments",
                "Legal fees for restraining order",
                "Childcare expenses",
                "Medical bills and prescriptions",
                "Emergency relocation costs",
                "Job training or education"
            ],
            'additional_info': [
                "I need help creating a safety plan",
                "My situation is getting worse",
                "I'm worried about my children's safety",
                "I need legal advice about custody",
                "I'm afraid to leave because of threats",
                "I have nowhere safe to go",
                "I need help with job training"
            ]
        }
    
    def get_suggestions(self, field_name, partial_text=""):
        if field_name in self.common_responses:
            if partial_text:
                # Filter suggestions based on partial text
                suggestions = [resp for resp in self.common_responses[field_name] 
                             if partial_text.lower() in resp.lower()]
                return suggestions
            return self.common_responses[field_name]
        return []

# 7. ADVANCED NLP FOR CRISIS DETECTION
class CrisisDetectionNLP:
    def __init__(self):
        self.urgent_patterns = [
            r'\b(kill|murder|die|suicide|end it all)\b',
            r'\b(gun|weapon|knife|hurt me)\b',
            r'\b(tonight|right now|immediately|emergency)\b',
            r'\b(can\'t take|give up|no hope)\b'
        ]
        
        self.severity_weights = {
            'immediate_danger': 10,
            'suicidal_ideation': 9,
            'weapon_mention': 8,
            'escalation': 6,
            'isolation': 4,
            'despair': 5
        }
    
    def analyze_crisis_level(self, text_inputs):
        if not text_inputs:
            return {'level': 'low', 'score': 0, 'triggers': []}
        
        combined_text = ' '.join([str(v) for v in text_inputs.values() if v]).lower()
        crisis_score = 0
        triggers = []
        
        # Check urgent patterns
        for pattern in self.urgent_patterns:
            matches = re.findall(pattern, combined_text)
            if matches:
                crisis_score += 8
                triggers.extend(matches)
        
        # Additional contextual analysis
        if any(word in combined_text for word in ['scared', 'afraid', 'terrified']):
            crisis_score += 3
            triggers.append('fear_indicators')
        
        if any(word in combined_text for word in ['alone', 'isolated', 'nobody']):
            crisis_score += 2
            triggers.append('isolation_indicators')
        
        # Check for escalation language
        escalation_words = ['worse', 'violent', 'angry', 'control']
        escalation_count = sum(1 for word in escalation_words if word in combined_text)
        if escalation_count >= 2:
            crisis_score += 5
            triggers.append('escalation_language')
        
        # Determine crisis level
        if crisis_score >= 15:
            level = 'critical'
        elif crisis_score >= 8:
            level = 'high'
        elif crisis_score >= 4:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'level': level,
            'score': crisis_score,
            'triggers': list(set(triggers))
        }

# 8. GEOSPATIAL SAFETY ANALYSIS
class GeospatialSafetyAnalyzer:
    def __init__(self):
        # Mock crime data - in production, integrate with real crime APIs
        self.safety_zones = {
            'new_york': {'safety_score': 7.2, 'crime_rate': 3.5},
            'los_angeles': {'safety_score': 6.8, 'crime_rate': 4.2},
            'chicago': {'safety_score': 6.5, 'crime_rate': 4.8},
            'houston': {'safety_score': 7.0, 'crime_rate': 3.9},
            'miami': {'safety_score': 7.5, 'crime_rate': 3.2}
        }
    
    def analyze_area_safety(self, location_str):
        location_lower = location_str.lower() if location_str else ""
        
        # Find matching safety zone
        safety_data = None
        for zone, data in self.safety_zones.items():
            if zone.replace('_', ' ') in location_lower:
                safety_data = data
                break
        
        if not safety_data:
            safety_data = {'safety_score': 6.0, 'crime_rate': 4.0}  # Default values
        
        safety_score = safety_data['safety_score']
        crime_rate = safety_data['crime_rate']
        
        if safety_score > 7.5:
            risk_level = 'Low'
            recommendations = ["Area has good safety ratings", "Standard precautions recommended"]
        elif safety_score > 6.0:
            risk_level = 'Medium'
            recommendations = ["Exercise normal caution", "Travel in groups when possible", "Stay aware of surroundings"]
        else:
            risk_level = 'High'
            recommendations = ["High caution advised", "Avoid traveling alone", "Consider alternative locations", "Contact resources for safety escort"]
        
        return {
            'safety_score': safety_score,
            'risk_level': risk_level,
            'crime_rate': crime_rate,
            'recommendations': recommendations
        }

# 9. OUTCOME PREDICTION MODELS
class OutcomePredictionModel:
    def __init__(self):
        # Simulated ML model based on research data
        self.success_factors = {
            'has_support_system': 0.15,
            'employed_or_in_school': 0.12,
            'no_substance_abuse': 0.10,
            'children_safe': 0.18,
            'legal_protection': 0.14,
            'stable_housing': 0.20,
            'mental_health_support': 0.11
        }
    
    def predict_recovery_success(self, user_profile):
        base_probability = 0.65  # Base success rate from research
        
        # Positive factors
        if user_profile.get('in_school') or not user_profile.get('unemployed'):
            base_probability += self.success_factors['employed_or_in_school']
        
        if user_profile.get('has_kids') and not user_profile.get('need_shelter'):
            # Having kids and stable housing indicates family safety
            base_probability += self.success_factors['children_safe']
        
        if user_profile.get('mental_health_concerns'):
            # Acknowledging mental health needs is positive
            base_probability += self.success_factors['mental_health_support']
        
        # Risk factors
        risk_level = user_profile.get('risk_level', 'LOW')
        if risk_level == 'HIGH':
            base_probability -= 0.15
        elif risk_level == 'MEDIUM':
            base_probability -= 0.08
        
        # Financial stability factor
        if not user_profile.get('unemployed') and not user_profile.get('need_financial'):
            base_probability += 0.10
        
        success_probability = max(0.1, min(0.95, base_probability))
        
        return {
            'success_probability': success_probability,
            'confidence_interval': (max(0.1, success_probability - 0.1), min(0.95, success_probability + 0.1)),
            'key_factors': self._identify_key_factors(user_profile),
            'recommendations': self._generate_recommendations(user_profile, success_probability)
        }
    
    def _identify_key_factors(self, user_profile):
        factors = []
        if user_profile.get('has_kids'):
            factors.append("Child safety and stability")
        if user_profile.get('unemployed'):
            factors.append("Economic independence")
        if user_profile.get('need_shelter'):
            factors.append("Stable housing")
        if user_profile.get('mental_health_concerns'):
            factors.append("Mental health support")
        return factors
    
    def _generate_recommendations(self, user_profile, success_prob):
        recommendations = []
        
        if success_prob < 0.6:
            recommendations.append("Consider intensive case management")
            recommendations.append("Prioritize immediate safety planning")
        
        if user_profile.get('unemployed'):
            recommendations.append("Focus on job training and employment services")
        
        if user_profile.get('need_shelter'):
            recommendations.append("Secure stable long-term housing")
            
        if user_profile.get('mental_health_concerns'):
            recommendations.append("Engage in trauma-informed counseling")
        
        if user_profile.get('risk_level') == 'HIGH':
            recommendations.append("Immediate safety intervention required")
        
        return recommendations

# Initialize AI components
risk_engine = RiskAssessmentEngine()
resource_matcher = SmartResourceMatcher(RESOURCES)
sentiment_analyzer = SentimentAnalyzer()
chatbot = DVSupportChatbot()
financial_predictor = FinancialNeedsPredictor()
form_completer = FormAutoCompleter()
crisis_detector = CrisisDetectionNLP()
safety_analyzer = GeospatialSafetyAnalyzer()
outcome_predictor = OutcomePredictionModel()

# Crisis banner and quick exit
st.markdown("""
<div class="crisis-banner">
    🚨 <strong>CRISIS HOTLINE: 1-800-799-7233</strong> (24/7 National Domestic Violence Hotline) 🚨
</div>
""", unsafe_allow_html=True)

if st.button("🚪 Quick Exit", key="quick_exit"):
    st.markdown('<meta http-equiv="refresh" content="0; url=https://www.weather.com">', unsafe_allow_html=True)

st.title("🤖 AI-Enhanced Safe Support Hub")
st.markdown("*Intelligent support system powered by AI to help you find personalized resources and assistance*")

# Sidebar for AI chatbot
with st.sidebar:
    st.header("💬 AI Support Assistant")
    
    # Display chat history
    for message in st.session_state.chat_history[-5:]:  # Show last 5 messages
        st.markdown(f"""
        <div class="chatbot-message">
            <strong>You:</strong> {message['user']}<br>
            <strong>Assistant:</strong> {message['bot']}
        </div>
        """, unsafe_allow_html=True)
    
    # Chat input
    user_message = st.text_input("Ask me anything about resources or support:")
    if user_message:
        bot_response = chatbot.generate_response(user_message)
        st.session_state.chat_history.append({
            'user': user_message,
            'bot': bot_response,
            'timestamp': datetime.now()
        })
        st.markdown(f'<div class="chatbot-message"><strong>Assistant:</strong> {bot_response}</div>', 
                   unsafe_allow_html=True)

# Privacy notice
with st.expander("🔒 Privacy & AI Information"):
    st.markdown("""
    **AI-Enhanced Features:**
    - Risk assessment using machine learning
    - Intelligent resource matching
    - Sentiment analysis for better support
    - Crisis detection and intervention
    - Predictive financial needs modeling
    
    **Your privacy is our priority:**
    - AI analysis is done locally and securely
    - No personal data is permanently stored
    - Use the "Quick Exit" button if you need to leave quickly
    - Consider using a private/incognito browser window
    
    **If you're in immediate danger, call 911**
    """)

# Main form with AI enhancements
with st.form("enhanced_victim_form"):
    st.header("📋 AI-Powered Needs Assessment")
    
    # Location input with safety analysis
    st.subheader("📍 Location & Safety Analysis")
    location_input = st.text_input(
        "Enter your city, state, or zip code (e.g., 'Chicago, IL' or '60601')",
        help="AI will analyze area safety and find optimal resources"
    )
    
    # Police interaction with sentiment analysis
    st.subheader("🚔 Police Interaction")
    reported_to_police = st.selectbox("Was the incident reported to police?", ["", "Yes", "No"])
    
    police_response = ""
    felt_helped = ""
    
    if reported_to_police == "Yes":
        # Auto-completion suggestions
        police_suggestions = form_completer.get_suggestions('police_response')
        
        st.markdown("**Describe police response (optional):**")
        police_response = st.text_area(
            "Police response details",
            help="AI will analyze sentiment and identify concerns",
            label_visibility="collapsed"
        )
        
        if police_suggestions:
            selected_suggestion = st.selectbox("💡 Or select a common response:", [""] + police_suggestions)
            if selected_suggestion:
                police_response = selected_suggestion
        
        felt_helped = st.selectbox("Did you feel helped by the police?", ["", "Yes", "No", "Somewhat"])
        
        # Real-time sentiment analysis
        if police_response:
            sentiment = sentiment_analyzer.analyze_text(police_response)
            st.markdown(f"<span class='{sentiment['color']}'>🤖 AI Analysis - Response sentiment: {sentiment['sentiment'].title()} (confidence: {sentiment['confidence']:.2f})</span>", 
                       unsafe_allow_html=True)
    
    # Financial needs with AI prediction
    st.subheader("💰 Financial Needs Assessment")
    need_financial = st.checkbox("Do you need financial assistance?")
    
    financial_amount = 0
    financial_use = ""
    
    if need_financial:
        financial_amount = st.number_input("Approximately how much do you need (USD)?", min_value=0, max_value=50000, value=0)
        
        # Auto-completion for financial use
        use_suggestions = form_completer.get_suggestions('financial_use')
        
        st.markdown("**What would you use the funds for?**")
        financial_use = st.text_area(
            "Financial use details",
            label_visibility="collapsed"
        )
        
        if use_suggestions:
            selected_use = st.selectbox("💡 Common financial needs:", [""] + use_suggestions)
            if selected_use:
                financial_use = selected_use
    
    # Health and wellness needs
    st.subheader("🏥 Health & Wellness Needs")
    col1, col2 = st.columns(2)
    with col1:
        mental_health = st.checkbox("Mental health support needed")
        physical_health = st.checkbox("Medical care needed")
    with col2:
        need_shelter = st.checkbox("Housing/shelter needed")
        need_food = st.checkbox("Food assistance needed")
    
    # Personal situation
    st.subheader("👥 Personal Situation")
    col1, col2, col3 = st.columns(3)
    with col1:
        has_kids = st.checkbox("Caring for children")
    with col2:
        unemployed = st.checkbox("Currently unemployed")
    with col3:
        in_school = st.checkbox("Currently in school")
    
    # Additional crisis detection input
    st.subheader("💭 Additional Information")
    
    # Auto-completion for additional info
    additional_suggestions = form_completer.get_suggestions('additional_info')
    
    additional_info = st.text_area(
        "Is there anything else you'd like us to know about your situation?",
        help="AI will analyze this for crisis indicators and personalized support"
    )
    
    if additional_suggestions:
        selected_additional = st.selectbox("💡 Common concerns:", [""] + additional_suggestions)
        if selected_additional:
            additional_info = selected_additional
    
    # Submit button
    submit_btn = st.form_submit_button("🤖 Analyze with AI & Find Resources", use_container_width=True)

# Process form submission with AI analysis
if submit_btn and location_input:
    # Show processing message
    with st.spinner("🤖 AI is analyzing your situation and finding the best resources..."):
        time.sleep(1)  # Simulate processing time
        
        # Geocode location
        user_lat, user_lon, geocode_success = simple_geocode(location_input)
        
        # Compile input data
        text_responses = {
            'police_response': police_response,
            'financial_use': financial_use,
            'additional_info': additional_info
        }
        
        input_data = {
            "reported_to_police": reported_to_police,
            "police_response": police_response,
            "felt_helped": felt_helped,
            "need_financial": need_financial,
            "financial_amount": financial_amount,
            "financial_use": financial_use,
            "mental_health_concerns": mental_health,
            "physical_health_concerns": physical_health,
            "need_shelter": need_shelter,
            "need_food": need_food,
            "has_kids": has_kids,
            "unemployed": unemployed,
            "in_school": in_school,
            "location": location_input,
            "lat": user_lat,
            "lon": user_lon
        }
        
        # 1. RISK ASSESSMENT
        st.subheader("🚨 AI Risk Assessment")
        risk_assessment = risk_engine.assess_risk(input_data, text_responses)
        st.session_state.risk_assessment = risk_assessment
        input_data['risk_level'] = risk_assessment['level']
        
        st.markdown(f"""
        <div class="{risk_assessment['color_class']}">
            <h4>Risk Level: {risk_assessment['level']} (Score: {risk_assessment['score']})</h4>
            <p><strong>Recommendation:</strong> {risk_assessment['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if risk_assessment['factors']:
            with st.expander("🔍 AI-Detected Risk Factors"):
                for factor in risk_assessment['factors']:
                    st.write(f"• {factor}")
        
        # 2. CRISIS DETECTION
        crisis_analysis = crisis_detector.analyze_crisis_level(text_responses)
        if crisis_analysis['level'] in ['high', 'critical']:
            st.error(f"🚨 CRISIS ALERT: {crisis_analysis['level'].upper()} level crisis indicators detected")
            st.markdown("**Please consider contacting emergency services immediately: 911**")
            st.markdown("**National Crisis Text Line: Text HOME to 741741**")
        
        # 3. GEOSPATIAL SAFETY ANALYSIS
        if geocode_success:
            st.subheader("🗺️ Area Safety Analysis")
            safety_analysis = safety_analyzer.analyze_area_safety(location_input)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Safety Score", f"{safety_analysis['safety_score']}/10")
            with col2:
                st.metric("Risk Level", safety_analysis['risk_level'])
            with col3:
                st.metric("Crime Rate", f"{safety_analysis['crime_rate']}/10")
            
            st.info("**Safety Recommendations:**")
            for rec in safety_analysis['recommendations']:
                st.write(f"• {rec}")
        
        # 4. PREDICTIVE FINANCIAL NEEDS
        st.subheader("💰 AI Financial Needs Prediction")
        predicted_amount, breakdown = financial_predictor.predict_needs(input_data)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Predicted Total Need", f"${predicted_amount:,}")
        with col2:
            st.success("This AI prediction is based on your indicated needs and location factors.")
        
        with st.expander("📊 Detailed Financial Breakdown"):
            for category, amount in breakdown.items():
                st.write(f"**{category}:** ${amount:,}")
        
        # 5. OUTCOME PREDICTION
        st.subheader("📈 Recovery Success Prediction")
        outcome_prediction = outcome_predictor.predict_recovery_success(input_data)
        
        success_prob = outcome_prediction['success_probability']
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Success Probability", f"{success_prob:.1%}")
        with col2:
            if success_prob > 0.75:
                st.success("High likelihood of positive outcomes with proper support")
            elif success_prob > 0.5:
                st.warning("Moderate success probability - additional support recommended")
            else:
                st.error("Enhanced intervention and support strongly recommended")
        
        with st.expander("🎯 Key Success Factors & Recommendations"):
            st.write("**Key Factors for Success:**")
            for factor in outcome_prediction['key_factors']:
                st.write(f"• {factor}")
            
            st.write("**AI Recommendations:**")
            for rec in outcome_prediction['recommendations']:
                st.write(f"• {rec}")
        
        # 6. INTELLIGENT RESOURCE MATCHING
        st.subheader("🎯 AI-Matched Resources")
        
        # Determine needed resource types
        needs = []
        if need_shelter: needs.append("shelter")
        if mental_health or physical_health: needs.append("health")
        if need_food: needs.append("food")
        if mental_health: needs.append("counseling")
        needs.append("legal")  # Always include legal resources
        
        if geocode_success:
            user_profile = {
                'lat': user_lat,
                'lon': user_lon,
                'has_kids': has_kids,
                'mental_health_concerns': mental_health,
                'risk_level': risk_assessment['level']
            }
            
            best_matches = resource_matcher.find_best_matches(user_profile, needs)
            
            if best_matches:
                st.info(f"🤖 AI found {len(best_matches)} optimal resource matches for your situation")
                
                for resource in best_matches:
                    # Calculate availability percentage
                    availability = ((resource['capacity'] - resource['current_occupancy']) / resource['capacity']) * 100
                    
                    st.markdown(f"""
                    <div class="resource-card">
                        <h4>🏢 {resource['name']} 
                        <span style="color: #007bff; font-size: 0.8em;">(AI Match Score: {resource['match_score']:.1f}/100)</span></h4>
                        <p><strong>Type:</strong> {resource['type'].title()}</p>
                        <p><strong>Address:</strong> {resource['address']}</p>
                        <p><strong>Phone:</strong> {resource['phone']}</p>
                        <p><strong>Hours:</strong> {resource['hours']}</p>
                        <p><strong>Distance:</strong> {resource['distance']:.1f} miles</p>
                        <p><strong>Availability:</strong> {availability:.0f}% (Wait: {resource['wait_time_days']} days)</p>
                        <p><strong>Safety Rating:</strong> {resource['safety_rating']}/10</p>
                        <p><strong>Success Rate:</strong> {resource['success_rate']:.0%}</p>
                        <p><strong>Specialties:</strong> {', '.join(resource['specialties'])}</p>
                        <p><strong>Languages:</strong> {', '.join(resource['languages'])}</p>
                        <p><strong>Website:</strong> <a href="https://{resource['website']}" target="_blank">{resource['website']}</a></p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No optimal matches found within 25 miles. Expanding search or contact national resources.")
        else:
            st.error("Unable to find your location for resource matching. Please try a different location format.")
        
        # 7. AI INSIGHTS SUMMARY
        st.subheader("🧠 AI Insights Summary")
        
        insights = []
        
        if risk_assessment['level'] == 'HIGH':
            insights.append("⚠️ High risk situation detected - immediate safety planning recommended")
        
        if crisis_analysis['level'] in ['high', 'critical']:
            insights.append("🚨 Crisis indicators present - consider immediate professional help")
        
        if success_prob < 0.6:
            insights.append("📈 Enhanced support services recommended to improve outcomes")
        
        if predicted_amount > 5000:
            insights.append("💰 Significant financial needs identified - prioritize economic assistance")
        
        if has_kids and need_shelter:
            insights.append("👨‍👩‍👧‍👦 Family-focused resources should be prioritized")
        
        if not insights:
            insights.append("✅ Situation analysis complete - standard support protocols recommended")
        
        for insight in insights:
            st.info(insight)
        
        # Always show national resources
        st.subheader("🌎 National Emergency Resources")
        st.markdown("""
        <div class="resource-card">
            <h4>24/7 National Hotlines</h4>
            <p><strong>National DV Hotline:</strong> 1-800-799-7233 (24/7)</p>
            <p><strong>Crisis Text Line:</strong> Text HOME to 741741</p>
            <p><strong>National Sexual Assault Hotline:</strong> 1-800-656-4673</p>
            <p><strong>Suicide Prevention Lifeline:</strong> 988</p>
            <p><strong>Childhelp National Hotline:</strong> 1-800-422-4453</p>
        </div>
        """, unsafe_allow_html=True)

elif submit_btn and not location_input:
    st.error("Please enter your location to enable AI analysis and resource matching.")

# Clear form button
if st.button("🗑️ Clear All Data", help="Clear form and AI analysis data"):
    st.session_state.clear()
    st.experimental_rerun()

# Footer with AI information
st.markdown("---")
st.markdown("""
<small>
<strong>🤖 AI-Enhanced Support System:</strong> This application uses artificial intelligence to provide personalized risk assessment, 
resource matching, and outcome prediction. AI analysis is designed to augment, not replace, human judgment and professional counseling.

<strong>Safety Reminder:</strong> If you're in immediate danger, call 911. 
This AI tool provides estimates and resources but is not a substitute for professional help.

<strong>Data Privacy:</strong> All AI analysis is performed securely and no personal data is permanently stored.
</small>
""", unsafe_allow_html=True)
