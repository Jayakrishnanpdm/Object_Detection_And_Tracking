"""
Alert Module for Theft Detection System
Handles email and Telegram notifications
"""

import cv2
import smtplib
import os
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
import requests
import json
from typing import Optional, Dict, Any
import base64
from io import BytesIO
import numpy as np

logger = logging.getLogger(__name__)


class AlertConfig:
    """Configuration for alert system"""
    # Email settings
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_address: str = "jay.work.developer@gmail.com"  # Replace with your email
    email_password: str = "anje vvtv dyiw oglg"  # Use app-specific password
    hr_email: str = "jayakrishnan9446718962@gmail.com"  # HR email to receive alerts
    
    # Telegram settings
    telegram_bot_token: str = "8442626184:AAFpFGCJx7iHT1pFBKEKhK5aUAusEHn-zks"  # Replace with bot token
    telegram_chat_id: str = "1348434893"  # Replace with chat ID
    
    # Alert settings
    save_alert_images: bool = True
    alert_image_dir: str = "alert_images"
    include_video_clip: bool = False
    video_clip_duration: int = 10  # seconds


class AlertManager:
    """Manages all alert operations"""
    
    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        self.setup_directories()
        self.email_configured = self.check_email_config()
        self.telegram_configured = self.check_telegram_config()
        
    def setup_directories(self):
        """Create necessary directories"""
        if self.config.save_alert_images:
            os.makedirs(self.config.alert_image_dir, exist_ok=True)
            
    def check_email_config(self) -> bool:
        """Check if email is properly configured"""
        return (self.config.email_address != "your_email@gmail.com" and 
                self.config.email_password != "your_app_password" and
                self.config.hr_email != "hr@company.com")
    
    def check_telegram_config(self) -> bool:
        """Check if Telegram is properly configured"""
        return (self.config.telegram_bot_token != "8442626184:AAFpFGCJx7iHT1pFBKEKhK5aUAusEHn-zks" and
                self.config.telegram_chat_id != "1348434893")
    
    def create_alert_frame(self, frame: np.ndarray, object_info: Dict, 
                          person_info: Dict, evidence: Dict) -> np.ndarray:
        """Create annotated frame for alert"""
        alert_frame = frame.copy()
        h, w = alert_frame.shape[:2]
        
        # Draw theft detection header
        cv2.rectangle(alert_frame, (0, 0), (w, 60), (0, 0, 255), -1)
        cv2.putText(alert_frame, "THEFT DETECTED - UNAUTHORIZED ACCESS", 
                   (w//2 - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Highlight suspect
        if 'bbox' in person_info:
            x1, y1, x2, y2 = map(int, person_info['bbox'])
            cv2.rectangle(alert_frame, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 3)
            cv2.putText(alert_frame, f"SUSPECT: Person #{person_info.get('id', 'Unknown')}", 
                       (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Highlight stolen object location (last known)
        if 'last_bbox' in object_info:
            ox1, oy1, ox2, oy2 = map(int, object_info['last_bbox'])
            cv2.rectangle(alert_frame, (ox1-3, oy1-3), (ox2+3, oy2+3), (0, 255, 255), 3)
            cv2.putText(alert_frame, f"STOLEN: {object_info.get('name', 'Unknown Object')}", 
                       (ox1, oy1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add evidence panel
        panel_y = h - 120
        cv2.rectangle(alert_frame, (0, panel_y), (w, h), (0, 0, 0), -1)
        cv2.rectangle(alert_frame, (0, panel_y), (w, h), (0, 0, 255), 2)
        
        # Evidence text
        cv2.putText(alert_frame, f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                   (10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(alert_frame, f"Object: {object_info.get('name', 'Unknown')} | ID: {object_info.get('id', 'N/A')}", 
                   (10, panel_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(alert_frame, f"Interactions: {evidence.get('interaction_count', 0)} | "
                   f"Confidence: {evidence.get('max_confidence', 0):.2%}", 
                   (10, panel_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(alert_frame, f"Suspicion Score: {evidence.get('suspicion_score', 0):.2f}", 
                   (10, panel_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return alert_frame
    
    def save_alert_image(self, frame: np.ndarray, object_name: str, person_id: int) -> str:
        """Save alert image to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"theft_alert_{timestamp}_{object_name}_{person_id}.jpg"
        filepath = os.path.join(self.config.alert_image_dir, filename)
        
        cv2.imwrite(filepath, frame)
        logger.info(f"Alert image saved: {filepath}")
        return filepath
    
    def send_email_alert(self, image_path: str, object_info: Dict, 
                        person_info: Dict, evidence: Dict) -> bool:
        """Send email alert to HR"""
        if not self.email_configured:
            logger.warning("Email not configured, skipping email alert")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🚨 THEFT ALERT: {object_info.get('name', 'Unknown Object')} - Unauthorized Access"
            msg['From'] = self.config.email_address
            msg['To'] = self.config.hr_email
            
            # Create HTML content
            html_content = self.create_email_html(object_info, person_info, evidence)
            
            # Attach HTML
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Attach image
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-ID', '<alert_image>')
                    img.add_header('Content-Disposition', 'inline', filename=os.path.basename(image_path))
                    msg.attach(img)
            
            # Send email
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.email_address, self.config.email_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent successfully to {self.config.hr_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def create_email_html(self, object_info: Dict, person_info: Dict, evidence: Dict) -> str:
        """Create HTML content for email"""
        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .alert-header {{ 
                    background-color: #ff4444; 
                    color: white; 
                    padding: 20px; 
                    text-align: center;
                }}
                .content {{ padding: 20px; }}
                .evidence-table {{ 
                    border-collapse: collapse; 
                    width: 100%;
                    margin-top: 20px;
                }}
                .evidence-table th, .evidence-table td {{ 
                    border: 1px solid #ddd; 
                    padding: 12px; 
                    text-align: left;
                }}
                .evidence-table th {{ 
                    background-color: #f2f2f2; 
                }}
                .urgent {{ 
                    color: #ff4444; 
                    font-weight: bold;
                }}
                .image-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="alert-header">
                <h1>🚨 THEFT DETECTION ALERT 🚨</h1>
                <p>Unauthorized person detected taking valuable object</p>
            </div>
            
            <div class="content">
                <h2 class="urgent">Immediate Action Required</h2>
                
                <p><strong>Time of Incident:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h3>Stolen Object Details</h3>
                <table class="evidence-table">
                    <tr>
                        <th>Property</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Object Type</td>
                        <td>{object_info.get('name', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td>Object ID</td>
                        <td>{object_info.get('id', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Last Seen</td>
                        <td>{object_info.get('last_seen', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td>Status</td>
                        <td class="urgent">MISSING - LIKELY STOLEN</td>
                    </tr>
                </table>
                
                <h3>Suspect Information</h3>
                <table class="evidence-table">
                    <tr>
                        <th>Property</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Person ID</td>
                        <td>{person_info.get('id', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td>Authorization Status</td>
                        <td class="urgent">UNAUTHORIZED</td>
                    </tr>
                    <tr>
                        <td>Interactions with Object</td>
                        <td>{evidence.get('interaction_count', 0)}</td>
                    </tr>
                    <tr>
                        <td>Interaction Confidence</td>
                        <td>{evidence.get('max_confidence', 0):.2%}</td>
                    </tr>
                    <tr>
                        <td>Suspicion Score</td>
                        <td>{evidence.get('suspicion_score', 0):.2f}</td>
                    </tr>
                </table>
                
                <h3>Evidence Image</h3>
                <div class="image-container">
                    <img src="cid:alert_image" alt="Theft Detection Image" style="max-width: 100%; height: auto;">
                </div>
                
                <h3>Recommended Actions</h3>
                <ol>
                    <li>Review security footage immediately</li>
                    <li>Check if the object is still in the premises</li>
                    <li>Identify and locate the suspect</li>
                    <li>Contact security if person is still on premises</li>
                    <li>File incident report</li>
                </ol>
                
                <hr>
                <p style="color: #666; font-size: 12px;">
                    This is an automated alert from the Theft Detection System. 
                    Please verify the incident before taking action.
                </p>
            </div>
        </body>
        </html>
        """
    
    def send_telegram_alert(self, image_path: str, object_info: Dict, 
                           person_info: Dict, evidence: Dict) -> bool:
        """Send Telegram alert"""
        if not self.telegram_configured:
            logger.warning("Telegram not configured, skipping Telegram alert")
            return False
        
        try:
            # Prepare message text
            message = f"""
🚨 *THEFT ALERT* 🚨

*Object:* {object_info.get('name', 'Unknown')}
*Object ID:* {object_info.get('id', 'N/A')}
*Status:* MISSING - LIKELY STOLEN

*Suspect:* Person #{person_info.get('id', 'Unknown')}
*Authorization:* UNAUTHORIZED

*Evidence:*
• Interactions: {evidence.get('interaction_count', 0)}
• Confidence: {evidence.get('max_confidence', 0):.2%}
• Suspicion Score: {evidence.get('suspicion_score', 0):.2f}

*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️ *Immediate action required!*
            """
            
            # Send photo with caption
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendPhoto"
            
            with open(image_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': self.config.telegram_chat_id,
                    'caption': message,
                    'parse_mode': 'Markdown'
                }
                
                response = requests.post(url, files=files, data=data)
                
                if response.status_code == 200:
                    logger.info("Telegram alert sent successfully")
                    return True
                else:
                    logger.error(f"Telegram API error: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
    
    def send_alert(self, frame: np.ndarray, object_info: Dict, 
                  person_info: Dict, evidence: Dict) -> Dict[str, bool]:
        """Send all configured alerts"""
        results = {'email': False, 'telegram': False, 'image_saved': False}
        
        # Create alert frame
        alert_frame = self.create_alert_frame(frame, object_info, person_info, evidence)
        
        # Save image
        if self.config.save_alert_images:
            image_path = self.save_alert_image(
                alert_frame, 
                object_info.get('name', 'unknown'),
                person_info.get('id', 0)
            )
            results['image_saved'] = True
        else:
            # Save temporary image for sending
            image_path = "temp_alert.jpg"
            cv2.imwrite(image_path, alert_frame)
        
        # Send email alert
        if self.email_configured:
            results['email'] = self.send_email_alert(image_path, object_info, person_info, evidence)
        
        # Send Telegram alert
        if self.telegram_configured:
            results['telegram'] = self.send_telegram_alert(image_path, object_info, person_info, evidence)
        
        # Clean up temporary image
        if not self.config.save_alert_images and os.path.exists(image_path):
            os.remove(image_path)
        
        logger.info(f"Alert results: {results}")
        return results


# Integration function for main system
def setup_alert_system(email_address: str = None, email_password: str = None,
                      hr_email: str = None, telegram_token: str = None,
                      telegram_chat_id: str = None) -> AlertManager:
    """
    Setup alert system with provided credentials
    
    Args:
        email_address: Sender email address (use Gmail)
        email_password: App-specific password for Gmail
        hr_email: HR email to receive alerts
        telegram_token: Telegram bot token
        telegram_chat_id: Telegram chat ID
    
    Returns:
        Configured AlertManager instance
    """
    config = AlertConfig()
    
    if email_address:
        config.email_address = email_address
    if email_password:
        config.email_password = email_password
    if hr_email:
        config.hr_email = hr_email
    if telegram_token:
        config.telegram_bot_token = telegram_token
    if telegram_chat_id:
        config.telegram_chat_id = telegram_chat_id
    
    return AlertManager(config)


# Example usage
if __name__ == "__main__":
    # Example configuration
    alert_manager = setup_alert_system(
        email_address="your_email@gmail.com",
        email_password="your_app_password",
        hr_email="hr@company.com",
        telegram_token="YOUR_BOT_TOKEN",
        telegram_chat_id="YOUR_CHAT_ID"
    )
    
    # Test data
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    test_object = {
        'name': 'laptop',
        'id': 1,
        'last_bbox': [100, 100, 300, 200],
        'last_seen': datetime.now().strftime('%H:%M:%S')
    }
    test_person = {
        'id': 5,
        'bbox': [400, 200, 550, 500]
    }
    test_evidence = {
        'interaction_count': 3,
        'max_confidence': 0.85,
        'suspicion_score': 7.5
    }
    
    # Send test alert
    # results = alert_manager.send_alert(test_frame, test_object, test_person, test_evidence)
    # print(f"Alert sent: {results}")