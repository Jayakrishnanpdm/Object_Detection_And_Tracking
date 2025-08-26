import streamlit as st
import time
import random
from datetime import datetime
import base64
import io
from PIL import Image, ImageDraw

# Page configuration
st.set_page_config(
    page_title="Smart Theft Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for the beautiful UI
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: white;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Header styles */
    .main-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .logo-icon {
        width: 50px;
        height: 50px;
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }
    
    .logo-text h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
    }
    
    .logo-text p {
        margin: 0;
        opacity: 0.7;
        font-size: 0.9rem;
    }
    
    .status-badge {
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    
    .status-active {
        background: linear-gradient(45deg, #00d4aa, #00a8ff);
    }
    
    .status-alert {
        background: linear-gradient(45deg, #ff4757, #ff3838);
        animation: alertPulse 1s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes alertPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Video section styles */
    .video-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
    }
    
    .video-placeholder {
        width: 100%;
        height: 500px;
        background: linear-gradient(45deg, #2c3e50, #3498db);
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    
    .video-overlay {
        position: absolute;
        top: 20px;
        left: 20px;
        background: rgba(0, 0, 0, 0.7);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        color: white;
        font-size: 0.9rem;
    }
    
    .video-placeholder-content {
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
    }
    
    /* Stats cards */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: white;
    }
    
    /* Alert panel */
    .alert-panel {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
    }
    
    .panel-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: white;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .alert-item {
        background: rgba(255, 71, 87, 0.1);
        border: 1px solid rgba(255, 71, 87, 0.3);
        border-left: 4px solid #ff4757;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .alert-item:hover {
        transform: translateX(5px);
        box-shadow: 0 10px 30px rgba(255, 71, 87, 0.2);
    }
    
    .alert-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .alert-title {
        font-weight: 600;
        color: #ff4757;
        font-size: 1.1rem;
    }
    
    .alert-time {
        font-size: 0.8rem;
        opacity: 0.7;
        color: white;
    }
    
    .alert-details {
        font-size: 0.9rem;
        line-height: 1.4;
        margin-bottom: 1rem;
        color: white;
    }
    
    .evidence-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .evidence-item {
        font-size: 0.8rem;
        padding: 0.5rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: white;
    }
    
    /* Objects panel */
    .object-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        margin-bottom: 1rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .object-item:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateX(5px);
    }
    
    .object-info {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .object-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }
    
    .object-safe {
        background: linear-gradient(45deg, #00d4aa, #00a8ff);
    }
    
    .object-interacted {
        background: linear-gradient(45deg, #feca57, #ff9ff3);
    }
    
    .object-missing {
        background: linear-gradient(45deg, #ff6b6b, #ff4757);
    }
    
    .object-details h4 {
        margin-bottom: 0.2rem;
        font-weight: 600;
        color: white;
        font-size: 1rem;
    }
    
    .object-status {
        font-size: 0.8rem;
        opacity: 0.7;
        color: white;
    }
    
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #00d4aa;
    }
    
    .status-warning {
        background: #feca57;
        animation: pulse 1s infinite;
    }
    
    .status-danger {
        background: #ff4757;
        animation: alertPulse 1s infinite;
    }
    
    /* Streamlit specific overrides */
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stMetric > div {
        color: white !important;
    }
    
    .stMetric label {
        color: rgba(255, 255, 255, 0.7) !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stMetric .metric-value {
        color: white !important;
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'objects_detected' not in st.session_state:
    st.session_state.objects_detected = 3
if 'persons_detected' not in st.session_state:
    st.session_state.persons_detected = 6
if 'interactions_count' not in st.session_state:
    st.session_state.interactions_count = 31
if 'alerts_count' not in st.session_state:
    st.session_state.alerts_count = 1
if 'runtime_start' not in st.session_state:
    st.session_state.runtime_start = time.time()

# Header
st.markdown("""
<div class="main-header">
    <div class="logo-section">
        <div class="logo-icon">🛡️</div>
        <div class="logo-text">
            <h1>Smart Theft Detection System</h1>
            <p>AI-Powered Security Monitoring</p>
        </div>
    </div>
    <div class="status-badge status-alert">
        🚨 Theft Detected
    </div>
</div>
""", unsafe_allow_html=True)

# Create two columns for main layout
col1, col2 = st.columns([2, 1])

with col1:
    # Video section
    st.markdown("""
    <div class="video-container">
        <div class="video-placeholder">
            <div class="video-placeholder-content">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📹</div>
                <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">Live Camera Feed</div>
                <div style="font-size: 0.9rem; opacity: 0.5;">
                    Connect your camera to see live detection
                </div>
            </div>
            <div class="video-overlay">
                <div>🎯 Detection: Active</div>
                <div>📊 FPS: 30.0</div>
                <div>⏱️ Runtime: 00:00:24</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats section
    col1_1, col1_2, col1_3, col1_4 = st.columns(4)
    
    with col1_1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{st.session_state.objects_detected}</div>
            <div class="stat-label">Objects Tracked</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col1_2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{st.session_state.persons_detected}</div>
            <div class="stat-label">Persons Detected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col1_3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{st.session_state.interactions_count}</div>
            <div class="stat-label">Interactions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col1_4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{st.session_state.alerts_count}</div>
            <div class="stat-label">Theft Alerts</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    # Recent Alerts Panel
    st.markdown("""
    <div class="alert-panel">
        <div class="panel-title">🚨 Recent Alerts</div>
        <div class="alert-item">
            <div class="alert-header">
                <div class="alert-title">THEFT DETECTED</div>
                <div class="alert-time">13:18:11</div>
            </div>
            <div class="alert-details">
                <strong>📱 Cell Phone</strong> stolen by <strong>Person #1</strong>
                <div style="margin-top: 0.5rem; color: #feca57;">
                    ⚠️ Unauthorized person detected
                </div>
            </div>
            <div class="evidence-grid">
                <div class="evidence-item">
                    <div>Confidence: <strong>88%</strong></div>
                </div>
                <div class="evidence-item">
                    <div>Duration: <strong>0.4s</strong></div>
                </div>
                <div class="evidence-item">
                    <div>Interactions: <strong>7</strong></div>
                </div>
                <div class="evidence-item">
                    <div>Score: <strong>0.17</strong></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a placeholder theft alert image
    def create_alert_image():
        img = Image.new('RGB', (400, 120), color='#2c3e50')
        draw = ImageDraw.Draw(img)
        
        # Draw bounding boxes
        draw.rectangle([50, 30, 130, 90], outline='#ff4757', width=3)
        draw.rectangle([270, 20, 330, 100], outline='#ff4757', width=3)
        
        # Add text (simplified since PIL text rendering is basic)
        draw.text((90, 10), "CELL PHONE", fill='#ff4757')
        draw.text((280, 5), "SUSPECT", fill='#ff4757')
        draw.text((150, 55), "Theft Alert Screenshot", fill='#ecf0f1')
        
        return img
    
    # Display the alert image
    alert_img = create_alert_image()
    st.image(alert_img, caption="Theft Alert Screenshot - Cell Phone", use_column_width=True)
    
    # Tracked Objects Panel
    st.markdown("""
    <div class="alert-panel">
        <div class="panel-title">📦 Tracked Objects</div>
        
        <div class="object-item">
            <div class="object-info">
                <div class="object-icon object-missing">📱</div>
                <div class="object-details">
                    <h4>Cell Phone #18</h4>
                    <div class="object-status">STOLEN - Missing for 60+ frames</div>
                </div>
            </div>
            <div class="status-indicator status-danger"></div>
        </div>

        <div class="object-item">
            <div class="object-info">
                <div class="object-icon object-interacted">💻</div>
                <div class="object-details">
                    <h4>Laptop #5</h4>
                    <div class="object-status">TOUCHED - 31 interactions</div>
                </div>
            </div>
            <div class="status-indicator status-warning"></div>
        </div>

        <div class="object-item">
            <div class="object-info">
                <div class="object-icon object-safe">🍾</div>
                <div class="object-details">
                    <h4>Bottle #8</h4>
                    <div class="object-status">SAFE - No interactions</div>
                </div>
            </div>
            <div class="status-indicator"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Auto-refresh functionality
if st.button("🔄 Refresh Data", key="refresh_btn"):
    # Simulate some random updates
    st.session_state.interactions_count += random.randint(1, 3)
    st.session_state.persons_detected = max(1, st.session_state.persons_detected + random.choice([-1, 0, 1]))
    st.rerun()

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)

# JavaScript for real-time updates (if needed)
st.markdown("""
<script>
// Function to add new alerts (can be called from Python)
function addNewAlert(alertData) {
    console.log('New theft alert:', alertData);
    // Implementation for adding new alerts dynamically
}

// Real-time updates simulation
setInterval(function() {
    // Update FPS and other real-time data
    const fpsElement = document.querySelector('.video-overlay');
    if (fpsElement) {
        // Update FPS display
    }
}, 1000);

console.log('🚀 Streamlit Theft Detection UI Loaded!');
</script>
""", unsafe_allow_html=True)

# Footer with additional controls
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    if st.button("🚨 Trigger Test Alert"):
        st.session_state.alerts_count += 1
        st.success("Test alert triggered!")
        st.rerun()

with col_f2:
    if st.button("📊 Reset Statistics"):
        st.session_state.objects_detected = 0
        st.session_state.persons_detected = 0
        st.session_state.interactions_count = 0
        st.session_state.alerts_count = 0
        st.success("Statistics reset!")
        st.rerun()

with col_f3:
    if st.button("💾 Export Report"):
        st.download_button(
            label="📄 Download Report",
            data=f"""Smart Theft Detection Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Statistics:
- Objects Tracked: {st.session_state.objects_detected}
- Persons Detected: {st.session_state.persons_detected}
- Interactions: {st.session_state.interactions_count}
- Theft Alerts: {st.session_state.alerts_count}

Status: {'Alert' if st.session_state.alerts_count > 0 else 'Normal'}
""",
            file_name=f"theft_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )