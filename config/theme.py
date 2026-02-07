"""
Professional theme configuration with mobile responsiveness and accessibility
"""

import streamlit as st


def apply_theme():
    """Apply professional, accessible, mobile-responsive theme"""
    st.markdown("""
<style>
    /* ============================================================
       FONTS & GLOBAL STYLES
       ============================================================ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ============================================================
       MAIN CONTAINER - RESPONSIVE
       ============================================================ */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .block-container {
        padding: 2rem 1rem;
        background: #ffffff;
        border-radius: 16px;
        margin: 1rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        max-width: 1400px;
    }
    
    /* ============================================================
       SIDEBAR - FIXED TO NOT BREAK TOGGLE
       ============================================================ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    /* Only style content inside sidebar, NOT the sidebar itself */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* DO NOT hide or modify sidebar controls */
    [data-testid="stSidebar"] [data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
    }
    
    /* ============================================================
       TYPOGRAPHY - IMPROVED CONTRAST
       ============================================================ */
    h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
        font-weight: 700;
        line-height: 1.2;
    }
    
    h1 {
        font-size: clamp(1.75rem, 4vw, 2.5rem);
        margin-bottom: 1rem;
    }
    
    h2 {
        font-size: clamp(1.5rem, 3.5vw, 2rem);
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        font-size: clamp(1.25rem, 3vw, 1.5rem);
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        color: #334155 !important;
    }
    
    p, span, div, label {
        color: #475569;
        line-height: 1.6;
    }
    
    .stMarkdown {
        color: #475569;
    }
    
    /* Strong contrast for readability */
    strong, b {
        color: #1e293b;
        font-weight: 600;
    }
    
    /* ============================================================
       HEADER STYLES
       ============================================================ */
    .main-header {
        color: #667eea;
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: clamp(1rem, 2.5vw, 1.2rem);
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: clamp(1.5rem, 3vw, 2rem);
        font-weight: 700;
        color: #1e293b !important;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #667eea;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* ============================================================
       BUTTONS - PROFESSIONAL & ACCESSIBLE
       ============================================================ */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        cursor: pointer;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5568d3 0%, #6941a3 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    .stButton > button:focus {
        outline: 2px solid #667eea;
        outline-offset: 2px;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
    }
    
    /* ============================================================
       METRIC CARDS - RESPONSIVE
       ============================================================ */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        height: 100%;
        min-height: 140px;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        border-color: #667eea;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: clamp(1.75rem, 4vw, 2.5rem);
        font-weight: 800;
        color: #667eea;
        margin: 0.25rem 0;
    }
    
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* ============================================================
       RECOMMENDATION BOXES - RESPONSIVE
       ============================================================ */
    .recommendation-box,
    .critical-box,
    .safe-box {
        padding: clamp(1.5rem, 3vw, 2.5rem);
        border-radius: 16px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .critical-box {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        animation: gentle-pulse 2s ease-in-out infinite;
    }
    
    .safe-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    @keyframes gentle-pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.01); }
    }
    
    .recommendation-title {
        font-size: clamp(1.1rem, 2.5vw, 1.5rem);
        font-weight: 700;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .recommendation-value {
        font-size: clamp(2.5rem, 6vw, 4.5rem);
        font-weight: 900;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .recommendation-details {
        font-size: clamp(0.95rem, 2vw, 1.1rem);
        line-height: 1.8;
        opacity: 0.95;
    }
    
    /* ============================================================
       INSIGHT CARDS - IMPROVED CONTRAST
       ============================================================ */
    .insight-card {
        background: #ffffff;
        border-left: 4px solid #667eea;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        border: 1px solid #e2e8f0;
    }
    
    .insight-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        border-left-color: #5568d3;
    }
    
    .insight-card strong {
        color: #1e293b;
    }
    
    .insight-icon {
        font-size: 1.4rem;
        margin-right: 0.5rem;
        vertical-align: middle;
    }
    
    /* ============================================================
       STATUS BADGES - HIGH CONTRAST
       ============================================================ */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 0.9rem;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.25rem;
    }
    
    .badge-success {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #10b981;
    }
    
    .badge-warning {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #f59e0b;
    }
    
    .badge-danger {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #ef4444;
    }
    
    .badge-info {
        background: #dbeafe;
        color: #1e40af;
        border: 1px solid #3b82f6;
    }
    
    /* ============================================================
       CHARTS & VISUALIZATIONS
       ============================================================ */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin: 1.5rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .chart-title {
        font-size: clamp(1.25rem, 3vw, 1.75rem);
        font-weight: 700;
        color: #1e293b !important;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* ============================================================
       DATA TABLES - IMPROVED READABILITY
       ============================================================ */
    .dataframe {
        border-radius: 8px !important;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .dataframe th {
        background: #f8fafc !important;
        color: #1e293b !important;
        font-weight: 600 !important;
        padding: 0.75rem !important;
    }
    
    .dataframe td {
        color: #475569 !important;
        padding: 0.6rem !important;
    }
    
    /* ============================================================
       FORM ELEMENTS - SIDEBAR & MAIN
       ============================================================ */
    .stSelectbox label,
    .stSlider label,
    .stNumberInput label,
    .stTextInput label,
    .stRadio label {
        font-weight: 600 !important;
        color: #334155 !important;
        font-size: 0.9rem !important;
    }
    
    /* ============================================================
       EXPANDER - IMPROVED VISIBILITY
       ============================================================ */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1rem;
        border: 1px solid #e2e8f0;
        color: #1e293b !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: #f1f5f9;
        border-color: #667eea;
    }
    
    .streamlit-expanderHeader p {
        color: #1e293b !important;
        font-weight: 600;
    }
    
    /* ============================================================
       ALERTS & INFO BOXES
       ============================================================ */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
        padding: 1rem;
    }
    
    .stAlert p {
        color: #1e293b !important;
        font-weight: 500;
    }
    
    /* ============================================================
       DIVIDERS
       ============================================================ */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* ============================================================
       FEATURE TAG
       ============================================================ */
    .feature-tag {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    /* ============================================================
       ANIMATIONS
       ============================================================ */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in {
        animation: slideIn 0.6s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* ============================================================
       MOBILE RESPONSIVENESS
       ============================================================ */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.75rem;
            margin: 0.5rem;
            border-radius: 12px;
        }
        
        .main-header {
            font-size: 2rem;
            margin-bottom: 0.75rem;
        }
        
        .sub-header {
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }
        
        .section-header {
            font-size: 1.4rem;
            margin-top: 1.5rem;
        }
        
        .metric-card {
            padding: 1.25rem;
            min-height: 120px;
        }
        
        .metric-value {
            font-size: 1.75rem;
        }
        
        .recommendation-box,
        .critical-box,
        .safe-box {
            padding: 1.5rem;
            margin: 1.5rem 0;
        }
        
        .recommendation-value {
            font-size: 2.5rem;
        }
        
        .recommendation-details {
            font-size: 0.95rem;
        }
        
        .chart-container {
            padding: 1rem;
        }
        
        .insight-card {
            padding: 1rem;
        }
        
        .stButton > button {
            padding: 0.65rem 1.5rem;
            font-size: 0.9rem;
        }
        
        /* Stack columns on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }
    
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.75rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
        
        .recommendation-value {
            font-size: 2rem;
        }
        
        h2 {
            font-size: 1.3rem;
        }
        
        h3 {
            font-size: 1.1rem;
        }
    }
    
    /* ============================================================
       ACCESSIBILITY
       ============================================================ */
    /* Focus states for keyboard navigation */
    button:focus,
    input:focus,
    select:focus {
        outline: 2px solid #667eea;
        outline-offset: 2px;
    }
    
    /* Skip link for screen readers */
    .skip-link {
        position: absolute;
        top: -40px;
        left: 0;
        background: #667eea;
        color: white;
        padding: 8px;
        text-decoration: none;
        z-index: 100;
    }
    
    .skip-link:focus {
        top: 0;
    }

</style>
""", unsafe_allow_html=True)
