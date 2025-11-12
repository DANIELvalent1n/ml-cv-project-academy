import streamlit as st
import sys
from pathlib import Path
import config
from datetime import datetime
import plotly.graph_objects as go

# Path setup
sys.path.append(str(Path(__file__).parent.parent))
from utils.auth import require_auth, init_session_state
from models.login import LoginModel
from database.db_manager import DatabaseManager

# Page config
st.set_page_config(
    page_title="Profile",
    page_icon="👤",
    layout="wide"
)

# === Custom CSS ===
st.markdown("""
<style>
    body {
        color: #e6e6e6;
        background-color: #0e0e10;
    }

    /* Global text color */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #e6e6e6 !important;
    }

    /* Header section */
    .profile-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #e6e6e6;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 5px 25px rgba(0,0,0,0.2);
    }

    /* Statistic cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(6px);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }

    .stat-number {
        font-size: 2.3rem;
        font-weight: 700;
        color: #8b9cff;
        margin-bottom: 0.5rem;
    }

    .stat-label {
        color: #cfcfcf;
        font-size: 1rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        color: white;
        margin: 0.5rem;
        font-size: 0.9rem;
    }

    .badge-beginner { background: #95a5a6; }
    .badge-intermediate { background: #3498db; }
    .badge-expert { background: #e74c3c; }
    .badge-master { background: #f39c12; }

    /* Activity item */
    .activity-item {
        background: rgba(255, 255, 255, 0.04);
        border-left: 4px solid #667eea;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.75rem 0;
        transition: background 0.3s ease;
    }

    .activity-item:hover {
        background: rgba(255, 255, 255, 0.08);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #ccc;
        font-weight: 600;
        font-size: 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #fff;
        border-bottom: 3px solid #8b9cff;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: #fff;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }

    /* Info/alert boxes */
    .stAlert {
        color: #e6e6e6 !important;
    }
</style>
""", unsafe_allow_html=True)

# === Initialize ===
init_session_state()

def main():
    if not st.session_state.logged_in:
        LoginModel.login()
        return

    st.title("👤 User Profile & Dashboard")

    db = DatabaseManager()
    user = st.session_state.user
    stats = db.get_user_stats(st.session_state.user_id)

    # === Header ===
    st.markdown(f"""
        <div class="profile-header">
            <h1>{user['username']}</h1>
            <p style="font-size:1.1rem;">{user['email']}</p>
            <p style="opacity: 0.85;">Member since: {user['created_at'][:10]}</p>
        </div>
    """, unsafe_allow_html=True)

    # === Tabs ===
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🏆 Achievements", "📈 Activity", "⚙️ Settings"])

    # === Dashboard ===
    with tab1:
        st.subheader("📊 Your Statistics Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="stat-card">
                    <p class="stat-number">{stats['total_classifications']}</p>
                    <p class="stat-label">News Classifications</p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class="stat-card">
                    <p class="stat-number">{stats['total_generated']}</p>
                    <p class="stat-label">Articles Generated</p>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div class="stat-card">
                    <p class="stat-number">{stats['total_ai_detections']}</p>
                    <p class="stat-label">AI Text Detections</p>
                </div>
            """, unsafe_allow_html=True)

        with col4:
            total_actions = stats['total_classifications'] + stats['total_detections'] + stats['total_generated'] + stats['total_ai_detections']
            st.markdown(f"""
                <div class="stat-card">
                    <p class="stat-number">{total_actions}</p>
                    <p class="stat-label">Total Actions</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📈 News Category Distribution")

        if stats['category_distribution']:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = go.Figure(data=[go.Pie(
                    labels=list(stats['category_distribution'].keys()),
                    values=list(stats['category_distribution'].values()),
                    hole=0.4,
                    marker_colors=['#8b9cff', '#00bfa6', '#ff6b6b', '#a66bff']
                )])
                fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', font_color='#e6e6e6')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Category Breakdown:**")
                for cat, count in stats['category_distribution'].items():
                    pct = (count / stats['total_classifications']) * 100
                    st.markdown(f"- **{cat}**: {count} ({pct:.1f}%)")

    # === Achievements ===
    with tab2:
        st.subheader("🏆 Achievements & Badges")

        total_actions = stats['total_classifications'] + stats['total_detections'] + stats['total_generated']
        if total_actions >= 100:
            level, badge_class, progress = "Master", "badge-master", 100
        elif total_actions >= 50:
            level, badge_class, progress = "Expert", "badge-expert", total_actions
        elif total_actions >= 20:
            level, badge_class, progress = "Intermediate", "badge-intermediate", total_actions / 0.5
        else:
            level, badge_class, progress = "Beginner", "badge-beginner", total_actions / 0.2

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
                <div style="text-align:center;">
                    <h3>Your Level</h3>
                    <span class="badge {badge_class}">{level}</span>
                    <p>Total Actions: {total_actions}</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("### Progress to Next Level")
            st.progress(min(progress / 100, 1))
            if level == "Beginner":
                st.info(f"Complete {20 - total_actions} more actions to reach Intermediate level.")
            elif level == "Intermediate":
                st.info(f"Complete {50 - total_actions} more actions to reach Expert level.")
            elif level == "Expert":
                st.info(f"Complete {100 - total_actions} more actions to reach Master level.")
            else:
                st.success("🎉 You've reached the Master level!")

    # === Activity ===
    with tab3:
        st.subheader("📈 Activity Timeline")
        db = DatabaseManager()
        all_activities = []

        for item in db.get_user_classifications(st.session_state.user_id, limit=10):
            all_activities.append({'icon': '🏷️', 'type': 'Classification', 'desc': f"Classified article as {item['predicted_category']}", 'time': item['created_at']})
        for item in db.get_user_detections(st.session_state.user_id, limit=10):
            all_activities.append({'icon': '🔍', 'type': 'Detection', 'desc': f"Analyzed article - Verdict: {item['verdict']}", 'time': item['created_at']})
        for item in db.get_user_generated_news(st.session_state.user_id, limit=10):
            all_activities.append({'icon': '✍️', 'type': 'Generation', 'desc': f"Generated article: {item['generated_title'][:40]}...", 'time': item['created_at']})
        for item in db.get_user_origin(st.session_state.user_id, limit=10):
            all_activities.append({'icon': '🤖', 'type': 'AI Detection', 'desc': f"Analyzed text - Verdict: {item['predicted_origin']}", 'time': item['created_at']})

        all_activities.sort(key=lambda x: x['time'], reverse=True)
        if all_activities:
            for a in all_activities[:20]:
                st.markdown(f"""
                    <div class="activity-item">
                        <strong>{a['icon']} {a['type']}</strong><br>
                        {a['desc']}<br>
                        <small style="color:#999;">{a['time']}</small>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No activity yet. Start using the app to build your timeline!")

    # === Settings ===
    with tab4:
        st.subheader("⚙️ Account Settings")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 👤 Account Information")
            st.info(f"""
            **Username:** {user['username']}  
            **Email:** {user['email']}  
            **Member Since:** {user['created_at'][:10]}  
            **Last Login:** {user['last_login'][:16] if user['last_login'] else 'N/A'}  
            """)
        with col2:
            st.markdown("#### 🎨 Preferences")
            st.selectbox("Theme (Coming Soon)", ["Light", "Dark", "Auto"], disabled=True)
            st.checkbox("Enable Notifications (Coming Soon)", disabled=True)
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.success("Logged out successfully!")
            st.rerun()
        # Danger zone
        with st.expander("⚠️ Danger Zone", expanded=False):
            st.error("**Delete Account**")
            st.markdown("This will permanently delete your account and all associated data.")
            
            if st.button("Delete My Account", type="primary"):
                db.delete_user(st.session_state.user_id)
                st.session_state.logged_in = False
                st.success("✅ Your account has been deleted successfully.")

    

if __name__ == "__main__":
    main()
