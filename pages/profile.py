import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.auth import require_auth, init_session_state, logout_user
from database.db_manager import DatabaseManager
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Profile",
    page_icon="👤",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .profile-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
        margin: 0;
    }
    
    .stat-label {
        color: #666;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    .activity-item {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem;
        color: white;
    }
    
    .badge-beginner { background: #95a5a6; }
    .badge-intermediate { background: #3498db; }
    .badge-expert { background: #e74c3c; }
    .badge-master { background: #f39c12; }
</style>
""", unsafe_allow_html=True)

# Initialize
init_session_state()

@require_auth
def main():
    st.title("👤 User Profile & Dashboard")
    
    db = DatabaseManager()
    user = st.session_state.user
    stats = db.get_user_stats(st.session_state.user_id)
    
    # Profile header
    st.markdown(f"""
        <div class="profile-header">
            <h1>👤 {user['username']}</h1>
            <p style="font-size: 1.2rem; margin: 0;">{user['email']}</p>
            <p style="margin-top: 1rem; opacity: 0.9;">
                Member since: {user['created_at'][:10]}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🏆 Achievements", "📈 Activity", "⚙️ Settings"])
    
    with tab1:
        st.markdown("### 📊 Your Statistics Overview")
        
        # Main stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
                <div class="stat-card">
                    <p class="stat-number">{}</p>
                    <p class="stat-label">News Classifications</p>
                </div>
            """.format(stats['total_classifications']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="stat-card">
                    <p class="stat-number">{}</p>
                    <p class="stat-label">Fake News Detections</p>
                </div>
            """.format(stats['total_detections']), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="stat-card">
                    <p class="stat-number">{}</p>
                    <p class="stat-label">Articles Generated</p>
                </div>
            """.format(stats['total_generated']), unsafe_allow_html=True)
        
        with col4:
            total_actions = stats['total_classifications'] + stats['total_detections'] + stats['total_generated']
            st.markdown("""
                <div class="stat-card">
                    <p class="stat-number">{}</p>
                    <p class="stat-label">Total Actions</p>
                </div>
            """.format(total_actions), unsafe_allow_html=True)
        
        # Category distribution
        if stats['category_distribution']:
            st.markdown("### 📈 News Category Distribution")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=list(stats['category_distribution'].keys()),
                        values=list(stats['category_distribution'].values()),
                        marker_colors=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'],
                        hole=0.4
                    )
                ])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Category Breakdown:**")
                for category, count in stats['category_distribution'].items():
                    percentage = (count / stats['total_classifications']) * 100
                    st.markdown(f"- **{category}**: {count} ({percentage:.1f}%)")
        
        # Recent activity summary
        st.markdown("### 📅 Recent Activity")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Latest Classifications**")
            recent_class = db.get_user_classifications(st.session_state.user_id, limit=5)
            if recent_class:
                for item in recent_class:
                    st.markdown(f"""
                        <div class="activity-item">
                            <strong>{item['predicted_category']}</strong><br>
                            <small>{item['created_at']}</small>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No classifications yet")
        
        with col2:
            st.markdown("**Latest Detections**")
            recent_detect = db.get_user_detections(st.session_state.user_id, limit=5)
            if recent_detect:
                for item in recent_detect:
                    st.markdown(f"""
                        <div class="activity-item">
                            <strong>{item['verdict']}</strong><br>
                            <small>{item['created_at']}</small>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No detections yet")
        
        with col3:
            st.markdown("**Latest Generated Articles**")
            recent_gen = db.get_user_generated_news(st.session_state.user_id, limit=5)
            if recent_gen:
                for item in recent_gen:
                    st.markdown(f"""
                        <div class="activity-item">
                            <strong>{item['generated_title'][:30]}...</strong><br>
                            <small>{item['created_at']}</small>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No generated articles yet")
    
    with tab2:
        st.markdown("### 🏆 Achievements & Badges")
        
        # Calculate achievements
        total_actions = stats['total_classifications'] + stats['total_detections'] + stats['total_generated']
        
        # User level
        if total_actions >= 100:
            level = "Master"
            badge_class = "badge-master"
            progress = 100
        elif total_actions >= 50:
            level = "Expert"
            badge_class = "badge-expert"
            progress = (total_actions / 100) * 100
        elif total_actions >= 20:
            level = "Intermediate"
            badge_class = "badge-intermediate"
            progress = (total_actions / 50) * 100
        else:
            level = "Beginner"
            badge_class = "badge-beginner"
            progress = (total_actions / 20) * 100
        
        # Display level
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
                <div style="text-align: center;">
                    <h2>Your Level</h2>
                    <span class="badge {badge_class}">{level}</span>
                    <p style="margin-top: 1rem;">Total Actions: {total_actions}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Progress to Next Level")
            st.progress(progress / 100)
            
            if level == "Beginner":
                st.info(f"Complete {20 - total_actions} more actions to reach Intermediate level")
            elif level == "Intermediate":
                st.info(f"Complete {50 - total_actions} more actions to reach Expert level")
            elif level == "Expert":
                st.info(f"Complete {100 - total_actions} more actions to reach Master level")
            else:
                st.success("🎉 Congratulations! You've reached the Master level!")
        
        # Achievements
        st.markdown("### 🎖️ Unlocked Achievements")
        
        achievements = []
        
        if stats['total_classifications'] >= 10:
            achievements.append(("📰 News Classifier", "Classified 10+ articles"))
        if stats['total_detections'] >= 10:
            achievements.append(("🔍 Fact Checker", "Analyzed 10+ articles for fake news"))
        if stats['total_generated'] >= 10:
            achievements.append(("✍️ Content Creator", "Generated 10+ articles"))
        if total_actions >= 50:
            achievements.append(("⭐ Power User", "Completed 50+ total actions"))
        if stats['total_classifications'] >= 1 and stats['total_detections'] >= 1 and stats['total_generated'] >= 1:
            achievements.append(("🎯 All-Rounder", "Used all features"))
        
        if achievements:
            cols = st.columns(3)
            for i, (title, desc) in enumerate(achievements):
                with cols[i % 3]:
                    st.success(f"**{title}**\n\n{desc}")
        else:
            st.info("Complete actions to unlock achievements!")
    
    with tab3:
        st.markdown("### 📈 Activity Timeline")
        
        # Get all recent activities
        all_activities = []
        
        # Add classifications
        for item in db.get_user_classifications(st.session_state.user_id, limit=10):
            all_activities.append({
                'type': 'Classification',
                'icon': '🏷️',
                'description': f"Classified article as {item['predicted_category']}",
                'time': item['created_at']
            })
        
        # Add detections
        for item in db.get_user_detections(st.session_state.user_id, limit=10):
            all_activities.append({
                'type': 'Detection',
                'icon': '🔍',
                'description': f"Analyzed article - Verdict: {item['verdict']}",
                'time': item['created_at']
            })
        
        # Add generated news
        for item in db.get_user_generated_news(st.session_state.user_id, limit=10):
            all_activities.append({
                'type': 'Generation',
                'icon': '✍️',
                'description': f"Generated article: {item['generated_title'][:40]}...",
                'time': item['created_at']
            })
        
        # Sort by time
        all_activities.sort(key=lambda x: x['time'], reverse=True)
        
        if all_activities:
            for activity in all_activities[:20]:
                st.markdown(f"""
                    <div class="activity-item">
                        <strong>{activity['icon']} {activity['type']}</strong><br>
                        {activity['description']}<br>
                        <small style="color: #999;">{activity['time']}</small>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No activity yet. Start using the features to see your timeline!")
    
    with tab4:
        st.markdown("### ⚙️ Account Settings")
        
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
            
            # Theme preference
            theme = st.selectbox(
                "Theme (Coming Soon)",
                ["Light", "Dark", "Auto"],
                disabled=True
            )
            
            # Notifications
            notifications = st.checkbox("Enable Notifications (Coming Soon)", disabled=True)
        
        st.markdown("---")
        
        # Account actions
        st.markdown("#### 🔧 Account Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚪 Logout", use_container_width=True):
                logout_user()
                st.success("Logged out successfully!")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.warning("This feature is coming soon!")
        
        # Danger zone
        with st.expander("⚠️ Danger Zone", expanded=False):
            st.error("**Delete Account**")
            st.markdown("This will permanently delete your account and all associated data.")
            
            if st.button("Delete My Account", type="primary"):
                st.warning("Account deletion feature coming soon. Contact support for assistance.")

if __name__ == "__main__":
    main()