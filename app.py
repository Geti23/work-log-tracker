import streamlit as st
import pandas as pd
import datetime
from datetime import timedelta
import textwrap
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- Configuration ---
SHEET_NAME = "Work Logs"
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# --- Google Sheets Backend Functions ---
@st.cache_resource
def get_client():
    """
    Establishes the connection to Google Sheets.
    Uses @st.cache_resource because this is a connection object that 
    should be created once and shared across the session.
    """
    # Load credentials from Streamlit Secrets
    creds_dict = st.secrets["gcp_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
    return gspread.authorize(creds)

@st.cache_data(ttl=600)  # Cache this data for 10 minutes (or until cleared)
def get_data():
    """
    Fetches data from Google Sheets and caches it in memory.
    The app will use this cached copy for searching/navigation 
    instead of calling Google every time.
    """
    client = get_client()
    try:
        sheet = client.open(SHEET_NAME).sheet1
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        if df.empty:
            return pd.DataFrame(columns=["date", "ticket_id", "description", "time_spent", "id"])

        # Ensure columns exist
        expected_cols = ["date", "ticket_id", "description", "time_spent"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ""

        # Standardize Date Format
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        if 'id' not in df.columns:
             df['id'] = df.index
             
        return df
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Error: Could not find Google Sheet named '{SHEET_NAME}'.")
        return pd.DataFrame()

def add_entry(date, ticket, desc, time):
    """
    Adds a new row to Google Sheets and clears the cache 
    so the new entry appears immediately.
    """
    client = get_client()
    sheet = client.open(SHEET_NAME).sheet1
    date_str = date.strftime("%Y-%m-%d")
    
    # 1. Write to Google (This will still take 1-2 seconds)
    sheet.append_row([date_str, ticket, desc, time])
    
    # 2. IMPORTANT: Clear the cache!
    # This forces get_data() to re-fetch from Google the next time it runs.
    get_data.clear()

def get_logs(search_term=None, start_date=None, end_date=None):
    """Filters and sorts the data from the cached DataFrame."""
    df = get_data()
    
    if df.empty:
        return df

    # Search Filter
    if search_term:
        mask = (df['ticket_id'].astype(str).str.contains(search_term, case=False, na=False)) | \
               (df['description'].astype(str).str.contains(search_term, case=False, na=False)) | \
               (df['date'].astype(str).str.contains(search_term, case=False, na=False))
        df = df[mask]

    # Date Range Filter
    if start_date and end_date:
        s_date = start_date.strftime("%Y-%m-%d")
        e_date = end_date.strftime("%Y-%m-%d")
        df = df[(df['date'] >= s_date) & (df['date'] <= e_date)]

    # Sorting Logic: Date DESC, ID ASC
    df = df.sort_values(by=['date', 'id'], ascending=[False, True])
    
    return df

# --- Helper Functions ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def parse_time_str(time_str):
    if not time_str:
        return 0
    time_str = str(time_str).lower().strip()
    total_minutes = 0
    
    decimal_h = re.match(r'^(\d+(\.\d+)?)h$', time_str)
    if decimal_h:
        return int(float(decimal_h.group(1)) * 60)

    hours = re.search(r'(\d+)h', time_str)
    minutes = re.search(r'(\d+)m', time_str)
    
    if hours:
        total_minutes += int(hours.group(1)) * 60
    if minutes:
        total_minutes += int(minutes.group(1))
        
    return total_minutes

def format_minutes(total_minutes):
    if total_minutes == 0:
        return ""
    h = total_minutes // 60
    m = total_minutes % 60
    if h > 0 and m > 0:
        return f"{h}h {m}m"
    elif h > 0:
        return f"{h}h"
    return f"{m}m"

# --- UI Layout & CSS ---
st.set_page_config(page_title="Work Log Tracker", layout="wide", page_icon="📅")

st.markdown("""
<style>
    /* 1. Main Background */
    /* We rely on Streamlit's default theme for the main background */

    /* 2. Calendar Card Style */
    .day-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 6px;
        padding: 10px;
        height: 75vh;
        display: flex;
        flex-direction: column;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .day-header {
        font-family: 'Courier New', Courier, monospace;
        text-align: center;
        border-bottom: 1px solid rgba(49, 51, 63, 0.2);
        padding-bottom: 5px;
        margin-bottom: 10px;
        color: var(--text-color);
        flex-shrink: 0;
    }
    
    .day-name {
        font-weight: bold;
        font-size: 1.1em;
        text-transform: uppercase;
        color: var(--text-color);
        opacity: 0.6;
    }
    
    .day-date {
        font-size: 1.5em;
        font-weight: bold;
        color: var(--text-color);
    }
    
    /* Scrollable middle section */
    .tickets-container {
        flex-grow: 1;
        overflow-y: auto;
        margin-bottom: 10px;
    }

    .ticket-entry {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.85em;
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid rgba(49, 51, 63, 0.1);
        color: var(--text-color);
    }
    
    /* Royal Blue Ticket ID */
    .ticket-id {
        color: #4169e1; 
        font-weight: bold;
    }
    
    /* Safe Green Time */
    .time-spent {
        color: #2ea043; 
        float: right;
        font-weight: bold;
    }

    /* Footer for Total Hours */
    .day-footer {
        border-top: 1px solid rgba(49, 51, 63, 0.2);
        padding-top: 8px;
        text-align: right;
        font-family: 'Consolas', 'Courier New', monospace;
        font-weight: bold;
        color: var(--text-color);
        font-size: 0.9em;
        flex-shrink: 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Main App Logic ---

# Sidebar
with st.sidebar:
    st.title("📝 New Entry")
    with st.form("add_log_form", clear_on_submit=True):
        entry_date = st.date_input("Date", datetime.date.today())
        entry_ticket = st.text_input("Ticket ID")
        entry_desc = st.text_area("Description")
        entry_time = st.text_input("Time Spent (e.g., 2h 30m)")
        
        if st.form_submit_button("Add Log"):
            if entry_desc:
                add_entry(entry_date, entry_ticket, entry_desc, entry_time)
                st.success("Saved to Google Sheets!")
                st.rerun()
            else:
                st.error("Description is required.")

    st.divider()
    
    # --- NEW REFRESH BUTTON ---
    if st.button("🔄 Refresh Data", help="Click this if you edited the Google Sheet manually"):
        get_data.clear()
        st.rerun()
    # --------------------------

    st.subheader("🔍 Search")
    search_query = st.text_input("Search (ID, Desc, Date)")
    
    st.divider()
    st.subheader("💾 Backup")
    # Fetch all data for export (using cache if available)
    all_data = get_logs() 
    if not all_data.empty:
        csv = convert_df_to_csv(all_data)
        st.download_button("Download All Data (CSV)", csv, 'work_logs_backup.csv', 'text/csv')

# Main Content Area
st.title("Work Log Calendar")

if search_query:
    st.subheader(f"Results for '{search_query}'")
    results = get_logs(search_term=search_query)
    if not results.empty:
        for _, row in results.iterrows():
            st.markdown(textwrap.dedent(f"""
            <div style="background-color: var(--secondary-background-color); padding: 10px; margin-bottom: 5px; border-radius: 5px; border: 1px solid rgba(49, 51, 63, 0.2);">
                <span style="color: #8b949e; font-weight:bold;">{row['date']}</span> | 
                <span style="color: #4169e1; font-weight:bold;">{row['ticket_id']}</span> 
                <span style="color: var(--text-color);">{row['description']}</span>
                <span style="color: #2ea043; float:right; font-weight:bold;">{row['time_spent']}</span>
            </div>
            """), unsafe_allow_html=True)
    else:
        st.info("No logs found.")
else:
    # Navigation State
    if 'week_offset' not in st.session_state:
        st.session_state.week_offset = 0

    col_prev, col_current, col_next, _ = st.columns([1, 2, 1, 6])
    with col_prev:
        if st.button("← Prev Week"):
            st.session_state.week_offset += 1 
    with col_next:
        if st.button("Next Week →"):
            st.session_state.week_offset -= 1

    # Date Calculations
    today = datetime.date.today()
    start_of_week = today - timedelta(days=today.weekday() + 1) - timedelta(weeks=st.session_state.week_offset)
    end_of_week = start_of_week + timedelta(days=6)
    
    with col_current:
        st.write(f"**Week of {start_of_week.strftime('%b %d')}**")

    # Fetch weekly data
    weekly_data = get_logs(start_date=start_of_week, end_date=end_of_week)
    
    # 7-Column Grid
    cols = st.columns(7)
    days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

    for i, col in enumerate(cols):
        current_day_date = start_of_week + timedelta(days=i)
        date_str = current_day_date.strftime("%Y-%m-%d")
        
        # --- Logic: Today / Tomorrow / Yesterday labels ---
        day_label = days[i]
        if current_day_date == today:
            day_label = "Today"
        elif current_day_date == today + timedelta(days=1):
            day_label = "Tomorrow"
        elif current_day_date == today - timedelta(days=1):
            day_label = "Yesterday"
        # --------------------------------------------------

        # Filter logs for this specific day
        day_logs = weekly_data[weekly_data['date'] == date_str]
        
        # Calculate daily total minutes
        day_total_minutes = 0
        
        # Build Card HTML
        html_content = textwrap.dedent(f"""
            <div class="day-card">
                <div class="day-header">
                    <div class="day-name">{day_label}</div>
                    <div class="day-date">{current_day_date.day}</div>
                </div>
                <div class="tickets-container">
        """)
        
        for _, row in day_logs.iterrows():
            t_id = row['ticket_id'] if row['ticket_id'] else ""
            t_time = row['time_spent'] if row['time_spent'] else ""
            desc = str(row['description']).replace(t_id, "").strip()
            
            day_total_minutes += parse_time_str(t_time)
            
            html_content += textwrap.dedent(f"""
                <div class="ticket-entry">
                    <span class="ticket-id">{t_id}</span>
                    {desc}
                    <span class="time-spent">{t_time}</span>
                </div>
            """)
        
        html_content += "</div>" # Close tickets-container
        
        # Add Footer with GREEN Total
        total_display = format_minutes(day_total_minutes)
        if total_display:
            html_content += textwrap.dedent(f"""
                <div class="day-footer">
                    Total: <span style="color: #2ea043">{total_display}</span>
                </div>
            """)
        
        html_content += "</div>" # Close day-card
        
        with col:
            st.markdown(html_content, unsafe_allow_html=True)