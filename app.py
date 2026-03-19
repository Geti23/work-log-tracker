import streamlit as st
import pandas as pd
import datetime
from datetime import timedelta
import textwrap
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import shutil

# --- RENDER DEPLOYMENT FIX ---
render_secret_path = "/etc/secrets/secrets.toml"
streamlit_secret_path = ".streamlit/secrets.toml"

if os.path.exists(render_secret_path):
    os.makedirs(".streamlit", exist_ok=True)
    try:
        shutil.copy(render_secret_path, streamlit_secret_path)
    except Exception as e:
        pass 
# -----------------------------

# --- Configuration ---
SHEET_NAME = "Work Logs"
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# --- LOGIN SYSTEM ---
def check_password():
    def password_entered():
        if st.session_state["username"] in st.secrets["passwords"] and \
           st.session_state["password"] == st.secrets["passwords"][st.session_state["username"]]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.title("🔒 Login Required")
    st.text_input("Username", key="username")
    st.text_input("Password", type="password", on_change=password_entered, key="password")
    
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 User not known or password incorrect")
    return False

# --- Google Sheets Backend Functions ---
@st.cache_resource
def get_client():
    creds_dict = st.secrets["gcp_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
    return gspread.authorize(creds)

@st.cache_data(ttl=600)
def get_data():
    client = get_client()
    try:
        sheet = client.open(SHEET_NAME).sheet1
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        if df.empty:
            return pd.DataFrame(columns=["date", "ticket_id", "description", "time_spent", "id"])

        expected_cols = ["date", "ticket_id", "description", "time_spent"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ""

        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        if 'id' not in df.columns:
             df['id'] = df.index
             
        return df
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Error: Could not find Google Sheet named '{SHEET_NAME}'.")
        return pd.DataFrame()

def add_entry(date, ticket, desc, time):
    client = get_client()
    sheet = client.open(SHEET_NAME).sheet1
    date_str = date.strftime("%Y-%m-%d")
    sheet.append_row([date_str, ticket, desc, time])
    get_data.clear()

def update_entry(df_index, date, ticket, desc, time):
    client = get_client()
    sheet = client.open(SHEET_NAME).sheet1
    
    sheet_row = int(df_index) + 2 
    date_str = date.strftime("%Y-%m-%d")
    
    sheet.update(f"A{sheet_row}:D{sheet_row}", [[date_str, ticket, desc, time]])
    get_data.clear()

def delete_entry(df_index):
    client = get_client()
    sheet = client.open(SHEET_NAME).sheet1
    
    sheet_row = int(df_index) + 2 
    sheet.delete_rows(sheet_row)
    get_data.clear()

def get_logs(search_term=None, start_date=None, end_date=None):
    df = get_data()
    if df.empty: return df

    if search_term:
        mask = (df['ticket_id'].astype(str).str.contains(search_term, case=False, na=False)) | \
               (df['description'].astype(str).str.contains(search_term, case=False, na=False)) | \
               (df['date'].astype(str).str.contains(search_term, case=False, na=False))
        df = df[mask]

    if start_date and end_date:
        s_date = start_date.strftime("%Y-%m-%d")
        e_date = end_date.strftime("%Y-%m-%d")
        df = df[(df['date'] >= s_date) & (df['date'] <= e_date)]

    df = df.sort_values(by=['date', 'id'], ascending=[False, True])
    return df

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def parse_time_str(time_str):
    if not time_str: return 0
    time_str = str(time_str).lower().strip()
    total_minutes = 0
    decimal_h = re.match(r'^(\d+(\.\d+)?)h$', time_str)
    if decimal_h: return int(float(decimal_h.group(1)) * 60)
    hours = re.search(r'(\d+)h', time_str)
    minutes = re.search(r'(\d+)m', time_str)
    if hours: total_minutes += int(hours.group(1)) * 60
    if minutes: total_minutes += int(minutes.group(1))
    return total_minutes

def format_minutes(total_minutes):
    if total_minutes == 0: return ""
    h = total_minutes // 60
    m = total_minutes % 60
    if h > 0 and m > 0: return f"{h}h {m}m"
    elif h > 0: return f"{h}h"
    return f"{m}m"

# --- MAIN EXECUTION ---
if check_password():

    st.set_page_config(page_title="Work Log Tracker", layout="wide", page_icon="📅")

    st.markdown("""
    <style>
        /* SCROLLING WRAPPER */
        .week-container {
            display: flex;
            overflow-x: auto;
            gap: 12px;
            padding-bottom: 15px;
            
            /* CENTERING MAGIC */
            margin: 0 auto;       
            width: fit-content;   
            max-width: 100%;      
        }
        
        .day-card {
            background-color: var(--secondary-background-color);
            
            /* DYNAMIC BORDER FIX: SAFE NEUTRAL GRAY */
            /* Visible on both White and Black backgrounds */
            border: 1px solid rgba(140, 140, 140, 0.35);
            
            border-radius: 6px;
            padding: 10px;
            height: 75vh;
            
            /* WIDTH SETTINGS */
            min-width: 250px; 
            flex: 0 0 250px;
            
            display: flex;
            flex-direction: column;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .day-header {
            font-family: 'Courier New', Courier, monospace;
            text-align: center;
            
            /* DYNAMIC BORDER FIX */
            border-bottom: 1px solid rgba(140, 140, 140, 0.35);
            
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
        .tickets-container {
            flex-grow: 1;
            overflow-y: auto;
            margin-bottom: 10px;
        }
        .ticket-entry {
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 0.85em;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(140, 140, 140, 0.5);
            color: var(--text-color);
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .ticket-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        .ticket-id {
            color: #4169e1; 
            font-weight: bold;
            background-color: rgba(65, 105, 225, 0.1);
            padding: 2px 6px;
            border-radius: 4px;
        }
        .time-spent {
            color: #2ea043; 
            font-weight: bold;
            background-color: rgba(46, 160, 67, 0.1);
            padding: 2px 6px;
            border-radius: 4px;
            white-space: nowrap;
        }
        .ticket-desc {
            line-height: 1.4;
            background-color: rgba(140, 140, 140, 0.1);
            padding: 6px 8px;
            border-radius: 4px;
            margin-top: 2px;
        }
        .day-footer {
            /* DYNAMIC BORDER FIX */
            border-top: 1px solid rgba(140, 140, 140, 0.5);
            
            padding-top: 8px;
            text-align: right;
            font-family: 'Consolas', 'Courier New', monospace;
            font-weight: bold;
            color: var(--text-color);
            font-size: 0.9em;
            flex-shrink: 0;
        }
        
        /* Custom Scrollbar */
        .week-container::-webkit-scrollbar {
            height: 8px;
        }
        .week-container::-webkit-scrollbar-track {
            background: rgba(140, 140, 140, 0.1);
            border-radius: 4px;
        }
        .week-container::-webkit-scrollbar-thumb {
            background: rgba(140, 140, 140, 0.3);
            border-radius: 4px;
        }
        .week-container::-webkit-scrollbar-thumb:hover {
            background: rgba(140, 140, 140, 0.5);
        }
    </style>
    """, unsafe_allow_html=True)

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
        if st.button("🔄 Refresh Data", help="Click if you edited Google Sheet manually"):
            get_data.clear()
            st.rerun()

        st.divider()
        st.subheader("✏️ Edit / Delete Entry")
        
        all_logs = get_logs()
        if not all_logs.empty:
            options_map = {
                f"ID {row['id']}: {row['date']} | {row['ticket_id']} ({row['time_spent']})": row['id'] 
                for _, row in all_logs.iterrows()
            }
            
            selected_label = st.selectbox("Select entry to modify", ["-- Select --"] + list(options_map.keys()))
            
            if selected_label != "-- Select --":
                selected_id = options_map[selected_label]
                selected_row = all_logs[all_logs['id'] == selected_id].iloc[0]
                
                with st.form("edit_delete_form"):
                    edit_date = pd.to_datetime(selected_row['date']).date()
                    new_date = st.date_input("Date", edit_date)
                    new_ticket = st.text_input("Ticket ID", str(selected_row['ticket_id']))
                    new_desc = st.text_area("Description", str(selected_row['description']))
                    new_time = st.text_input("Time Spent", str(selected_row['time_spent']))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("💾 Update"):
                            if new_desc:
                                update_entry(selected_id, new_date, new_ticket, new_desc, new_time)
                                st.success("Entry updated!")
                                st.rerun()
                            else:
                                st.error("Description is required.")
                    with col2:
                        if st.form_submit_button("🗑️ Delete"):
                            delete_entry(selected_id)
                            st.success("Entry deleted!")
                            st.rerun()

        st.divider()
        st.subheader("🔍 Search")
        search_query = st.text_input("Search (ID, Desc, Date)")
        
        st.divider()
        st.subheader("💾 Backup")
        all_data = get_logs() 
        if not all_data.empty:
            csv = convert_df_to_csv(all_data)
            st.download_button("Download All Data (CSV)", csv, 'work_logs_backup.csv', 'text/csv')

    # Main Content Area
    st.markdown("<h1 style='text-align: center;'>Work Log Calendar</h1>", unsafe_allow_html=True)

    if search_query:
        st.subheader(f"Results for '{search_query}'")
        results = get_logs(search_term=search_query)
        if not results.empty:
            for _, row in results.iterrows():
                st.markdown(textwrap.dedent(f"""
                <div style="background-color: var(--secondary-background-color); padding: 10px; margin-bottom: 5px; border-radius: 5px; border: 1px solid rgba(140, 140, 140, 0.35);">
                    <span style="color: #8b949e; font-weight:bold;">{row['date']}</span> | 
                    <span style="color: #4169e1; font-weight:bold;">{row['ticket_id']}</span> 
                    <span style="color: var(--text-color);">{row['description']}</span>
                    <span style="color: #2ea043; float:right; font-weight:bold;">{row['time_spent']}</span>
                </div>
                """), unsafe_allow_html=True)
        else:
            st.info("No logs found.")
    else:
        if 'week_offset' not in st.session_state:
            st.session_state.week_offset = 0

        # --- CENTERED NAVIGATION ---
        _, col_prev, col_text, col_next, _ = st.columns([5, 1, 2, 1, 5])
        
        with col_prev:
            if st.button("←", use_container_width=True):
                st.session_state.week_offset += 1 
        with col_next:
            if st.button("→", use_container_width=True):
                st.session_state.week_offset -= 1

        today = datetime.date.today()
        start_of_week = today - timedelta(days=today.weekday() + 1) - timedelta(weeks=st.session_state.week_offset)
        end_of_week = start_of_week + timedelta(days=6)
        
        with col_text:
            st.markdown(
                f"<div style='text-align: center; font-weight: bold; padding-top: 10px; white-space: nowrap;'>"
                f"Week of {start_of_week.strftime('%b %d')}"
                f"</div>", 
                unsafe_allow_html=True
            )

        weekly_data = get_logs(start_date=start_of_week, end_date=end_of_week)
        days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

        week_html = '<div class="week-container">'
        
        for i in range(7):
            current_day_date = start_of_week + timedelta(days=i)
            date_str = current_day_date.strftime("%Y-%m-%d")
            
            day_label = days[i]
            if current_day_date == today:
                day_label = "<span style='color: #FFDF00; font-weight: bold; background-color: rgba(255, 223, 0, 0.1); padding: 2px 6px; border-radius: 4px; white-space: nowrap;'>Today</span><br>" + days[i]
        #    elif current_day_date == today + timedelta(days=1):
        #        day_label = "Tomorrow"
        #    elif current_day_date == today - timedelta(days=1):
        #        day_label = "Yesterday"

            day_logs = weekly_data[weekly_data['date'] == date_str]
            day_total_minutes = 0
            
            week_html += textwrap.dedent(f"""
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
                
                week_html += textwrap.dedent(f"""
                    <div class="ticket-entry">
                        <div class="ticket-header">
                            <span class="ticket-id">{t_id}</span>
                            <span class="time-spent">{t_time}</span>
                        </div>
                        <div class="ticket-desc">
                            {desc}
                        </div>
                    </div>
                """)
            
            week_html += "</div>"
            total_display = format_minutes(day_total_minutes)
            if total_display:
                week_html += textwrap.dedent(f"""
                    <div class="day-footer">
                        Total: <span style="color: #2ea043; font-weight: bold; background-color: rgba(46, 160, 67, 0.1); padding: 2px 6px; border-radius: 4px; white-space: nowrap;">{total_display}</span>
                    </div>
                """)
            
            week_html += "</div>"

        week_html += '</div>'
        st.markdown(week_html, unsafe_allow_html=True)