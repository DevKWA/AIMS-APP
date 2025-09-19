import streamlit as st
import pandas as pd
import gspread

st.set_page_config(layout="wide")

# ---------------------------
# Global CSS for white background and black text
# ---------------------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: white;
        }

        html, body, [class*="css"] {
            color: black !important;
        }

        /* Expander full width */
        .streamlit-expander {
            width: 100% !important;
        }

        /* DataFrame text color and background */
        .stDataFrame div {
            color: black !important;
            background-color: white !important;
        }

        .stDataFrame th {
            color: black !important;
            background-color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Header image/banner
# ---------------------------
st.image('assets/AULA_HORIZONTAL_GREEN_BANNER.png', use_container_width=True)
st.image('assets/survivor_dashboard_banner.png', use_container_width=True)

# ---------------------------
# Google Sheets loader
# ---------------------------
@st.cache_data(ttl=600)
def load_sheet(sheet_url, worksheet_name=None):
    gc = gspread.service_account(filename="/workspaces/AIMS-APP/practical-lodge-341703-f87bbdac7e18.json")
    sh = gc.open_by_url(sheet_url)
    ws = sh.worksheet(worksheet_name) if worksheet_name else sh.sheet1
    data = ws.get_all_records()
    return pd.DataFrame(data)

# ---------------------------
# Sidebar filters
# ---------------------------
sheet_url = st.sidebar.text_input("📄 Google Sheet URL (public or private)")
worksheet_name = None
df = None

if sheet_url:
    try:
        gc = gspread.service_account(filename="/workspaces/AIMS-APP/practical-lodge-341703-f87bbdac7e18.json")
        sh = gc.open_by_url(sheet_url)
        worksheet_names = [ws.title for ws in sh.worksheets()]
        worksheet_name = st.sidebar.selectbox("📑 Choose a worksheet/tab", worksheet_names)
        df = load_sheet(sheet_url, worksheet_name)
        df = df.drop_duplicates()
        st.sidebar.success(f"✅ Loaded worksheet: {worksheet_name}")
    except Exception as ex:
        st.sidebar.error(f"❌ Error loading sheet: {ex}")

if df is None or df.empty:
    st.warning("Please enter a valid Google Sheet URL and choose a worksheet.")
    st.stop()

# ---------------------------
# Filter options
# ---------------------------
regions = sorted(df['REGION'].dropna().unique())
industries = sorted(df['INDUSTRY'].dropna().unique())
categories = sorted(df['CATEGORY'].dropna().unique())

st.sidebar.header("🔍 Filters")
selected_region = st.sidebar.selectbox("🌍 Select Region", options=["All"] + regions)
selected_industry = st.sidebar.selectbox("🏭 Select Industry", options=["All"] + industries)
selected_category = st.sidebar.selectbox("📂 Select Category", options=["All"] + categories)

# ---------------------------
# Filter logic
# ---------------------------
invalid_filter = False
filtered_df = df.copy()

if selected_region != "All":
    if selected_region in df['REGION'].values:
        filtered_df = filtered_df[filtered_df['REGION'] == selected_region]
    else:
        invalid_filter = True

if selected_industry != "All":
    if selected_industry in df['INDUSTRY'].values:
        filtered_df = filtered_df[filtered_df['INDUSTRY'] == selected_industry]
    else:
        invalid_filter = True

if selected_category != "All":
    if selected_category in df['CATEGORY'].values:
        filtered_df = filtered_df[filtered_df['CATEGORY'] == selected_category]
    else:
        invalid_filter = True

if invalid_filter or filtered_df.empty:
    st.warning("❌ One or more filters are invalid — no results to display.")
    st.stop()

# ---------------------------
# Showing results header
# ---------------------------
st.markdown(
    f"""
    <div style="padding:10px; border-radius:8px; background-color:#ffffff; border-left:6px solid #000000;">
        <h3 style="color:black; text-align:left;">📊 Showing Results: Region: {selected_region} | Industry: {selected_industry} | Category: {selected_category}</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Emojis/icons for rows
# ---------------------------
region_icons = {"Africa": "🌍", "Europe": "🌎", "Asia": "🌏"}
industry_icons = {"Oil & Gas": "🏭", "Agriculture": "🌾", "Technology": "💻"}
category_icons = {"Labor": "👷", "Human Rights": "✊", "Safety": "🛡️"}

for _, row in filtered_df.iterrows():
    region_icon = region_icons.get(row.get('REGION', ''), "🌐")
    industry_icon = industry_icons.get(row.get('INDUSTRY', ''), "🏢")
    category_icon = category_icons.get(row.get('CATEGORY', ''), "📂")

    st.markdown(
        f"""
        <div style="padding:15px; border-radius:12px; background-color:#ffffff; margin-bottom:10px; border:1px solid #e0e0e0;">
            <strong>{region_icon} {industry_icon} {category_icon} {row.get('TITLE','Untitled')}</strong><br>
            {row.get('DESCRIPTION','')}<br>
            {"🌐 <a href='" + row['link'] + "' target='_blank'>Read More</a>" if 'link' in row and row['link'] else ""}
       
        """,
        unsafe_allow_html=True
    )
