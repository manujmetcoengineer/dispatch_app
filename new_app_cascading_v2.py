import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  # Keep this import
from groq import Groq
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import os
import re
import difflib

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Truck Data AI Dashboard", layout="wide")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ Missing GROQ_API_KEY in .env file. Please add it and restart.")
    st.stop()

# Initialize Groq client
try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"❌ Failed to initialize Groq client: {e}")
    st.stop()

# --- NEW: Define our vibrant color palette ---
VIVID_COLORS = px.colors.qualitative.Vivid
# -------------------------------------------

# ---------------- UTILITIES ----------------
def best_match_key(d: dict, targets, cutoff=0.6):
    """
    Return the key from dict d (case-insensitive) that best matches any of the target names.
    """
    if not d:
        return None
    keys = list(d.keys())
    lower_map = {k.lower(): k for k in keys}
    candidates = list(lower_map.keys())
    for t in targets:
        # exact (case-insensitive)
        if t.lower() in lower_map:
            return lower_map[t.lower()]
        # close match
        m = difflib.get_close_matches(t.lower(), candidates, n=1, cutoff=cutoff)
        if m:
            return lower_map[m[0]]
    return None

def try_parse_date_series(s: pd.Series, date_col_name=""):
    """
    Try multiple strategies to parse dates in a series with enhanced month/year parsing.
    Also normalizes two-digit years like 'Jan-25' -> 'Jan-2025' before parsing.
    """
    s = s.astype(str).str.strip()
    
    # Replace common placeholders with NA and drop
    s = s.replace(['nan', 'None', '', 'NaN', 'NaT', 'nat'], pd.NA)
    s = s.dropna()
    
    if len(s) == 0:
        return pd.Series(dtype='datetime64[ns]')

    # If looks like a range, take the first date-like token
    def first_date_from_range(x):
        if pd.isna(x):
            return x
        x = str(x)
        if " to " in x.lower():
            x = x.split(" to ")[0].strip()
        elif " - " in x and len(x.split(" - ")) == 2:
            parts = x.split(" - ")
            # If second part contains month name, keep as-is (not a range)
            if any(word in parts[1].lower() for word in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                                         'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                return x
            x = parts[0].strip()
        return x

    s1 = s.apply(first_date_from_range)

    # Normalize two-digit years like "Jan-25" -> "Jan-2025" (assume 20xx for 00-99)
    def normalize_two_digit_year(x):
        if pd.isna(x):
            return x
        x = str(x).strip()
        # Patterns like Jan-25, Jan 25, Jan/25, Jan_25
        m = re.match(r"^([A-Za-z]{3,9})[\s\-\_/](\d{2})$", x)
        if m:
            month_part = m.group(1)
            yy = int(m.group(2))
            yyyy = 2000 + yy
            return f"{month_part} {yyyy}"
        # Patterns like 01-25 or 1-25 -> ambiguous; prefer treating as month-two-digit-year if second part length=2
        m2 = re.match(r"^(\d{1,2})[\s\-\_/](\d{2})$", x)
        if m2:
            mm = int(m2.group(1))
            yy = int(m2.group(2))
            if 1 <= mm <= 12:
                yyyy = 2000 + yy
                return f"{yyyy}-{mm:02d}-01"
        return x

    s2 = s1.apply(normalize_two_digit_year)

    # Try direct parsing first, dayfirst True (handles many formats)
    dt = pd.to_datetime(s2, errors="coerce", dayfirst=True)

    # If many NaT, attempt additional strategies
    if dt.isna().mean() > 0.3:
        # Attempt common ISO-like fixes first
        s3 = s2.str.replace(r"^(\d{4})-(\d{1,2})$", r"\1-\2-01", regex=True)
        s3 = s3.str.replace(r"^(\d{1,2})-(\d{4})$", r"\2-\1-01", regex=True)
        s3 = s3.str.replace(r"^(\d{1,2})/(\d{4})$", r"\2-\1-01", regex=True)
        s3 = s3.str.replace(r"^(\d{4})/(\d{1,2})$", r"\1-\2-01", regex=True)
        
        # Reordered map to check for long names FIRST
        month_map = {
            'january': '01', 'jan': '01',
            'february': '02', 'feb': '02',
            'march': '03', 'mar': '03',
            'april': '04', 'apr': '04',
            'may': '05',
            'june': '06', 'jun': '06',
            'july': '07', 'jul': '07',
            'august': '08', 'aug': '08',
            'september': '09', 'sep': '09',
            'october': '10', 'oct': '10',
            'november': '11', 'nov': '11',
            'december': '12', 'dec': '12'
        }
        
        def month_name_to_iso(x):
            if pd.isna(x):
                return x
            x = str(x).strip()
            # Accept patterns: "Jan-2025", "January 2025", "2025 Jan", "Jan 25" (normalized earlier)
            patterns = [
                r"^([A-Za-z]+)[\s\-_/](\d{4})$",
                r"^(\d{4})[\s\-_/]([A-Za-z]+)$"
            ]
            for pattern in patterns:
                m = re.match(pattern, x)
                if m:
                    part1, part2 = m.group(1), m.group(2)
                    if part1.isdigit():
                        year, month_str = part1, part2
                    else:
                        month_str, year = part1, part2
                    month_lower = month_str.lower()
                    mm = None
                    # This loop now correctly checks long names first
                    for month_name, month_num in month_map.items():
                        if month_lower.startswith(month_name):
                            mm = month_num
                            break
                    if mm:
                        return f"{year}-{mm}-01"
            return x

        s4 = s3.apply(month_name_to_iso)
        dt = pd.to_datetime(s4, errors="coerce")

    return dt

def choose_date_column(df: pd.DataFrame):
    """Choose the most likely date column from the dataframe."""
    candidates = ["Date Range", "Month Range", "Date", "Month", "Week", "Day", "Period"]
    cols = [c for c in df.columns]
    
    for c in candidates:
        for col in cols:
            if col.lower() == c.lower():
                return col
    
    for c in candidates:
        for col in cols:
            if c.lower() in col.lower():
                return col
    
    if len(cols) > 0:
        return cols[0]
    
    return None

# ---------------- HELPERS ----------------
@st.cache_data
def load_csv(file):
    """Load CSV file and return DataFrame."""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

@st.cache_data
def load_excel(file):
    """Load all sheets from Excel into lowercase-named DataFrames."""
    try:
        xls = pd.ExcelFile(file)
        sheets = {name: xls.parse(name) for name in xls.sheet_names}
        normalized = {name.lower(): df for name, df in sheets.items()}
        return normalized
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return {}

# --- ⭐️ UPDATED melt_data FUNCTION ⭐️ ---
# This function is now ONLY responsible for melting a single sheet
# It correctly handles a 'Date' column OR 'Year'/'Month' columns
def melt_data(df: pd.DataFrame, sheet_type=""):
    """
    Convert pivoted Excel sheet into Date | Entity | Distance (Miles) format with robust parsing.
    Handles 'Date | Entity_1 | ...' format
    Handles 'Year | Month | Entity_1 | ...' format
    """
    if df is None or df.empty:
        return pd.DataFrame() # Return empty df

    df_processed = df.copy()
    df_processed.columns = [str(c).strip() if c is not None else "Unnamed" for c in df_processed.columns]
    
    df_processed = df_processed.dropna(how='all')
    if df_processed.empty:
        return pd.DataFrame()

    cols_lower = [c.lower() for c in df_processed.columns]
    has_year = 'year' in cols_lower
    has_month = 'month' in cols_lower

    id_vars = []
    value_cols = []
    has_parsed_date = False

    # --- LOGIC for Year/Month columns ---
    if has_year and has_month:
        diag_expander = st.session_state.get('diag_expander')
        if diag_expander:
            with diag_expander:
                st.info(f"ℹ️ Found 'Year' and 'Month' columns in '{sheet_type}' data. Using them as ID vars.")
        
        # Get actual column names
        year_col_name = df_processed.columns[cols_lower.index('year')]
        month_col_name = df_processed.columns[cols_lower.index('month')]
        
        id_vars = [year_col_name, month_col_name]
        
        # Add 'Week' if it also exists
        if 'week' in cols_lower:
             week_col_name = df_processed.columns[cols_lower.index('week')]
             id_vars.append(week_col_name)
             
        value_cols = [c for c in df_processed.columns if c not in id_vars and c not in ['Unnamed']]
        
        melted = df_processed.melt(id_vars=id_vars, value_vars=value_cols, var_name="Entity", value_name="Distance (Miles)")
        
        melted = melted.rename(columns={year_col_name: 'Year', month_col_name: 'Month'})
        
        melted["Entity"] = melted["Entity"].astype(str).replace({"nan": "Unknown", "None": "Unknown"}).fillna("Unknown")
        melted["Distance (Miles)"] = pd.to_numeric(melted["Distance (Miles)"], errors="coerce").fillna(0)
        
        # ⭐️ ADDED: Create Date column for consistency
        try:
            melted['Date'] = pd.to_datetime(melted['Month'].astype(str) + " " + melted['Year'].astype(str), errors='coerce')
            has_parsed_date = True
        except Exception as e:
            if diag_expander:
                with diag_expander:
                    st.warning(f"⚠️ Could not create 'Date' column from Year/Month. Error: {e}")
        
    # --- Existing logic for 'daily' and 'weekly' sheets ---
    else:
        date_col = choose_date_column(df_processed)
        if date_col not in df_processed.columns:
            st.warning(f"Could not find a date column (e.g., 'Date', 'Month') in {sheet_type} data.")
            return pd.DataFrame()

        id_vars = [date_col]
        if "Week" in df_processed.columns and "Week" != date_col:
            id_vars.append("Week")

        value_cols = [c for c in df_processed.columns if c not in id_vars and c not in ['Unnamed']]
        
        if not value_cols:
            st.warning(f"No data columns found in {sheet_type} data after excluding ID vars.")
            return pd.DataFrame()

        melted = df_processed.melt(id_vars=id_vars, value_vars=value_cols, var_name="Entity", value_name="Distance (Miles)")
        
        melted["Date"] = try_parse_date_series(melted[date_col], date_col)
        has_parsed_date = True
    
    
    melted["Entity"] = melted["Entity"].astype(str).replace({"nan": "Unknown", "None": "Unknown"}).fillna("Unknown")
    melted["Distance (Miles)"] = pd.to_numeric(melted["Distance (Miles)"], errors="coerce").fillna(0)

    if has_parsed_date:
        melted = melted.dropna(subset=["Date"])
    
    # Define all possible columns
    all_cols = ['Date', 'Year', 'Month', 'Week', 'Entity', 'Distance (Miles)']
    # Filter for columns that *actually exist* in the melted df
    cols_to_keep = [c for c in all_cols if c in melted.columns]
    
    out = melted[cols_to_keep]
    if "Date" in out.columns:
        out = out.sort_values("Date").reset_index(drop=True)
    
    return out


@st.cache_data
def ai_summary(prompt, df_summary_for_ai):
    """Send summarized data and prompt to Llama3 via Groq for professional insight."""
    if df_summary_for_ai is None or df_summary_for_ai.empty:
        return "_No data available for AI summary._"
    sample_txt = df_summary_for_ai.to_markdown(index=False)
    user_prompt = f"""
You are a factual data analyst. Your sole purpose is to analyze the data provided and answer the user's prompt.

- **Stick strictly to the data provided in the 'Data sample'.**
- Respond with the **Top 3-4 data-driven observations** as bullet points.
- Each bullet point must be a **single, short sentence**.
- **Do NOT** extrapolate, predict, or invent information not in the sample.
- **Do NOT** use paragraphs.

User Prompt:
{prompt}

Data sample:
{sample_txt}
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.1,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"_AI summary unavailable: {e}_"

# ---------------- UI START ----------------
st.title("🚛 Truck Data AI Dashboard")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    st.divider()
    st.subheader("Performance Filters")
    top_n = st.slider("Top N Performers", 1, 10, 5, key="top_n_slider")
    bottom_n = st.slider("Bottom N Performers", 1, 10, 5, key="bottom_n_slider")
    st.divider()
    diag_expander = st.expander("🔎 Data diagnostics", expanded=True)
    st.session_state['diag_expander'] = diag_expander # Store expander in session state

if not uploaded:
    st.info("Please upload your CSV or Excel file to see the analysis.")
    st.stop()

file_extension = uploaded.name.split('.')[-1].lower()

# --- ⭐️ NEW DATA LOADING AND GENERATION LOGIC ⭐️ ---

# Initialize empty dfs
daily_df = pd.DataFrame()
weekly_df = pd.DataFrame()
monthly_df = pd.DataFrame() 
raw_df = None # To hold the single raw sheet

if file_extension == 'csv':
    raw_df = load_csv(uploaded)
    daily_df = melt_data(raw_df, "daily")
else:
    sheets = load_excel(uploaded)
    if not sheets:
        st.error("Could not load any sheets from the Excel file.")
        st.stop()

    # We now ONLY look for the 'daily' sheet
    daily_key = best_match_key(sheets, ["daily"])
    
    with diag_expander:
        st.write(f"Detected Sheets: {', '.join(sorted(sheets.keys())) or '_none_'}")
        if not daily_key:
            st.warning("⚠️ Could not find a 'Daily' sheet (tried: daily). Please upload a file with a 'daily' sheet.")
            st.stop()
        else:
            st.info(f"✅ Found and loading 'daily' sheet: {daily_key}")
            
    raw_df = sheets.get(daily_key)
    daily_df = melt_data(raw_df, "daily")

# --- Post-processing and Generation ---
if daily_df.empty or 'Date' not in daily_df.columns:
    st.error("Could not process the Daily data. Please ensure it has a valid 'Date' column (or 'Year'/'Month' columns) and data.")
    st.stop()
else:
    st.success("✅ Daily data loaded and processed.")
    with st.spinner("Generating weekly and monthly aggregates..."):
        try:
            daily_df['Date'] = pd.to_datetime(daily_df['Date'])
            df_resample = daily_df.set_index('Date')

            # --- ⭐️ UPDATED: Generate weekly_df using "Majority Day" (Thursday) logic ⭐️ ---
            weekly_agg = df_resample.groupby('Entity').resample('W-Mon', label='left', closed='left')['Distance (Miles)'].sum().reset_index()
            weekly_agg = weekly_agg[weekly_agg['Distance (Miles)'] > 0] # Filter out empty weeks
            
            # --- ⭐️ NEWER Week Labeling Logic (Majority Day) ⭐️ ---
            # Per user request, assign week to the month that has the majority of its days.
            # For a Mon-Sun week, the 4th day (Thursday) determines the majority month.
            weekly_agg['Majority_Day'] = weekly_agg['Date'] + pd.Timedelta(days=3) # 'Date' is Monday, so +3 is Thursday
            
            # Get Year, Month, and Week Num from this "Majority Day" (Thursday)
            weekly_agg['Filter_Year'] = weekly_agg['Majority_Day'].dt.year
            weekly_agg['Filter_Month'] = weekly_agg['Majority_Day'].dt.strftime('%B')
            weekly_agg['Week_Num'] = weekly_agg['Majority_Day'].dt.day.apply(lambda d: (d-1)//7 + 1)
            
            # Create the final "W1", "W2" etc. label for the filter
            weekly_agg['Week_Num_Str'] = 'W' + weekly_agg['Week_Num'].astype(str)
            
            # Create the 'Week' column for display (e.g., "Feb W5")
            # We use the month from the majority day
            weekly_agg['Week'] = weekly_agg['Majority_Day'].dt.strftime('%b') + ' ' + weekly_agg['Week_Num_Str']
            
            weekly_df = weekly_agg
            # --- ⭐️ END OF UPDATED WEEKLY LOGIC ⭐️ ---


            # --- Generate monthly_df ---
            monthly_agg = df_resample.groupby('Entity').resample('MS')['Distance (Miles)'].sum().reset_index()
            monthly_agg = monthly_agg[monthly_agg['Distance (Miles)'] > 0] # Filter out empty months
            
            monthly_agg['Year'] = monthly_agg['Date'].dt.year
            monthly_agg['Month'] = monthly_agg['Date'].dt.strftime('%B')
            monthly_df = monthly_agg
            
            st.success("✅ Weekly and monthly data generated.")
        except Exception as e:
            st.error(f"Failed to generate aggregates: {e}")
            st.stop()

# --- END NEW DATA LOADING ---


with diag_expander:
    st.write("### Processed Data Counts")
    st.write(f"- Daily rows: {len(daily_df)}")
    st.write(f"- Weekly rows (generated): {len(weekly_df)}")
    st.write(f"- Monthly rows (generated): {len(monthly_df)}")

    st.write("### Raw Sheet Preview")
    if raw_df is not None:
        st.write("**Raw 'Daily' Sheet:**")
        st.dataframe(raw_df.head(15))
    
    st.write("### Processed Data Samples")
    if not daily_df.empty:
        st.write("**Processed Daily sample:**")
        st.dataframe(daily_df.head())
    if not weekly_df.empty:
        st.write("**Generated Weekly sample:**")
        st.dataframe(weekly_df.head())
    if not monthly_df.empty:
        st.write("**Generated Monthly sample:**")
        st.dataframe(monthly_df.head())

# --- 1️⃣ KPI OVERVIEW ---
st.header("Fleet Overview")
if not daily_df.empty:
    try:
        # ⭐️ RENAMED 'Activity' to 'Distance (Miles)' ⭐️
        total_activity = daily_df["Distance (Miles)"].sum()
        peak_day_activity = daily_df.groupby("Date")["Distance (Miles)"].sum().max()
        peak_day_date = daily_df.groupby("Date")["Distance (Miles)"].sum().idxmax().strftime('%b %d, %Y')
        total_entities = daily_df["Entity"].nunique()

        col1, col2, col3, col4 = st.columns(4)
        # ⭐️ RENAMED 'Activity' to 'Distance (Miles)' ⭐️
        col1.metric("Total Distance (Daily)", f"{total_activity:,.0f} Miles")
        col2.metric("Total Active Entities", f"{total_entities}")
        col3.metric("Peak Day Distance", f"{peak_day_activity:,.0f} Miles")
        col4.metric("Peak Day Date", f"{peak_day_date}")
    except Exception as e:
        st.info(f"Could not calculate daily KPIs. Daily data might be empty or invalid. Error: {e}")
else:
    st.info("Upload a sheet with daily data to see KPI Overviews.")

st.divider()

# --- 2️⃣ ANALYSIS TABS ---
# All tabs below this line will now work using the generated dfs
tab_perf, tab_util, tab_trends_daily, tab_trends_weekly, tab_risk = st.tabs([
    "🗓️ Performance (Daily)",
    "📊 Utilization (Weekly)",
    "📈 Trend (Daily Peak)",
    "🔬 Deep Dive (Weekly)",
    "📉 Risk & Anomaly (Monthly)"
])

# Performance Tab
with tab_perf:
    st.header("Performance & Consistency Analysis")
    prompt1 = """Analyze the following data of top performers.
Identify who is most consistent (high Mean/Median) vs. 'bursty' (high Total, low Active_Days)."""
    if not daily_df.empty:
        # ⭐️ RENAMED 'Activity' to 'Distance (Miles)' ⭐️
        stats = (
            daily_df.groupby("Entity", as_index=False)
            .agg(Mean=("Distance (Miles)","mean"),
                 Median=("Distance (Miles)","median"),
                 Total=("Distance (Miles)","sum"),
                 Active_Days=("Distance (Miles)", lambda s: (s > 0).sum()))
        )
        daily_smooth = daily_df.copy().sort_values(by=['Entity','Date'])
        # ⭐️ RENAMED 'Activity' to 'Distance (Miles)' ⭐️
        daily_smooth['Distance (7-Day Avg)'] = daily_smooth.groupby('Entity')['Distance (Miles)'] \
                                                    .rolling(window=7, min_periods=1) \
                                                    .mean() \
                                                    .reset_index(level=0, drop=True)
        entity_totals = stats[['Entity','Total']]
        daily_smooth_with_totals = pd.merge(daily_smooth, entity_totals, on='Entity')
        daily_smooth_with_totals['Legend Label'] = daily_smooth_with_totals.apply(
            lambda row: f"{row['Entity']} (Total: {row['Total']:,.0f})", axis=1
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Performers (7-Day Avg)")
            top_stats = stats.nlargest(top_n, "Total")
            top_entities = top_stats['Entity'].tolist()
            top_legend_labels = [f"{row['Entity']} (Total: {row['Total']:,.0f})" for index, row in top_stats.iterrows()]
            plot_data_top = daily_smooth_with_totals[daily_smooth_with_totals['Entity'].isin(top_entities)]
            # ⭐️ RENAMED 'Activity' to 'Distance (Miles)' ⭐️
            fig1 = px.line(plot_data_top, x="Date", y="Distance (7-Day Avg)", color="Legend Label",
                           title="Top Performers (7-Day Avg)", category_orders={"Legend Label": top_legend_labels},
                           hover_name="Entity", hover_data={"Distance (7-Day Avg)": ":.1f", "Legend Label": False, "Date": False},
                           color_discrete_sequence=VIVID_COLORS,
                           labels={"Distance (7-Day Avg)": "Distance (Miles) (7-Day Avg)"}) # Add label for y-axis
            fig1.update_layout(legend_title_text='Performer')
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("Bottom Performers (7-Day Avg)")
            bottom_stats = stats.nsmallest(bottom_n, "Total")
            bottom_entities = bottom_stats['Entity'].tolist()
            bottom_legend_labels = [f"{row['Entity']} (Total: {row['Total']:,.0f})" for index, row in bottom_stats.iterrows()]
            plot_data_bottom = daily_smooth_with_totals[daily_smooth_with_totals['Entity'].isin(bottom_entities)]
            # ⭐️ RENAMED 'Activity' to 'Distance (Miles)' ⭐️
            fig2 = px.line(plot_data_bottom, x="Date", y="Distance (7-Day Avg)", color="Legend Label",
                           title="Bottom Performers (7-Day Avg)", category_orders={"Legend Label": bottom_legend_labels},
                           hover_name="Entity", hover_data={"Distance (7-Day Avg)": ":.1f", "Legend Label": False, "Date": False},
                           color_discrete_sequence=VIVID_COLORS,
                           labels={"Distance (7-Day Avg)": "Distance (Miles) (7-Day Avg)"}) # Add label for y-axis
            fig2.update_layout(legend_title_text='Performer')
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("💡Executive Insight"):
            st.write(ai_summary(prompt1, stats.sort_values("Total", ascending=False).head(10)))

        with st.expander("View Raw Performance Stats (Cumulative Total)"):
            st.dataframe(stats.sort_values("Total", ascending=False), use_container_width=True)
    else:
        st.warning("No daily data available for Performance Analysis. Check the 'Daily' sheet's date column and values.")

# Utilization Tab
with tab_util:
    st.header("Asset Utilization & Efficiency")
    prompt2 = """Analyze the weekly activity split by 'Personnel' vs 'Non-Personnel'.
Summarize the utilization efficiency and identify any peak pattern behavior."""
    if not weekly_df.empty:
        dfw = weekly_df.copy()
        dfw["Entity"] = dfw["Entity"].fillna("Unknown")
        dfw["Category"] = np.where(dfw["Entity"].str.contains("extra", case=False, na=False), "Non-Personnel", "Personnel")
        # ⭐️ RENAMED 'Activity' to 'Distance (Miles)' ⭐️
        total = dfw.groupby(["Date","Category"],as_index=False)["Distance (Miles)"].sum()
        fig3 = px.bar(total, x="Date", y="Distance (Miles)", color="Category", title="Weekly Distance (Miles) by Category",
                      color_discrete_sequence=VIVID_COLORS)
        st.plotly_chart(fig3, use_container_width=True)
        np_trend = total[total["Category"]=="Non-Personnel"]
        if not np_trend.empty:
            # ⭐️ RENAMED 'Activity' to 'Distance (Miles)' ⭐️
            fig4 = px.line(np_trend, x="Date", y="Distance (Miles)", title="Non-Personnel Trend",
                           color_discrete_sequence=VIVID_COLORS)
            st.plotly_chart(fig4, use_container_width=True)
        with st.expander("💡Executive Insight"):
            st.write(ai_summary(prompt2, total))
    else:
        st.warning("No weekly data available. Weekly data is now generated from the 'daily' sheet.")
    st.divider()

    st.header("Strategic Planning & Seasonality")
    prompt6 = """Analyze the weekly total distance and the Week-over-Week % change.
Provide insights on ramp-up/ramp-down speed and operational agility."""
    if not weekly_df.empty:
        # ⭐️ RENAMED 'Activity' to 'Distance (Miles)' ⭐️
        agg = weekly_df.groupby("Date",as_index=False)["Distance (Miles)"].sum().sort_values("Date")
        if len(agg) >= 2:
            agg["WoW_%"] = agg["Distance (Miles)"].pct_change() * 100
            col1, col2 = st.columns(2)
            with col1:
                # ⭐️ RENAMED 'Activity' to 'Distance (Miles)' ⭐️
                fig8 = px.line(agg, x="Date", y="Distance (Miles)", title="Weekly Total Distance (Miles) Trend",
                               color_discrete_sequence=VIVID_COLORS)
                st.plotly_chart(fig8, use_container_width=True)
            with col2:
                fig7 = px.line(agg, x="Date", y="WoW_%", title="Week-over-Week % Change",
                               color_discrete_sequence=VIVID_COLORS,
                               labels={"WoW_%": "WoW % Change"}) # Add label
                st.plotly_chart(fig7, use_container_width=True)
        else:
            st.info("Need at least 2 weeks of data to compute WoW%.")
        with st.expander("💡Executive Insight"):
            st.write(ai_summary(prompt6, agg))
    else:
        st.warning("Strategic seasonality needs weekly data.")

# Trend (Daily Peak) Tab
with tab_trends_daily:
    st.header("Trend Diagnostics (Daily Peak Decline)")
    prompt3 = """Analyze the fleet distance surrounding the single highest-distance day.
Assess whether the decline post-peak is gradual or sudden."""
    if not daily_df.empty:
        # ⭐️ RENAMED 'Activity' to 'Distance (Miles)' ⭐️
        total_daily_activity = daily_df.groupby("Date",as_index=False)["Distance (Miles)"].sum()
        if not total_daily_activity.empty:
            peak_day = total_daily_activity.loc[total_daily_activity["Distance (Miles)"].idxmax(),"Date"]
            window = daily_df[(daily_df["Date"] >= peak_day - pd.Timedelta(days=30)) &
                                (daily_df["Date"] <= peak_day + pd.Timedelta(days=60))]
            if not window.empty:
                # ⭐️ RENAMED 'Activity' to 'Distance (Miles)' ⭐️
                fig5 = px.area(window, x="Date", y="Distance (Miles)", color="Entity", title="Fleet Distance (Miles) around Peak",
                               color_discrete_sequence=VIVID_COLORS)
                st.plotly_chart(fig5, use_container_width=True)
                with st.expander("💡Executive Insight"):
                    window_summary = window.groupby('Date', as_index=False)['Distance (Miles)'].sum()
                    st.write(ai_summary(prompt3, window_summary))
            else:
                st.info("Could not create a window around the peak — check date parsing.")
        else:
            st.warning("Daily data exists but total aggregation is empty.")
    else:
        st.warning("Trend diagnostics need daily data.")

# --- ⭐️ WEEKLY DEEP DIVE TAB (No changes needed, it uses the new columns) ⭐️ ---
with tab_trends_weekly:
    st.header("Weekly Deep Dive Analysis")
    prompt_weekly_dive = "Analyze the performance of all entities for this specific week."
    
    if not weekly_df.empty:
        # ⭐️ UPDATED: Check for the new pre-calculated columns
        if 'Filter_Year' not in weekly_df.columns or 'Filter_Month' not in weekly_df.columns or 'Week_Num_Str' not in weekly_df.columns:
            st.warning("Cannot create filters. Key columns ('Filter_Year', 'Filter_Month', 'Week_Num_Str') are missing from generated weekly data.")
            st.stop()
            
        # --- START NEW LOGIC ---
        
        # 1. Define month order for sorting
        month_order = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        
        # 2. Use the pre-calculated columns
        df_filter = weekly_df.copy()
        
        # 3. Create 3 columns for dropdowns
        col1, col2, col3 = st.columns(3)

        # 4. Filter 1: Year (Use Filter_Year)
        with col1:
            all_years = sorted(df_filter['Filter_Year'].unique(), reverse=True)
            if not all_years:
                st.info("No data available.")
                st.stop()
            selected_year = st.selectbox("Select Year", all_years, key="deep_dive_year")

        # 5. Filter 2: Month (Cascading) (Use Filter_Month)
        with col2:
            # Get months for the selected year
            months_for_year = df_filter[df_filter['Filter_Year'] == selected_year]['Filter_Month'].unique()
            # Sort them using the dictionary
            sorted_months = sorted(months_for_year, key=lambda m: month_order.get(m, 99))
            if not sorted_months:
                st.info("No months found for this year.")
                st.stop()
            selected_month = st.selectbox("Select Month", sorted_months, key="deep_dive_month")

        # 6. Filter 3: Week (Cascading) (Use Week_Num_Str)
        with col3:
            # Get weeks for the selected year AND month
            weeks_for_month = df_filter[
                (df_filter['Filter_Year'] == selected_year) &
                (df_filter['Filter_Month'] == selected_month)
            ]['Week_Num_Str'].unique()
            
            sorted_weeks = sorted(weeks_for_month) # Simple sort (W1, W2...)
            if not sorted_weeks:
                st.info("No weeks found for this month.")
                st.stop()
            selected_week = st.selectbox("Select Week", sorted_weeks, key="deep_dive_week")

        # 7. Update the final filtering logic
        week_data_melted = df_filter[
            (df_filter['Filter_Year'] == selected_year) &
            (df_filter['Filter_Month'] == selected_month) &
            (df_filter['Week_Num_Str'] == selected_week)
        ]
        
        # --- END NEW LOGIC ---
        
        st.divider()
        
        if not week_data_melted.empty:
            # 8. Update labels to use all 3 filters
            st.subheader(f"Distance (Miles) Comparison for {selected_week}, {selected_month} {selected_year}")
            
            week_data_filtered = week_data_melted[week_data_melted['Distance (Miles)'] > 0].sort_values("Distance (Miles)", ascending=False)
            
            if not week_data_filtered.empty:
                fig_compare_title = f"Entity Performance for {selected_week}, {selected_month} {selected_year}"
                fig_compare = px.bar(week_data_filtered, x="Entity", y="Distance (Miles)", color="Entity",
                                     title=fig_compare_title,
                                     color_discrete_sequence=VIVID_COLORS)
                fig_compare.update_layout(xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig_compare, use_container_width=True)
                
                with st.expander("💡Executive Insight"):
                    st.write(ai_summary(prompt_weekly_dive, week_data_filtered))
            else:
                st.info(f"No distance recorded for any entity in {selected_week}, {selected_month} {selected_year}.")
        else:
            st.info(f"No data found for {selected_week}, {selected_month} {selected_year}.")
    else:
        st.warning("Weekly data is not available (generated from 'daily' sheet).")
# --- ⭐️ END OF WEEKLY TAB ⭐️ ---


# --- ⭐️ UPDATED Risk & Anomaly Tab (Using GENERATED Data) ⭐️ ---
with tab_risk:
    st.header("Risk & Dependency Analysis (Select Year & Month)")
    
    # Create a sort key to order month names correctly
    month_order = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }

    if not monthly_df.empty:
        mdf = monthly_df.copy()
        
        # We know these columns exist from our generation step
        if 'Year' not in mdf.columns or 'Month' not in mdf.columns:
            st.warning("Monthly sheet is missing 'Year' or 'Month' columns. Check data diagnostics.")
        
        else:
            # --- 1. Setup Filters ---
            try:
                mdf['Year'] = pd.to_numeric(mdf['Year'], errors='coerce')
                mdf['Month'] = mdf['Month'].astype(str).str.strip()
                mdf = mdf.dropna(subset=['Year']) # Clean any bad data
                mdf['Year'] = mdf['Year'].astype(int)
            except Exception as e:
                st.error(f"Could not format Year/Month columns. Error: {e}")
                st.stop()

            years = sorted(mdf["Year"].unique(), reverse=True)
            if not years:
                st.info("No year information found in generated monthly data.")
            else:
                col_a, col_b = st.columns([1,1])
                with col_a:
                    selected_year = st.selectbox("Select Year", years, index=0, key="risk_year_select")

                # Build month options for selected year in order
                months_for_year = mdf[mdf["Year"] == selected_year]["Month"].unique().tolist()
                sorted_months = sorted(months_for_year, key=lambda m: month_order.get(m.title(), 99))

                with col_b:
                    if sorted_months:
                        default_index = max(0, len(sorted_months) - 1)
                        sel_month_label = st.selectbox("Select Month", sorted_months, index=default_index, key="risk_month_select")
                    else:
                        sel_month_label = st.selectbox("Select Month", ["_no months_"], key="risk_month_select")

                if not sel_month_label or sel_month_label == "_no months_":
                    st.info("No months found for the selected year.")
                else:
                    # --- 2. Bar Chart for Selected Month ---
                    mdf_selected = mdf[(mdf["Year"] == selected_year) & (mdf["Month"] == sel_month_label)].copy()
                    
                    st.divider()
                    st.write(f"### Showing data for: **{sel_month_label} {selected_year}**")
                    
                    st.subheader(f"Distance (Miles) for {sel_month_label} {selected_year}")
                    if not mdf_selected.empty:
                        mdf_selected_chart = mdf_selected[mdf_selected["Distance (Miles)"] > 0].sort_values("Distance (Miles)", ascending=False)
                        if not mdf_selected_chart.empty:
                            fig_bar_monthly = px.bar(mdf_selected_chart, x="Entity", y="Distance (Miles)", color="Entity",
                                                     title=f"Entity Performance for {sel_month_label} {selected_year}",
                                                     color_discrete_sequence=VIVID_COLORS)
                            fig_bar_monthly.update_layout(xaxis={'categoryorder':'total descending'})
                            st.plotly_chart(fig_bar_monthly, use_container_width=True)
                            
                            # --- ⭐️ ADDED AI INSIGHT FOR MONTHLY ⭐️ ---
                            with st.expander("💡Executive Insight (Monthly)"):
                                prompt_monthly = f"Analyze the performance of entities for {sel_month_label} {selected_year}, noting any major dependencies."
                                st.write(ai_summary(prompt_monthly, mdf_selected_chart))

                            with st.expander("View Filtered Data Table"):
                                st.dataframe(mdf_selected, use_container_width=True)
                        else:
                            st.info(f"No distance recorded for any entity in {sel_month_label} {selected_year}.")
                    else:
                        st.info(f"No data found for {sel_month_label} {selected_year}.")

                    st.divider()

                    # --- 3. Advanced Charts (These REQUIRE a valid 'Date' column) ---
                    # The 'Date' column was generated with monthly_df, so this just works.
                    try:
                        mdf_with_date = mdf.dropna(subset=['Date']).copy()
                        
                        if mdf_with_date.empty:
                            raise ValueError("Date column is empty after dropping NaNs.")

                        # --- ⭐️ ADDED QUARTERLY ANALYSIS ⭐️ ---
                        st.header(f"Quarterly Analysis")
                        mdf_with_date["Quarter"] = mdf_with_date["Date"].dt.to_period("Q").astype(str)
                        
                        # Aggregate all data by quarter
                        q_totals = mdf_with_date.groupby("Quarter", as_index=False)["Distance (Miles)"].sum()
                        
                        if not q_totals.empty:
                            fig_q = px.bar(q_totals, x="Quarter", y="Distance (Miles)", title="Total Distance (Miles) by Quarter",
                                           color="Quarter", color_discrete_sequence=VIVID_COLORS)
                            fig_q.update_layout(xaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig_q, use_container_width=True)
                            
                            # Show peak/low quarters based on all data
                            peak_q = q_totals.loc[q_totals["Distance (Miles)"].idxmax(), "Quarter"]
                            low_q = q_totals.loc[q_totals["Distance (Miles)"].idxmin(), "Quarter"]
                            
                            st.write(f"**Overall Peak Quarter:** {peak_q} ({q_totals[q_totals['Quarter']==peak_q]['Distance (Miles)'].values[0]:,.0f} total miles)")
                            st.write(f"**Overall Low Quarter:** {low_q} ({q_totals[q_totals['Quarter']==low_q]['Distance (Miles)'].values[0]:,.0f} total miles)")

                            with st.expander("💡Executive Insight (QuarterDly)"):
                                prompt_quarterly = "Analyze the quarterly distance data, identifying peak quarters and any significant trends."
                                st.write(ai_summary(prompt_quarterly, q_totals))
                        else:
                            st.info("No quarterly aggregation available.")
                    
                    except Exception as e:
                        st.warning(f"⚠️ Could not display quarterly charts. Error: {e}")

    else:
        st.warning("No monthly data available for Risk Analysis. (Generated from 'daily' sheet).")

st.success("✅ Dashboard loaded successfully.")