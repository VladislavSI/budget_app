"""
Personal Budget Tracker - Streamlit App
Reads income/expenses from Google Sheets in Google Drive
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from google.oauth2.service_account import Credentials
import gspread

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Budget Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Google Sheets Configuration
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets.readonly',
    'https://www.googleapis.com/auth/drive.readonly'
]

# Your Google Sheet ID (from the URL)
# Example: https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit
SPREADSHEET_ID = st.secrets.get("SPREADSHEET_ID", "YOUR_SPREADSHEET_ID_HERE")

# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
<style>
    /* Import distinctive fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Main app styling */
    .stApp {
        font-family: 'DM Sans', sans-serif;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid #0f3460;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
    }
    
    /* Positive/Negative indicators */
    .positive { color: #10b981 !important; }
    .negative { color: #ef4444 !important; }
    
    /* Custom card */
    .budget-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Table styling */
    .dataframe {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA CONNECTION
# ============================================================================

@st.cache_resource
def get_google_client():
    """Initialize Google Sheets client using service account credentials."""
    try:
        # Load credentials from Streamlit secrets
        credentials = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=SCOPES
        )
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        st.error(f"Failed to connect to Google: {e}")
        return None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_budget_data(_client, spreadsheet_id: str) -> pd.DataFrame:
    """
    Load budget data from Google Sheets.
    
    Supports Vlad's FamilyBudget format:
    | Date | Flag | Category | Where | IncomeAmount | OutcomeAmount | ... |
    
    Maps to internal format:
    | Date | Category | Type | Description | Amount |
    
    Loads from both 'Facts_New' (EUR) and 'Facts' (BGN, converted to EUR) tabs.
    """
    
    # BGN to EUR conversion rate (fixed rate)
    BGN_TO_EUR = 1.95583
    
    def parse_european_number(val):
        """Parse numbers with European format (comma as decimal, space as thousands)."""
        if pd.isna(val) or val == '' or val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        # Handle string: remove spaces (thousands separator), replace comma with dot
        val_str = str(val).strip()
        # Check if it uses European format: "1 234,56" or "1234,56"
        # Remove space (thousands separator)
        val_str = val_str.replace(' ', '')
        # Replace comma with dot (decimal separator)
        val_str = val_str.replace(',', '.')
        try:
            return float(val_str)
        except:
            return 0.0
    
    def process_sheet(sheet, convert_bgn_to_eur: bool = False) -> pd.DataFrame:
        """Process a single sheet and return cleaned DataFrame."""
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        if df.empty:
            return pd.DataFrame()
        
        # Parse Date (handles DD.MM.YYYY format)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        
        # Map Type from Flag column (Income/Expense)
        if 'Flag' in df.columns and 'Type' not in df.columns:
            df['Type'] = df['Flag']
        
        # Map Description from Where column
        if 'Where' in df.columns and 'Description' not in df.columns:
            df['Description'] = df['Where']
        
        # Combine IncomeAmount and OutcomeAmount into single Amount column
        if 'IncomeAmount' in df.columns and 'OutcomeAmount' in df.columns:
            df['IncomeAmount'] = df['IncomeAmount'].apply(parse_european_number)
            df['OutcomeAmount'] = df['OutcomeAmount'].apply(parse_european_number)
            
            # Income is positive, Expenses are negative
            df['Amount'] = df['IncomeAmount'] - df['OutcomeAmount']
        elif 'Amount' in df.columns:
            df['Amount'] = df['Amount'].apply(parse_european_number)
        
        # Convert BGN to EUR if needed
        if convert_bgn_to_eur:
            df['Amount'] = df['Amount'] / BGN_TO_EUR
        
        # Ensure required columns exist
        required = ['Date', 'Category', 'Amount']
        for col in required:
            if col not in df.columns:
                return pd.DataFrame()
        
        # Fill missing optional columns
        if 'Type' not in df.columns:
            df['Type'] = df['Amount'].apply(lambda x: 'Income' if x > 0 else 'Expense')
        
        if 'Description' not in df.columns:
            df['Description'] = ''
        
        # Clean up: remove rows with invalid dates or zero amounts
        df = df.dropna(subset=['Date'])
        df = df[df['Amount'] != 0]
        
        # Select and reorder final columns
        return df[['Date', 'Category', 'Type', 'Description', 'Amount']]
    
    try:
        st.info(f"Attempting to open spreadsheet: {spreadsheet_id[:20]}...")
        spreadsheet = _client.open_by_key(spreadsheet_id)
        
        all_data = []
        
        # Load Facts_New sheet (already in EUR)
        try:
            st.info("Loading 'Facts_New' tab (EUR)...")
            sheet_new = spreadsheet.worksheet("Facts_New")
            df_new = process_sheet(sheet_new, convert_bgn_to_eur=False)
            if not df_new.empty:
                all_data.append(df_new)
                st.info(f"Facts_New: {len(df_new)} transactions (EUR)")
        except Exception as e:
            st.warning(f"Could not load Facts_New: {e}")
        
        # Load Facts sheet (BGN - convert to EUR)
        try:
            st.info("Loading 'Facts' tab (BGN → EUR)...")
            sheet_old = spreadsheet.worksheet("Facts")
            df_old = process_sheet(sheet_old, convert_bgn_to_eur=True)
            if not df_old.empty:
                all_data.append(df_old)
                st.info(f"Facts: {len(df_old)} transactions (converted from BGN)")
        except Exception as e:
            st.warning(f"Could not load Facts: {e}")
        
        # Combine all data
        if not all_data:
            st.error("No data found in any sheet")
            return pd.DataFrame()
        
        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates()  # Remove any duplicates
        df = df.sort_values('Date', ascending=False)
        
        st.success(f"Successfully loaded {len(df)} total transactions!")
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {type(e).__name__}: {e}")
        return pd.DataFrame()
        st.error(f"Error loading data: {type(e).__name__}: {e}")
        return pd.DataFrame()


def load_demo_data() -> pd.DataFrame:
    """Generate demo data for testing without Google connection."""
    import random
    
    categories_income = ['Salary', 'Freelance', 'Investments', 'Other Income']
    categories_expense = ['Rent', 'Groceries', 'Utilities', 'Transport', 
                          'Entertainment', 'Dining', 'Shopping', 'Healthcare']
    
    data = []
    start_date = datetime.now() - timedelta(days=180)
    
    for i in range(180):
        current_date = start_date + timedelta(days=i)
        
        # Monthly salary on 1st
        if current_date.day == 1:
            data.append({
                'Date': current_date,
                'Category': 'Salary',
                'Type': 'Income',
                'Description': 'Monthly salary',
                'Amount': random.randint(4500, 5500)
            })
        
        # Random expenses
        if random.random() > 0.5:
            cat = random.choice(categories_expense)
            data.append({
                'Date': current_date,
                'Category': cat,
                'Type': 'Expense',
                'Description': f'{cat} expense',
                'Amount': -random.randint(10, 300)
            })
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date', ascending=False)


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Render the app header."""
    st.markdown("""
        <div style="text-align: center; padding: 20px 0 40px 0;">
            <h1 style="font-size: 3rem; margin-bottom: 0; 
                       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                💰 Budget Tracker
            </h1>
            <p style="color: #64748b; font-size: 1.1rem; margin-top: 8px;">
                Your personal income & expense dashboard
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_kpi_cards(df: pd.DataFrame, period_label: str):
    """Render the main KPI metric cards."""
    total_income = df[df['Amount'] > 0]['Amount'].sum()
    total_expenses = abs(df[df['Amount'] < 0]['Amount'].sum())
    net_balance = total_income - total_expenses
    savings_rate = (net_balance / total_income * 100) if total_income > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=f"💵 Total Income ({period_label})",
            value=f"${total_income:,.2f}",
        )
    
    with col2:
        st.metric(
            label=f"💸 Total Expenses ({period_label})",
            value=f"${total_expenses:,.2f}",
        )
    
    with col3:
        delta_color = "normal" if net_balance >= 0 else "inverse"
        st.metric(
            label="📊 Net Balance",
            value=f"${net_balance:,.2f}",
            delta=f"{'Surplus' if net_balance >= 0 else 'Deficit'}",
            delta_color=delta_color
        )
    
    with col4:
        st.metric(
            label="🎯 Savings Rate",
            value=f"{savings_rate:.1f}%",
            delta=f"{'On track' if savings_rate >= 20 else 'Below target'}",
            delta_color="normal" if savings_rate >= 20 else "inverse"
        )


def render_trend_chart(df: pd.DataFrame):
    """Render income vs expenses trend over time."""
    monthly = df.copy()
    monthly['Month'] = monthly['Date'].dt.to_period('M').astype(str)
    
    income_monthly = monthly[monthly['Amount'] > 0].groupby('Month')['Amount'].sum().reset_index()
    income_monthly.columns = ['Month', 'Income']
    
    expense_monthly = monthly[monthly['Amount'] < 0].groupby('Month')['Amount'].sum().abs().reset_index()
    expense_monthly.columns = ['Month', 'Expenses']
    
    merged = pd.merge(income_monthly, expense_monthly, on='Month', how='outer').fillna(0)
    merged = merged.sort_values('Month')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=merged['Month'],
        y=merged['Income'],
        name='Income',
        line=dict(color='#10b981', width=3),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=merged['Month'],
        y=merged['Expenses'],
        name='Expenses',
        line=dict(color='#ef4444', width=3),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.1)'
    ))
    
    fig.update_layout(
        title='Monthly Income vs Expenses Trend',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans'),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_category_breakdown(df: pd.DataFrame):
    """Render expense breakdown by category."""
    expenses = df[df['Amount'] < 0].copy()
    expenses['Amount'] = expenses['Amount'].abs()
    
    by_category = expenses.groupby('Category')['Amount'].sum().reset_index()
    by_category = by_category.sort_values('Amount', ascending=False)
    
    # Color palette
    colors = ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899', 
              '#f43f5e', '#f97316', '#eab308']
    
    fig = px.pie(
        by_category,
        values='Amount',
        names='Category',
        hole=0.5,
        color_discrete_sequence=colors
    )
    
    fig.update_layout(
        title='Expense Breakdown by Category',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans'),
        showlegend=True,
        legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.02)
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_top_expenses(df: pd.DataFrame, n: int = 10):
    """Render top expense categories as horizontal bar chart."""
    expenses = df[df['Amount'] < 0].copy()
    expenses['Amount'] = expenses['Amount'].abs()
    
    by_category = expenses.groupby('Category')['Amount'].sum().reset_index()
    by_category = by_category.sort_values('Amount', ascending=True).tail(n)
    
    fig = go.Figure(go.Bar(
        x=by_category['Amount'],
        y=by_category['Category'],
        orientation='h',
        marker=dict(
            color=by_category['Amount'],
            colorscale='Reds',
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        text=[f'${x:,.0f}' for x in by_category['Amount']],
        textposition='inside',
        textfont=dict(color='white', family='JetBrains Mono')
    ))
    
    fig.update_layout(
        title=f'Top {n} Expense Categories',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans'),
        xaxis_title='Amount ($)',
        yaxis_title='',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_transactions_table(df: pd.DataFrame):
    """Render recent transactions table."""
    display_df = df.head(20).copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    display_df['Amount'] = display_df['Amount'].apply(
        lambda x: f"${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}"
    )
    
    st.dataframe(
        display_df[['Date', 'Category', 'Type', 'Description', 'Amount']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'Date': st.column_config.TextColumn('Date', width='small'),
            'Category': st.column_config.TextColumn('Category', width='medium'),
            'Type': st.column_config.TextColumn('Type', width='small'),
            'Description': st.column_config.TextColumn('Description', width='large'),
            'Amount': st.column_config.TextColumn('Amount', width='small'),
        }
    )


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar(df: pd.DataFrame):
    """Render sidebar with filters."""
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: #a78bfa;">⚙️ Filters</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Data source toggle
    use_demo = st.sidebar.toggle("Use Demo Data", value=True, 
                                  help="Toggle to use demo data or connect to Google Sheets")
    
    st.sidebar.divider()
    
    # Date range filter
    if not df.empty:
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        date_range = st.sidebar.date_input(
            "📅 Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Quick date presets
        st.sidebar.markdown("**Quick Select:**")
        col1, col2 = st.sidebar.columns(2)
        
        preset = None
        with col1:
            if st.button("This Month", use_container_width=True):
                preset = 'this_month'
            if st.button("Last 3 Mo", use_container_width=True):
                preset = 'last_3_months'
        with col2:
            if st.button("Last Month", use_container_width=True):
                preset = 'last_month'
            if st.button("This Year", use_container_width=True):
                preset = 'this_year'
        
        st.sidebar.divider()
        
        # Category filter
        categories = ['All'] + sorted(df['Category'].unique().tolist())
        selected_categories = st.sidebar.multiselect(
            "🏷️ Categories",
            options=categories,
            default=['All']
        )
        
        # Type filter
        type_filter = st.sidebar.radio(
            "💱 Transaction Type",
            options=['All', 'Income', 'Expense'],
            horizontal=True
        )
        
        return {
            'use_demo': use_demo,
            'date_range': date_range,
            'preset': preset,
            'categories': selected_categories,
            'type_filter': type_filter
        }
    
    return {'use_demo': use_demo}


def apply_filters(df: pd.DataFrame, filters: dict) -> tuple:
    """Apply sidebar filters to dataframe."""
    filtered = df.copy()
    period_label = "All Time"
    
    # Date filtering
    if 'preset' in filters and filters['preset']:
        today = datetime.now()
        if filters['preset'] == 'this_month':
            start = today.replace(day=1)
            filtered = filtered[filtered['Date'] >= start]
            period_label = "This Month"
        elif filters['preset'] == 'last_month':
            first_this_month = today.replace(day=1)
            last_month_end = first_this_month - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            filtered = filtered[(filtered['Date'] >= last_month_start) & 
                               (filtered['Date'] <= last_month_end)]
            period_label = "Last Month"
        elif filters['preset'] == 'last_3_months':
            start = today - timedelta(days=90)
            filtered = filtered[filtered['Date'] >= start]
            period_label = "Last 3 Months"
        elif filters['preset'] == 'this_year':
            start = today.replace(month=1, day=1)
            filtered = filtered[filtered['Date'] >= start]
            period_label = "This Year"
    elif 'date_range' in filters and len(filters['date_range']) == 2:
        start, end = filters['date_range']
        filtered = filtered[(filtered['Date'].dt.date >= start) & 
                           (filtered['Date'].dt.date <= end)]
        period_label = f"{start} to {end}"
    
    # Category filtering
    if 'categories' in filters and 'All' not in filters['categories']:
        filtered = filtered[filtered['Category'].isin(filters['categories'])]
    
    # Type filtering
    if 'type_filter' in filters and filters['type_filter'] != 'All':
        if filters['type_filter'] == 'Income':
            filtered = filtered[filtered['Amount'] > 0]
        else:
            filtered = filtered[filtered['Amount'] < 0]
    
    return filtered, period_label


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    render_header()
    
    # Load initial data for sidebar
    initial_df = load_demo_data()
    
    # Render sidebar and get filters
    filters = render_sidebar(initial_df)
    
    # Load actual data based on toggle
    if filters.get('use_demo', True):
        df = load_demo_data()
        st.sidebar.success("✅ Using demo data")
    else:
        client = get_google_client()
        if client:
            df = load_budget_data(client, SPREADSHEET_ID)
            if df.empty:
                st.warning("No data found. Check your Google Sheet configuration.")
                df = load_demo_data()
        else:
            st.warning("Could not connect to Google. Using demo data.")
            df = load_demo_data()
    
    if df.empty:
        st.error("No data available to display.")
        return
    
    # Apply filters
    filtered_df, period_label = apply_filters(df, filters)
    
    if filtered_df.empty:
        st.warning("No transactions match your filters.")
        return
    
    # Render KPIs
    render_kpi_cards(filtered_df, period_label)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_trend_chart(filtered_df)
    
    with col2:
        render_category_breakdown(filtered_df)
    
    # Second row
    col1, col2 = st.columns([1, 1])
    
    with col1:
        render_top_expenses(filtered_df)
    
    with col2:
        st.subheader("📋 Recent Transactions")
        render_transactions_table(filtered_df)
    
    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 40px 0 20px 0; color: #64748b;">
            <p>Built with Streamlit • Data refreshes every 5 minutes</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
