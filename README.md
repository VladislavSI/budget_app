# 💰 Personal Budget Tracker

A clean, modern budgeting dashboard built with Streamlit that reads your income/expenses from Google Sheets.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- 📊 **Real-time dashboard** with income vs expenses trends
- 🥧 **Category breakdown** pie charts
- 📈 **Monthly trend analysis**
- 🔍 **Flexible filtering** by date, category, and transaction type
- 🎨 **Dark theme** with modern UI
- 🔄 **Auto-refresh** every 5 minutes
- 📱 **Responsive** design

## Quick Start

### 1. Clone & Install

```bash
git clone <your-repo>
cd budget_app
pip install -r requirements.txt
```

### 2. Set Up Google Sheets

#### Create Your Budget Spreadsheet

Create a Google Sheet with this structure:

| Date | Category | Type | Description | Amount |
|------|----------|------|-------------|--------|
| 2024-01-01 | Salary | Income | Monthly salary | 5000 |
| 2024-01-05 | Groceries | Expense | Weekly shopping | -150 |
| 2024-01-10 | Rent | Expense | January rent | -1200 |

**Important:**
- `Date`: Format as YYYY-MM-DD
- `Amount`: Positive for income, negative for expenses
- `Type`: "Income" or "Expense"
- `Category`: Your custom categories

#### Create Google Cloud Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Enable APIs:
   - Google Sheets API
   - Google Drive API
4. Create Service Account:
   - Go to **IAM & Admin** → **Service Accounts**
   - Click **Create Service Account**
   - Give it a name (e.g., "budget-app-reader")
   - Click **Create and Continue**
   - Skip role assignment (click **Done**)
5. Create Key:
   - Click on your new service account
   - Go to **Keys** tab
   - Click **Add Key** → **Create new key**
   - Select **JSON** and download

#### Share Your Sheet

Share your Google Sheet with the service account email:
- Open your budget spreadsheet
- Click **Share**
- Add the service account email (from the JSON file, looks like: `name@project.iam.gserviceaccount.com`)
- Give **Viewer** access

### 3. Configure Secrets

#### For Local Development

```bash
mkdir -p .streamlit
cp secrets.toml.template .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:
- Paste your spreadsheet ID
- Paste the entire service account JSON content

#### For Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy your app
4. Go to **Settings** → **Secrets**
5. Paste your secrets in TOML format

### 4. Run

```bash
streamlit run app.py
```

Visit `http://localhost:8501` 🎉

## Project Structure

```
budget_app/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── secrets.toml.template     # Template for secrets configuration
├── README.md                 # This file
└── .streamlit/
    └── secrets.toml          # Your actual secrets (gitignored)
```

## Configuration

### Environment Variables / Secrets

| Key | Description |
|-----|-------------|
| `SPREADSHEET_ID` | Your Google Sheet ID from the URL |
| `gcp_service_account` | Full service account JSON credentials |

### Customization

**Change categories:**
Edit the `categories_expense` and `categories_income` lists in `load_demo_data()`.

**Adjust cache time:**
Change `@st.cache_data(ttl=300)` - value is in seconds.

**Modify styling:**
Edit the CSS in `st.markdown()` under `CUSTOM STYLING` section.

## Deployment Options

### Streamlit Community Cloud (Recommended - Free)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Add secrets in settings
5. Deploy!

### Other Options

- **Heroku**: Add `Procfile` with `web: streamlit run app.py`
- **Railway**: Automatic detection
- **Self-hosted**: Run with Docker or directly

## Troubleshooting

### "Permission denied" errors
- Make sure you shared the Google Sheet with the service account email
- Check the service account has at least Viewer access

### "API not enabled" errors
- Go to Google Cloud Console
- Enable both Google Sheets API and Google Drive API

### Data not loading
- Verify your spreadsheet ID is correct
- Check date format is YYYY-MM-DD
- Ensure Amount column contains numbers only

## Demo Mode

The app includes demo data for testing. Toggle "Use Demo Data" in the sidebar to switch between demo and your actual Google Sheet data.

## License

MIT License - feel free to modify and use as you wish!

---

Built with ❤️ using [Streamlit](https://streamlit.io)
