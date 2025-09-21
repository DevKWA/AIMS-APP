import io
import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import google.generativeai as genai
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import re
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

# ---------------------------
# Streamlit page config
# ---------------------------
st.warning(
"""
⚠️ **Content Warning:** This dashboard contains real survivor stories of modern slavery and human trafficking. Some descriptions may be distressing.

All stories are from open, public, approved sources.
Each story includes a link to the full source. Please consult the original source.

Please use this tool responsibly, with respect and sensitivity toward survivors. Younger audiences, survivors, or those who may be triggered by descriptions of violence, exploitation, or abuse should proceed with caution.

These stories are shared for educational and awareness purposes only and should be engaged with thoughtfully, keeping in mind the dignity and resilience of survivors.
"""
)
st.set_page_config(layout="wide", page_title="Survivor Dashboard + AIMS LLM")

# ---------------------------
# Global CSS
# ---------------------------
st.markdown("""
<style>
.stApp { background-color: white; }
html, body, [class*="css"] { color: black !important; }
.streamlit-expander { width: 100% !important; }
.stDataFrame td, .stDataFrame th { color: black !important; background-color: white !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Banner images
# ---------------------------
st.image('assets/AULA_HORIZONTAL_GREEN_BANNER.png', use_container_width=True)
st.image('assets/survivor_dashboard_banner.png', use_container_width=True)

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📖 Dashboard Documentation & Instructions", "📊 Survivor Dashboard", "🤖 Insight on Supply Chains (from AIMS)"])

# ---------------------------
# TAB 1: Survivor Dashboard
# ---------------------------
with tab2:
    st.sidebar.header("Data Source Options")
    sheet_url = st.sidebar.text_input("📄 Google Sheet URL (public or private)")

    df = None
    if sheet_url:
        try:
            SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
                      "https://www.googleapis.com/auth/drive.file"]
            creds = Credentials.from_service_account_info(
                st.secrets["google_service_account"], scopes=SCOPES
            )
            gc = gspread.authorize(creds)
            sh = gc.open_by_url(sheet_url)
            worksheet_names = [ws.title for ws in sh.worksheets()]
            worksheet_name = st.sidebar.selectbox("📑 Choose a worksheet/tab", worksheet_names)
            ws = sh.worksheet(worksheet_name)
            df = pd.DataFrame(ws.get_all_records())
            st.sidebar.success(f"✅ Loaded worksheet: {worksheet_name}")
        except Exception as ex:
            st.sidebar.error(f"❌ Error loading sheet: {ex}")

    if df is None or df.empty:
        st.warning("Please provide a Google Sheet URL to continue.")
        st.stop()

    # ---------------------------
    # Sidebar filters (drill-down)
    # ---------------------------
    st.sidebar.header("🔍 Filters")
    filtered_for_filters = df.copy()

    # Region filter
    regions = sorted(filtered_for_filters['REGION'].dropna().unique()) if 'REGION' in df.columns else []
    selected_region = st.sidebar.selectbox("🌍 Select Region", options=["All"] + regions)
    if selected_region != "All":
        filtered_for_filters = filtered_for_filters[filtered_for_filters['REGION'] == selected_region]

    # Industry filter
    industries = sorted(filtered_for_filters['INDUSTRY'].dropna().unique()) if 'INDUSTRY' in filtered_for_filters.columns else []
    selected_industry = st.sidebar.selectbox("🏭 Select Industry", options=["All"] + industries)
    if selected_industry != "All":
        filtered_for_filters = filtered_for_filters[filtered_for_filters['INDUSTRY'] == selected_industry]

    # Category filter
    categories = sorted(filtered_for_filters['CATEGORY'].dropna().unique()) if 'CATEGORY' in filtered_for_filters.columns else []
    selected_category = st.sidebar.selectbox("📂 Select Category", options=["All"] + categories)
    if selected_category != "All":
        filtered_for_filters = filtered_for_filters[filtered_for_filters['CATEGORY'] == selected_category]

    # Title filter
    titles = sorted(filtered_for_filters['TITLE'].dropna().unique()) if 'TITLE' in filtered_for_filters.columns else []
    selected_title = st.sidebar.selectbox("📌 Select Title", options=["All"] + titles)
    if selected_title != "All":
        filtered_for_filters = filtered_for_filters[filtered_for_filters['TITLE'] == selected_title]

    # Link filter
    link_filter = st.sidebar.radio("🔗 Filter by Links", options=["All", "With Links", "Without Links"])
    filtered_df = filtered_for_filters.copy()
    if link_filter == "With Links" and 'LINK' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['LINK'].notna() & (filtered_df['LINK'] != "")]
    elif link_filter == "Without Links" and 'LINK' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['LINK'].isna() | (filtered_df['LINK'] == "")]

    if filtered_df.empty:
        st.warning("❌ No results match the selected filters.")
        st.stop()

    # ---------------------------
    # Results header
    # ---------------------------
    st.markdown(f"""
    <div style="padding:10px; border-radius:8px; background-color:#ffffff; border-left:6px solid #000000;">
        <h3 style="color:black; text-align:left;">
        📊 Showing Results: Region: {selected_region} | Industry: {selected_industry} | Category: {selected_category} | Title: {selected_title} | Links: {link_filter}
        </h3>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------
    # Display rows with emojis
    # ---------------------------
    region_icons = {"Africa": "🌍", "Europe": "🌎", "Asia": "🌏"}
    industry_icons = {"Oil & Gas": "🏭", "Agriculture": "🌾", "Technology": "💻"}
    category_icons = {"Labor": "👷", "Human Rights": "✊", "Safety": "🛡️"}

    for _, row in filtered_df.iterrows():
        region_icon = region_icons.get(row.get('REGION', ''), "🌐")
        industry_icon = industry_icons.get(row.get('INDUSTRY', ''), "🏢")
        category_icon = category_icons.get(row.get('CATEGORY', ''), "📂")
        link_html = f"<a href='{row.get('LINK','')}' target='_blank' style='color:blue; text-decoration:underline;'>Read More</a>" if row.get('LINK') else ""
        st.markdown(f"""
            <div style="padding:15px; border-radius:12px; background-color:#ffffff; margin-bottom:10px; border:1px solid #e0e0e0;">
                <strong>{region_icon} {industry_icon} {category_icon} {row.get('TITLE','Untitled')}</strong><br>
                {row.get('DESCRIPTION','')}<br>
                {link_html}
            </div>
        """, unsafe_allow_html=True)

# ---------------------------
# TAB 2: AIMS LLM
# ---------------------------
# ---------------------------
# Initialize AIMSDistill model and tokenizer
# ---------------------------
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AimsDistillModel(nn.Module):
    def __init__(self, tokenizer, model_name, dropout=0.0):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 11)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.dropout(self.classifier(pooled_output))
        return logits

model = AimsDistillModel(tokenizer, model_name).to(device)
model.eval()

# StoryDataset for AIMSDistill
class StoryDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        return input_ids, attention_mask

# ---------------------------
# TAB 2 UI
# ---------------------------
# =========================================================
# TAB 2: AIMS LLM
# =========================================================
# ---------------------------
# TAB 2: AIMS LLM with ModernBERT
# ---------------------------

# ---------------------------
# TAB 2: AIMS LLM
# ---------------------------
with tab3:
    st.header("🤖 Insight on Supply Chains with AIMSDistill + Gemini")
    st.info("Upload a text file or summarize filtered Survivor Dashboard entries.")

    # ---------------------------
    # Configure Gemini
    # ---------------------------
    import google.generativeai as genai
    genai.configure(api_key=st.secrets["genai"]["api_key"])
    g_model = genai.GenerativeModel("gemini-2.5-flash")

    summarize_choice = st.radio(
        "Choose source to summarize:",
        ["Upload Text File", "Filtered Survivor Dashboard Entries"]
    )

    # ---------------------------
    # Initialize AIMSDistill model & tokenizer
    # ---------------------------
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class AimsDistillModel(nn.Module):
        def __init__(self, tokenizer, model_name, dropout=0.0):
            super().__init__()
            self.bert = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
            self.dropout = nn.Dropout(p=dropout)
            self.classifier = nn.Linear(self.bert.config.hidden_size, 11)

        def forward(self, input_ids, attention_mask=None):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            logits = self.dropout(self.classifier(pooled_output))
            return logits

    model = AimsDistillModel(tokenizer, model_name).to(device)
    model.eval()

    # ---------------------------
    # StoryDataset for AIMSDistill
    # ---------------------------
    class StoryDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            inputs = self.tokenizer(
                text, return_tensors="pt", padding="max_length",
                truncation=True, max_length=self.max_length
            )
            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)
            return input_ids, attention_mask

    # ---------------------------
    # Option 1: Uploaded file
    # ---------------------------
    if summarize_choice == "Upload Text File":
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if uploaded_file is not None:
            text_data = uploaded_file.read().decode("utf-8")
            if st.button("Generate AIMSDistill Summary from File"):
                st.info("Processing uploaded file...")
                try:
                    # Sentence segmentation
                    abbrev = r'(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|U\.S\.A|i\.e|e\.g|St|a\.m|p\.m|Fig|No)\.'
                    text = re.sub(abbrev, lambda x: x.group().replace('.', '<prd>'), text_data)
                    text = re.sub(r'(\d)\.(\d)', r'\1<prd>\2', text)
                    text = re.sub(r'([.!?])\s+', r'\1<stop>', text)
                    sentences = [s.replace('<prd>', '.').strip() for s in text.split('<stop>') if s.strip()]

                    # Predict risk sentences with AIMSDistill
                    dataset = StoryDataset(sentences, tokenizer, max_length=60)
                    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
                    risk_sentences = []
                    with torch.no_grad():
                        for batch in dataloader:
                            input_ids, attention_mask = [b.to(device) for b in batch]
                            logits = model(input_ids=input_ids, attention_mask=attention_mask)
                            preds = (logits[:, 6] > 0.9).cpu().numpy()  # c3 = risk description
                            for i, pred in enumerate(preds):
                                if pred == 1:
                                    risk_sentences.append(sentences.pop(0))

                    if not risk_sentences:
                        st.warning("No risk description sentences detected.")
                        st.stop()

                    # Chunk sentences
                    def chunk_sentences(sent_list, chunk_size=80):
                        for i in range(0, len(sent_list), chunk_size):
                            yield sent_list[i:i+chunk_size]

                    chunks = list(chunk_sentences(risk_sentences))

                    # Summarize each chunk with Gemini
                    chunk_summaries = []
                    for chunk in chunks:
                        prompt = f"""
You are given a collection of sentences classified as "risk descriptions".
Summarize them clearly, concisely, and factually in Canadian English.
Do not use poetic or figurative language.

Sentences:
{chr(10).join(chunk)}
"""
                        response = g_model.generate_content(prompt)
                        chunk_summaries.append(response.text)

                    # Merge summaries
                    final_prompt = f"""
You are given multiple summaries of batches of sentences.
Merge them into one coherent, factual summary.
Do not be poetic; emphasise repeated risks.

Summaries:
{chr(10).join(chunk_summaries)}
"""
                    final_summary = g_model.generate_content(final_prompt).text
                    st.success("✅ Summary generated from uploaded file!")
                    st.text_area("📄 Summary", value=final_summary, height=400)

                except Exception as e:
                    st.error(f"❌ Error processing uploaded file: {e}")

    # ---------------------------
    # Option 2: Filtered Tab 1 entries
    # ---------------------------
    elif summarize_choice == "Filtered Survivor Dashboard Entries":
        if 'filtered_df' in locals() and not filtered_df.empty:
            if st.button("Generate AIMS Distill Summary From Survivor Dashboard"):
                st.info("Processing Filtered Survivor Dashboard Entries...")
                try:
                    combined_text = "\n\n".join(
                        f"{row.get('TITLE','')} - {row.get('DESCRIPTION','')} "
                        f"{'[Link]('+row.get('LINK','')+')' if row.get('LINK') else ''}"
                        for _, row in filtered_df.iterrows()
                    )
                    sentences = combined_text.split(". ")

                    # Chunk sentences
                    def chunk_sentences(sent_list, chunk_size=80):
                        for i in range(0, len(sent_list), chunk_size):
                            yield sent_list[i:i+chunk_size]

                    chunks = list(chunk_sentences(sentences))

                    # Summarize each chunk
                    chunk_summaries = []
                    for chunk in chunks:
                        prompt = f"""
You are given a collection of sentences classified as "risk descriptions".
Summarize them clearly, concisely, and factually in Canadian English.
Do not use poetic or figurative language.

Sentences:
{chr(10).join(chunk)}
"""
                        response = g_model.generate_content(prompt)
                        chunk_summaries.append(response.text)

                    # Merge summaries
                    final_prompt = f"""
You are given multiple summaries of batches of sentences.
Merge them into one coherent, factual summary.
Do not be poetic; emphasise repeated risks.

Summaries:
{chr(10).join(chunk_summaries)}
"""
                    final_summary = g_model.generate_content(final_prompt).text
                    st.success("✅ Summary Generated From Filtered Survivor Dashboard Entries!")
                    st.text_area("📄 Summary", value=final_summary, height=400)

                except Exception as e:
                    st.error(f"❌ Error processing Filtered Tab 1 Entries: {e}")
        else:
            st.warning("No Filtered Tab 1 Entries Available To Summarize.")

with tab1:
    st.header("📖 Dashboard Documentation & Instructions")

    doc_text = """
    # Survivor Dashboard + AIMS LLM Documentation

    ## Overview
    This dashboard allows users to explore survivor stories of modern slavery and human trafficking,
    and generate summarized insights using AIMSDistill + Gemini.

    ## Features
    - Filter stories by Region, Industry, Category, Title, or Links.
    - Generate AI-assisted summaries of filtered content.
    - Access educational content responsibly.

    ## User Instructions
    - Enter a Google Sheet URL (public or private) in the sidebar to load stories.
    - Use filters to narrow down data.
    - Click the "Generate Summary" button to see AI insights.
    - Review sources responsibly via the included links.

    ## Warnings
    - Content is sensitive. Some stories may be distressing.
    - Intended for educational and awareness purposes only.

    ## API / Model Details
    - Uses `bert-base-uncased` for AIMSDistill predictions.
    - Summarization powered by Gemini 2.5 via `google.generativeai`.
    - Ensure API keys are stored securely in Streamlit secrets.

    ## Additional Resources
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [Transformers Documentation](https://huggingface.co/docs/transformers/)
    - [Google Generative AI Documentation](https://developers.generativeai.google/)
    """

    st.markdown(doc_text)

    # Create a downloadable Markdown file
    markdown_bytes = doc_text.encode('utf-8')
    st.download_button(
        label="📥 Download Documentation (Markdown)",
        data=markdown_bytes,
        file_name="Survivor_Dashboard_Documentation.md",
        mime="text/markdown"
    )

    # Optional: Download as PDF
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "This is the documentation for the dashboard...")
        

        # Save PDF to BytesIO
        pdf_bytes = pdf.output(dest="S").encode("latin1")  # dest="S" returns as string, encode to bytes

        st.download_button(
            label="📥 Download Documentation (PDF)",
            data=pdf_bytes,
            file_name="Survivor_Dashboard_Documentation.pdf",
            mime="application/pdf"
        )
    except ImportError:
        st.warning("Install `fpdf` to enable PDF download: pip install fpdf")