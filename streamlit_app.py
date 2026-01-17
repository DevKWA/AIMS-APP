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
# Streamlit page config (must be first Streamlit call)
# ---------------------------
st.set_page_config(layout="wide", page_title="Survivor Dashboard + AIMS LLM")

st.warning(
    """
⚠️ **Content Warning:** This dashboard contains real survivor stories of modern slavery and human trafficking. Some descriptions may be distressing.

All stories are from open, public, approved sources.
Each story includes a link to the full source. Please consult the original source.

Please use this tool responsibly, with respect and sensitivity toward survivors. Younger audiences, survivors, or those who may be triggered by descriptions of violence, exploitation, or abuse should proceed with caution.

These stories are shared for educational and awareness purposes only and should be engaged with thoughtfully, keeping in mind the dignity and resilience of survivors.
"""
)

# ---------------------------
# Global CSS (OS/browser theme via prefers-color-scheme)
# ---------------------------
st.markdown(
    """
    <style>
      .streamlit-expander { width: 100% !important; }

      /* Results banner (OS/browser theme) */
      .result-banner {
        padding: 10px 14px;
        border-radius: 8px;
        border-left: 6px solid;
        margin-bottom: 0.5rem;
        border: 1px solid;
      }
      .result-banner h3 {
        margin: 0;
        text-align: left;
        color: inherit;
      }

      /* Story cards (OS/browser theme) */
      .story-card-os {
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
        border: 1px solid;
      }
      .story-card-os strong { color: inherit; }
      .story-card-os a { text-decoration: underline; }

      @media (prefers-color-scheme: light) {
        .result-banner, .story-card-os {
          background: #ffffff;
          color: #111111;
          border-color: #e5e7eb;
        }
        .result-banner { border-left-color: #111111; }
        .story-card-os a { color: #1d4ed8; }
      }

      @media (prefers-color-scheme: dark) {
        .result-banner, .story-card-os {
          background: #111827;
          color: #f9fafb;
          border-color: #374151;
        }
        .result-banner { border-left-color: #f9fafb; }
        .story-card-os a { color: #93c5fd; }
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Banner images
# ---------------------------
st.image("assets/AULA_HORIZONTAL_GREEN_BANNER.png", width="stretch")
st.image("assets/survivor_dashboard_banner.png", width="stretch")

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(
    ["📖 Dashboard Documentation & Instructions", "📊 Survivor Dashboard", "🤖 Insight on Supply Chains (from AIMS)"]
)

# ---------------------------
# Cached functions for performance
# ---------------------------
@st.cache_resource
def load_tokenizer_and_model(model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer, model


@st.cache_resource
def load_aims_model(_tokenizer, model_name="bert-base-uncased", dropout=0.0):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AimsDistillModel(_tokenizer, model_name, dropout=dropout).to(device)
    model.eval()
    return model, device


@st.cache_data
def load_google_sheet(sheet_url, _creds):
    # IMPORTANT: use _creds (not a global 'creds')
    gc = gspread.authorize(_creds)
    sh = gc.open_by_url(sheet_url)
    worksheets = {ws.title: pd.DataFrame(ws.get_all_records()) for ws in sh.worksheets()}
    return worksheets


@st.cache_data
def preprocess_text(text_data: str):
    abbrev = r"(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|U\.S\.A|i\.e|e\.g|St|a\.m|p\.m|Fig|No)\."
    text = re.sub(abbrev, lambda x: x.group().replace(".", "<prd>"), text_data)
    text = re.sub(r"(\d)\.(\d)", r"\1<prd>\2", text)
    text = re.sub(r"([.!?])\s+", r"\1<stop>", text)
    sentences = [s.replace("<prd>", ".").strip() for s in text.split("<stop>") if s.strip()]
    return sentences


@st.cache_data
def make_dataloader(sentences, tokenizer, max_length=60, batch_size=32):
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
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)
            return input_ids, attention_mask

    dataset = StoryDataset(sentences, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@st.cache_data
def chunk_sentences(sent_list, chunk_size=80):
    return [sent_list[i : i + chunk_size] for i in range(0, len(sent_list), chunk_size)]


@st.cache_data
def summarize_chunk(chunk, _g_model):
    prompt = f"""
You are given a collection of sentences classified as "risk descriptions".
Summarize them clearly, concisely, and factually in Canadian English.
Do not use poetic or figurative language.

Sentences:
{chr(10).join(chunk)}
"""
    response = _g_model.generate_content(prompt)
    return response.text


@st.cache_data
def merge_summaries(chunk_summaries, _g_model):
    final_prompt = f"""
You are given multiple summaries of batches of sentences.
Merge them into one coherent, factual summary.
Do not be poetic; emphasise repeated risks.

Summaries:
{chr(10).join(chunk_summaries)}
"""
    final_summary = _g_model.generate_content(final_prompt).text
    return final_summary


# ---------------------------
# TAB 2: Survivor Dashboard
# ---------------------------
with tab2:
    st.sidebar.header("Data Source Options")
    DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1zDp2_x2U631K9Hbc0kt0GL59aUTCfz07snIzav0-HsQ/edit?gid=801915987#gid=801915987"
    sheet_url = st.sidebar.text_input("📄 Google Sheet URL (public or private)", value=DEFAULT_SHEET_URL)

    df = None
    if sheet_url:
        try:
            SCOPES = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive.file",
            ]
            creds = Credentials.from_service_account_info(st.secrets["google_service_account"], scopes=SCOPES)
            worksheets = load_google_sheet(sheet_url, creds)
            worksheet_names = list(worksheets.keys())
            worksheet_name = st.sidebar.selectbox("📑 Choose a worksheet/tab", worksheet_names)
            df = worksheets[worksheet_name]
            st.sidebar.success(f"✅ Loaded worksheet: {worksheet_name}")
        except Exception as ex:
            st.sidebar.error(f"❌ Error loading sheet: {ex}")
            df = None

    if df is None or df.empty:
        st.warning("Please provide a Google Sheet URL to continue.")
        st.stop()

    st.sidebar.header("🔍 Filters")
    filtered_for_filters = df.copy()

    # Region filter
    regions = sorted(filtered_for_filters["REGION"].dropna().unique()) if "REGION" in df.columns else []
    selected_region = st.sidebar.selectbox("🌍 Select Region", options=["All"] + regions)
    if selected_region != "All":
        filtered_for_filters = filtered_for_filters[filtered_for_filters["REGION"] == selected_region]

    # Industry filter
    industries = (
        sorted(filtered_for_filters["INDUSTRY"].dropna().unique()) if "INDUSTRY" in filtered_for_filters.columns else []
    )
    selected_industry = st.sidebar.selectbox("🏭 Select Industry", options=["All"] + industries)
    if selected_industry != "All":
        filtered_for_filters = filtered_for_filters[filtered_for_filters["INDUSTRY"] == selected_industry]

    # Category filter
    categories = (
        sorted(filtered_for_filters["CATEGORY"].dropna().unique()) if "CATEGORY" in filtered_for_filters.columns else []
    )
    selected_category = st.sidebar.selectbox("📂 Select Category", options=["All"] + categories)
    if selected_category != "All":
        filtered_for_filters = filtered_for_filters[filtered_for_filters["CATEGORY"] == selected_category]

    # Title filter
    titles = sorted(filtered_for_filters["TITLE"].dropna().unique()) if "TITLE" in filtered_for_filters.columns else []
    selected_title = st.sidebar.selectbox("📌 Select Title", options=["All"] + titles)
    if selected_title != "All":
        filtered_for_filters = filtered_for_filters[filtered_for_filters["TITLE"] == selected_title]

    # Link filter
    link_filter = st.sidebar.radio("🔗 Filter by Links", options=["All", "With Links", "Without Links"])
    filtered_df = filtered_for_filters.copy()
    if link_filter == "With Links" and "LINK" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["LINK"].notna() & (filtered_df["LINK"] != "")]
    elif link_filter == "Without Links" and "LINK" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["LINK"].isna() | (filtered_df["LINK"] == "")]

    if filtered_df.empty:
        st.warning("❌ No results match the selected filters.")
        st.stop()

    # Results header (OS/browser theme)
    st.markdown(
        f"""
        <div class="result-banner">
          <h3>
            📊 Showing Results: Region: {selected_region} | Industry: {selected_industry} | Category: {selected_category} | Title: {selected_title} | Links: {link_filter}
          </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display rows with emojis
    region_icons = {"Africa": "🌍", "Europe": "🌎", "Asia": "🌏"}
    industry_icons = {"Oil & Gas": "🏭", "Agriculture": "🌾", "Technology": "💻"}
    category_icons = {"Labor": "👷", "Human Rights": "✊", "Safety": "🛡️"}

    for _, row in filtered_df.iterrows():
        region_icon = region_icons.get(row.get("REGION", ""), "🌐")
        industry_icon = industry_icons.get(row.get("INDUSTRY", ""), "🏢")
        category_icon = category_icons.get(row.get("CATEGORY", ""), "📂")

        link = row.get("LINK", "")
        # IMPORTANT: no inline color so CSS can theme it
        link_html = f"<a href='{link}' target='_blank' rel='noopener noreferrer'>Read More</a>" if link else ""

        st.markdown(
            f"""
            <div class="story-card-os">
              <strong>{region_icon} {industry_icon} {category_icon} {row.get('TITLE','Untitled')}</strong><br>
              {row.get('DESCRIPTION','')}<br>
              {link_html}
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------
# TAB 3: AIMSDistill + Gemini
# ---------------------------
with tab3:
    st.header("🤖 Insight on Supply Chains with AIMSDistill + Gemini")
    st.info("Upload a text file or summarize filtered Survivor Dashboard entries.")

    genai.configure(api_key=st.secrets["genai"]["api_key"])
    g_model = genai.GenerativeModel("gemini-2.5-flash")

    summarize_choice = st.radio("Choose source to summarize:", ["Upload Text File", "Filtered Survivor Dashboard Entries"])

    tokenizer, _ = load_tokenizer_and_model()
    model, device = load_aims_model(tokenizer)

    if summarize_choice == "Upload Text File":
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if uploaded_file is not None:
            text_data = uploaded_file.read().decode("utf-8")
            if st.button("Generate AIMSDistill Summary from File"):
                st.info("Processing uploaded file...")
                try:
                    sentences = preprocess_text(text_data)
                    dataloader = make_dataloader(sentences, tokenizer)

                    risk_sentences = []
                    idx = 0

                    with torch.no_grad():
                        for batch in dataloader:
                            input_ids, attention_mask = [b.to(device) for b in batch]
                            logits = model(input_ids=input_ids, attention_mask=attention_mask)
                            preds = (logits[:, 6] > 0.9).cpu().numpy()

                            # Track indices safely (avoid popping from sentences while iterating)
                            batch_size = len(preds)
                            for j in range(batch_size):
                                if idx + j < len(sentences) and preds[j] == 1:
                                    risk_sentences.append(sentences[idx + j])
                            idx += batch_size

                    if not risk_sentences:
                        st.warning("No risk description sentences detected.")
                        st.stop()

                    chunks = chunk_sentences(risk_sentences)
                    chunk_summaries = [summarize_chunk(chunk, g_model) for chunk in chunks]
                    final_summary = merge_summaries(chunk_summaries, g_model)

                    st.success("✅ Summary generated from uploaded file!")
                    st.text_area("📄 Summary", value=final_summary, height=400)
                except Exception as e:
                    st.error(f"❌ Error processing uploaded file: {e}")

    elif summarize_choice == "Filtered Survivor Dashboard Entries":
        if "filtered_df" in locals() and not filtered_df.empty:
            if st.button("Generate AIMS Distill Summary From Survivor Dashboard"):
                st.info("Processing Filtered Survivor Dashboard Entries...")
                try:
                    combined_text = "\n\n".join(
                        f"{row.get('TITLE','')} - {row.get('DESCRIPTION','')} {row.get('LINK','')}"
                        for _, row in filtered_df.iterrows()
                    )

                    sentences = preprocess_text(combined_text)
                    chunks = chunk_sentences(sentences)
                    chunk_summaries = [summarize_chunk(chunk, g_model) for chunk in chunks]
                    final_summary = merge_summaries(chunk_summaries, g_model)

                    st.success("✅ Summary Generated From Filtered Survivor Dashboard Entries!")
                    st.text_area("📄 Summary", value=final_summary, height=400)
                except Exception as e:
                    st.error(f"❌ Error processing Filtered Tab 1 Entries: {e}")
        else:
            st.warning("No Filtered Tab 1 Entries Available To Summarize.")

# ---------------------------
# TAB 1: Documentation
# ---------------------------
with tab1:
    st.header("📖 Dashboard Documentation & Instructions")

    doc_text = """
# Survivor Dashboard + AIMS LLM Documentation

## Overview
The Survivor Dashboard is designed to help users explore survivor stories of modern slavery and human trafficking.
It allows filtering by multiple dimensions and generating AI-assisted summaries of risk-related content using **AIMSDistill** + **Gemini 2.5**.

**Important Note:** Some stories contain sensitive content. Please read responsibly and ensure content is appropriate for your audience.
"""

    st.markdown(doc_text)
    st.download_button(
        label="📥 Download Documentation (Markdown)",
        data=doc_text.encode("utf-8"),
        file_name="Survivor_Dashboard_Documentation.md",
        mime="text/markdown",
    )

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    left_margin = 15 * mm
    top_margin = height - 15 * mm
    line_height = 12
    y = top_margin

    c.setFont("Helvetica", 12)

    for line in doc_text.split("\n"):
        safe_line = "".join([ch if ord(ch) < 128 else "" for ch in line])
        if y < 15:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = top_margin
        c.drawString(left_margin, y, safe_line)
        y -= line_height

    c.save()
    buffer.seek(0)
    pdf_bytes = buffer.read()

    st.download_button(
        label="📥 Download Documentation (PDF)",
        data=pdf_bytes,
        file_name="Survivor_Dashboard.pdf",
        mime="application/pdf",
    )