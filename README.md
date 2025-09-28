
# Bank Statement PDF Parser Agent

This project is an AI-powered custom parser agent designed to extract structured data from bank statement PDFs.
It converts the unstructured PDF data into a structured pandas.DataFrame matching a predefined CSV schema and validates the result.

## 📂 Folder Structure

```
ai-agent-challenge/
│
├── custom_parsers/
│   ├── icici_parser.py          # Parser logic implementation
│
├── data/
│   ├── icici/
│       ├── icici_sample.pdf     # Input sample PDF
│       ├── icici_expected.csv   # Expected CSV output
│
├── README.md
├── requirements.txt
```

## ⚙ Installation

Follow these steps to set up and run the parser:

1. **Clone the repository**
   ```bash
   git clone https://github.com/apurv-korefi/ai-agent-challenge.git
   cd ai-agent-challenge
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate       # Windows
   ```

3. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 Usage

Run the parser and validate the output against the expected CSV:

```bash
python -c "from custom_parsers.icici_parser import parse; import pandas as pd; df_actual=parse('data/icici/icici_sample.pdf'); df_expected=pd.read_csv('data/icici/icici_expected.csv'); assert df_actual.equals(df_expected), 'Parsed DataFrame does not match expected CSV'; print('Parsing successful — Data matches expected CSV.')"
```

## 📌 How It Works

The agent performs the following steps:

1. **PDF Input:** Reads the bank statement PDF file.
2. **Text Extraction:** Uses PDF parsing tools (e.g., PDFMiner) to extract raw text.
3. **Parsing Logic:** Applies custom parsing to extract relevant columns (Date, Description, Debit Amount, Credit Amount, Balance).
4. **Data Structuring:** Converts parsed data into a pandas.DataFrame.
5. **Validation:** Compares the parsed output against the expected CSV using DataFrame.equals.
6. **Output:** Confirms parsing success or throws an error if there is a mismatch.

## 🧠 Agent Workflow Diagram

```
+------------+      +-----------------+      +--------------------+
| Input PDF  | ---> | Text Extraction | ---> | Parsing Logic      |
+------------+      +-----------------+      +--------------------+
                                                       |
                                                       v
                                            +----------------------+
                                            | Structured DataFrame|
                                            +----------------------+
                                                       |
                                                       v
                                            +----------------------+
                                            | CSV Comparison       |
                                            +----------------------+
                                                       |
                                                       v
                                            +----------------------+
                                            | Output Result        |
                                            +----------------------+
```

## 📂 Example

Suppose you have:
- PDF: data/icici/icici_sample.pdf
- Expected CSV: data/icici/icici_expected.csv

Run the following command:
```bash
python -c "from custom_parsers.icici_parser import parse; import pandas as pd; df_actual=parse('data/icici/icici_sample.pdf'); df_expected=pd.read_csv('data/icici/icici_expected.csv'); assert df_actual.equals(df_expected), 'Parsed DataFrame does not match expected CSV'; print('Parsing successful — Data matches expected CSV.')"
```

Expected output:
```
Parsing successful — Data matches expected CSV.
```

## 📌 Notes

- The parser is custom-built for ICICI bank statements; you may need to adjust logic for other banks.
- Ensure the PDF structure matches the parser’s expectations.
- Tested on Python 3.8+.

## 📜 License
MIT License © 2025

## 🧾 Author
Srujan Patel — Final Year Engineering Student
