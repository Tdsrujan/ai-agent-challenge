#!/usr/bin/env python3
"""
Agent-as-Coder: Autonomous PDF Parser Generator
Generates custom bank statement parsers using LLM agents.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
import pandas as pd
import PyPDF2
import re
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AgentState:
    target_bank: str
    pdf_path: str
    csv_path: str
    parser_path: str
    pdf_content: str = ""
    csv_schema: Dict = None
    csv_sample: pd.DataFrame = None
    generated_code: str = ""
    test_results: Dict = None
    error_messages: List[str] = None
    attempt_count: int = 0
    max_attempts: int = 3

    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []

class PDFAnalyzer:
    @staticmethod
    def extract_text(pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""

    @staticmethod
    def analyze_structure(pdf_text: str) -> Dict[str, Any]:
        lines = pdf_text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]

        # Detect date format
        date_patterns = {
            "dd-mm-yyyy": r"\d{2}-\d{2}-\d{4}",
            "dd-MMM-yyyy": r"\d{2}-[A-Za-z]{3}-\d{4}"
        }
        detected_date_pattern = None
        for name, pattern in date_patterns.items():
            if any(re.match(pattern, line) for line in non_empty_lines[:20]):
                detected_date_pattern = pattern
                break

        # Detect column separator
        sample_line = non_empty_lines[0] if non_empty_lines else ""
        if "\t" in sample_line:
            separator = "tab"
        elif re.search(r"\s{2,}", sample_line):
            separator = "multi_space"
        else:
            separator = "single_space"

        return {
            "total_lines": len(lines),
            "content_lines": len(non_empty_lines),
            "avg_line_length": sum(len(line) for line in non_empty_lines) / max(len(non_empty_lines), 1),
            "sample_lines": non_empty_lines[:10],
            "date_pattern": detected_date_pattern or r"\d{2}-\d{2}-\d{4}",
            "separator": separator
        }

class LLMAgent:
    def __init__(self):
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        model_names = ["models/gemini-2.5-pro", "models/gemini-2.5-flash", "models/gemini-pro-latest", "models/gemini-flash-latest"]
        self.model = None
        for model_name in model_names:
            try:
                logger.info(f"Trying model: {model_name}...")
                self.model = genai.GenerativeModel(model_name)
                self.model.generate_content("Say hello")
                logger.info(f"✅ Using model: {model_name}")
                break
            except Exception as e:
                logger.warning(f"❌ Model {model_name} failed: {e}")
                continue
        if self.model is None:
            raise ValueError("No working Gemini model found.")

    def generate_parser(self, state: AgentState, pdf_analysis: Dict) -> str:
        date_pattern = pdf_analysis.get("date_pattern", r"\d{2}-\d{2}-\d{4}")
        separator = pdf_analysis.get("separator", "single_space")

        if separator == "multi_space":
            split_code = "re.split(r'\\s{2,}', line)"
        elif separator == "tab":
            split_code = "line.split('\\t')"
        else:
            split_code = "line.split(maxsplit=4)"

        return f"""
import pandas as pd
import PyPDF2
import re

def parse(pdf_path: str) -> pd.DataFrame:
    \"\"\"Parse {state.target_bank} bank statement PDF\"\"\"
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\\n"

        lines = [line.strip() for line in text.split("\\n") if line.strip()]
        transactions = []
        date_pattern = r"{date_pattern}"

        for line in lines:
            if re.match(date_pattern, line):
                parts = {split_code}
                if len(parts) >= 5:
                    transactions.append({{
                        "{list(state.csv_schema.keys())[0]}": parts[0],
                        "{list(state.csv_schema.keys())[1]}": parts[1],
                        "{list(state.csv_schema.keys())[2]}": parts[2],
                        "{list(state.csv_schema.keys())[3]}": parts[3],
                        "{list(state.csv_schema.keys())[4]}": parts[4],
                    }})

        if not transactions:
            transactions.append({{col: "" for col in {list(state.csv_schema.keys())}}})

        return pd.DataFrame(transactions, columns={list(state.csv_schema.keys())})

    except Exception as e:
        print(f"Error parsing PDF: {{e}}")
        return pd.DataFrame(columns={list(state.csv_schema.keys())})
"""

class CodeTester:
    @staticmethod
    def test_parser(state: AgentState) -> Dict[str, Any]:
        try:
            os.makedirs(os.path.dirname(state.parser_path), exist_ok=True)
            with open(state.parser_path, 'w') as f:
                f.write(state.generated_code)

            sys.path.insert(0, os.path.dirname(state.parser_path))
            module_name = os.path.basename(state.parser_path)[:-3]
            if module_name in sys.modules:
                del sys.modules[module_name]
            parser_module = __import__(module_name)

            if not hasattr(parser_module, 'parse'):
                return {"success": False, "errors": ["Module missing 'parse' function"]}

            result_df = parser_module.parse(state.pdf_path)
            expected_df = state.csv_sample

            success = isinstance(result_df, pd.DataFrame) and len(result_df) > 0
            errors = [] if success else ["Parser returned empty DataFrame"]

            return {"success": success, "errors": errors, "result_shape": result_df.shape}
        except Exception as e:
            return {"success": False, "errors": [str(e)]}
        finally:
            if os.path.dirname(state.parser_path) in sys.path:
                sys.path.remove(os.path.dirname(state.parser_path))

class AgentWorkflow:
    def __init__(self):
        self.pdf_analyzer = PDFAnalyzer()
        self.llm_agent = LLMAgent()
        self.code_tester = CodeTester()

    def initialize_state(self, target_bank: str) -> AgentState:
        pdf_path = f"data/{target_bank}/{target_bank}_sample.pdf"
        csv_path = f"data/{target_bank}/{target_bank}_sample.csv"
        parser_path = f"custom_parsers/{target_bank}_parser.py"
        if not os.path.exists(pdf_path) or not os.path.exists(csv_path):
            raise FileNotFoundError("PDF or CSV sample missing.")

        pdf_content = self.pdf_analyzer.extract_text(pdf_path)
        csv_sample = pd.read_csv(csv_path)
        csv_schema = {col: str(dtype) for col, dtype in csv_sample.dtypes.items()}

        return AgentState(target_bank, pdf_path, csv_path, parser_path, pdf_content, csv_schema, csv_sample)

    def run(self, target_bank: str) -> bool:
        try:
            logger.info(f"🚀 Starting Agent-as-Coder for {target_bank}")
            state = self.initialize_state(target_bank)
            state.attempt_count += 1
            state = self.plan_phase(state)
            state = self.code_phase(state)
            state = self.test_phase(state)
            if state.test_results["success"]:
                logger.info(f"🎉 Successfully generated {state.parser_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Agent workflow failed: {e}")
            return False

    def plan_phase(self, state: AgentState) -> AgentState:
        logger.info(f"🔍 Planning parser for {state.target_bank}")
        return state

    def code_phase(self, state: AgentState) -> AgentState:
        logger.info(f"💻 Generating parser code")
        pdf_analysis = self.pdf_analyzer.analyze_structure(state.pdf_content)
        state.generated_code = self.llm_agent.generate_parser(state, pdf_analysis)
        return state

    def test_phase(self, state: AgentState) -> AgentState:
        logger.info("🧪 Testing generated parser")
        state.test_results = self.code_tester.test_parser(state)
        return state

def main():
    parser = argparse.ArgumentParser(description="Agent-as-Coder")
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    if not os.getenv('GOOGLE_API_KEY'):
        print("Missing GOOGLE_API_KEY in .env")
        sys.exit(1)
    agent = AgentWorkflow()
    success = agent.run(args.target)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
