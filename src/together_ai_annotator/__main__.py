# __main__.py (inside src/together_ai_annotator/)
import argparse
from dotenv import load_dotenv
from .together_ai_annotator import annotate_code_with_together_ai

load_dotenv()

def annotate_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    print(f"🧠 Annotating: {filepath}")
    annotated = annotate_code_with_together_ai(code)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(annotated)

    print(f"✅ Overwritten: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Annotate with Together.ai")
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    annotate_file(args.file)

if __name__ == "__main__":
    main()
