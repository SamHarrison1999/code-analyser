"""
Safe Python code file used as a baseline for static analysis and ML training.
Contains no known CWE violations or risky patterns.
"""

# ✅ Best Practice: Use parameterised queries for database access
def fetch_user_by_email(db_conn, email):
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    return cursor.fetchone()

# ✅ Best Practice: Validate and restrict file paths before access
def read_config_file(config_name):
    from pathlib import Path
    config_dir = Path("/etc/myapp/configs").resolve()
    config_path = (config_dir / config_name).resolve()
    if not str(config_path).startswith(str(config_dir)):
        raise ValueError("Invalid config path")
    with open(config_path, "r") as f:
        return f.read()

# ✅ Best Practice: Avoid hardcoded secrets; use environment variables
import os
def get_api_key():
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY not set in environment")
    return api_key

# ✅ Best Practice: Sanitize user input when writing to files
def save_user_notes(username, notes):
    from pathlib import Path
    notes_dir = Path("/user/notes").resolve()
    safe_filename = f"{username.replace('..', '').replace('/', '')}.txt"
    file_path = (notes_dir / safe_filename).resolve()
    if not str(file_path).startswith(str(notes_dir)):
        raise ValueError("Unsafe file path")
    with open(file_path, "w") as f:
        f.write(notes)

# ✅ Best Practice: Avoid use of eval or exec
def calculate_expression_safe(expr: str):
    # Use ast.literal_eval for safe evaluation
    import ast
    try:
        return ast.literal_eval(expr)
    except (ValueError, SyntaxError):
        return None
