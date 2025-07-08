"""
Sample Python file containing known CWE patterns for use in static analysis
and machine learning SAST model training.

Each block is annotated with the relevant CWE ID, description, and indicators.
"""

# ⚠️ CWE-89: Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')
# ⚠️ SAST Risk: Using string concatenation to build SQL queries introduces injection vectors
# 🧠 ML Signal: Presence of unparameterized raw SQL in function body
def unsafe_sql_query(user_input):
    import sqlite3
    conn = sqlite3.connect("example.db")
    cursor = conn.cursor()
    # Vulnerable SQL statement
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    cursor.execute(query)
    conn.close()

# ✅ Safe Alternative: Parameterized query to prevent injection
def safe_sql_query(user_input):
    import sqlite3
    conn = sqlite3.connect("example.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE name = ?", (user_input,))
    conn.close()

# ⚠️ CWE-22: Path Traversal: '../' in filename allows access outside base directory
# ⚠️ SAST Risk: User-controlled path can access sensitive files
# 🧠 ML Signal: Use of open() with user input that is not sanitised or validated
def load_user_file(filename):
    with open(f"/user/data/{filename}", "r") as f:
        return f.read()

# ✅ Safe Alternative: Enforce a whitelist or use secure path joining
def load_user_file_safe(filename):
    from pathlib import Path
    base = Path("/user/data").resolve()
    target = (base / filename).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError("Invalid path")
    with open(target, "r") as f:
        return f.read()

# ⚠️ CWE-798: Use of Hard-coded Credentials
# ⚠️ SAST Risk: Static secret in source allows credential leakage
# 🧠 ML Signal: Hard-coded string assigned to password or key variable
def connect_to_service():
    password = "SuperSecret123!"  # hardcoded password
    return f"Connected using password: {password}"

# ✅ Safe Alternative: Retrieve secrets from environment or secrets manager
import os
def connect_to_service_safe():
    password = os.environ.get("SERVICE_PASSWORD")
    if not password:
        raise EnvironmentError("Missing password")
    return f"Connected using password: {password}"
