"""
Deliberately vulnerable Python file for use in SAST tooling and ML model training.
Each function demonstrates a known CWE vulnerability pattern.
"""

# ⚠️ CWE-78: OS Command Injection
# ⚠️ SAST Risk: Passing unsanitised user input to system shell
# 🧠 ML Signal: Use of os.system or subprocess with interpolated user input
import os
def delete_user_dir(username):
    os.system(f"rm -rf /home/{username}")  # dangerous

# ⚠️ CWE-79: Cross-Site Scripting (XSS) in a Python web context (e.g., Flask)
# ⚠️ SAST Risk: Returning unsanitised input directly to HTML
from flask import Flask, request
app = Flask(__name__)
@app.route("/greet")
def greet():
    name = request.args.get("name")
    return f"<h1>Hello {name}</h1>"  # unsanitised HTML output

# ⚠️ CWE-94: Code Injection via eval()
# ⚠️ SAST Risk: Arbitrary code execution from user input
def eval_user_input(data):
    return eval(data)  # never safe

# ⚠️ CWE-732: Incorrect Permission Assignment for Critical Resource
# ⚠️ SAST Risk: Writing files with overly permissive modes
def write_secret_file(data):
    with open("secret.txt", "w") as f:
        f.write(data)
    os.chmod("secret.txt", 0o777)  # too permissive

# ⚠️ CWE-502: Deserialization of Untrusted Data
# ⚠️ SAST Risk: Loading pickled objects from user input
import pickle
def load_data(serialised):
    return pickle.loads(serialised)  # untrusted input

# ⚠️ CWE-611: XML External Entity (XXE) Injection
# ⚠️ SAST Risk: Using XML parsers without disabling external entity loading
from lxml import etree
def parse_xml(xml_string):
    return etree.fromstring(xml_string)  # no parser safety

# ⚠️ CWE-327: Use of Weak Cryptographic Algorithm
# ⚠️ SAST Risk: MD5 is not suitable for cryptographic use
import hashlib
def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()  # insecure

# ⚠️ CWE-798: Hardcoded Credentials
# ⚠️ SAST Risk: Credentials embedded in source code
def connect_admin():
    return login("admin", "admin123")  # static creds

def login(user, pwd):
    return f"Logged in as {user}"
