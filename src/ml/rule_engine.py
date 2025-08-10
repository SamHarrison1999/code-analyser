# --- file: src/ml/rule_engine.py ---
# Annotation: This lightweight rule engine aggregates reasons from AST heuristics and optional local linters (flake8, pyflakes, pydocstyle, pylint, bandit) and SonarLint CLI, and can also suggest a label per line from those reasons. It now includes robust parsers and a rules-first decision helper.
from __future__ import annotations

# Annotation: Standard library imports only; external tools are invoked via subprocess when available.
import ast
import re
import subprocess
import tempfile
import shutil
import json
import os
import sys
import html

# Annotation: Pathlib is used for robust path handling in temp file creation.
from pathlib import Path

# Annotation: Typing helps document inputs/outputs clearly.
from typing import Dict, List, Tuple, Iterable

# Annotation: Environment-controlled debug switch; set CODE_ANALYSER_RULE_DEBUG=1 (or RULE_DEBUG=1) to print raw tool outputs and parsed hits.
_DEBUG = (
    os.environ.get("CODE_ANALYSER_RULE_DEBUG") or os.environ.get("RULE_DEBUG") or "0"
).strip() not in ("", "0", "false", "False")


# Annotation: Small helper to print debug lines consistently without altering programme flow.
def _dbg(msg: str) -> None:
    if _DEBUG:
        print(f"[RULE-DEBUG] {msg}", file=sys.stderr)


# Annotation: Simple AST visitor that records reasons against line numbers for common risky constructs (Bandit-like).
class _HeuristicVisitor(ast.NodeVisitor):
    # Annotation: The constructor takes a list to append (lineno, reason) tuples.
    def __init__(self, out: List[Tuple[int, str]]):
        # Annotation: Store the output sink.
        self.out = out

    # Annotation: Flag use of eval() and other risky calls.
    def visit_Call(self, node: ast.Call):
        # Annotation: Handle attribute/identifier calls generically.
        try:
            # Annotation: Extract dotted name if possible.
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                parts = []
                cur = node.func
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                func_name = ".".join(reversed(parts))
            # Annotation: Heuristic checks mapped to Bandit/Sonar rules.
            if func_name == "eval":
                self.out.append(
                    (node.lineno, "Use of eval() on untrusted input allows code execution.")
                )
            if func_name == "exec":
                self.out.append(
                    (node.lineno, "exec() executes arbitrary code; avoid or sandbox strictly.")
                )
            if func_name.startswith("pickle.loads"):
                self.out.append(
                    (
                        node.lineno,
                        "Unpickling untrusted data is unsafe; prefer a safe serialisation format such as JSON.",
                    )
                )
            if func_name.startswith("yaml.load"):
                self.out.append(
                    (
                        node.lineno,
                        "yaml.load can be unsafe on untrusted input; use yaml.safe_load instead.",
                    )
                )
            if func_name.startswith("requests.") and any(
                (
                    isinstance(k, ast.keyword)
                    and k.arg == "verify"
                    and getattr(k.value, "value", None) is False
                )
                for k in node.keywords
            ):
                self.out.append(
                    (
                        node.lineno,
                        "TLS verification disabled (verify=False); this undermines HTTPS security.",
                    )
                )
        except Exception:
            # Annotation: Never let heuristics break parsing.
            pass
        # Annotation: Continue generic traversal.
        self.generic_visit(node)


# Annotation: Regex reasons that are quick to evaluate at line level and work even on invalid fragments (e.g., a lone 'return eval(x)').
_REGEX_REASONS: List[Tuple[re.Pattern, str]] = [
    # Annotation: Eval / Exec risks.
    (re.compile(r"\beval\s*\("), "Use of eval() on untrusted input allows code execution."),
    (re.compile(r"\bexec\s*\("), "exec() executes arbitrary code; avoid or sandbox strictly."),
    # Annotation: Known insecure APIs.
    (
        re.compile(r"\bpickle\.loads\s*\("),
        "Unpickling untrusted data is unsafe; prefer a safe serialisation format such as JSON.",
    ),
    (
        re.compile(r"\byaml\.load\s*\("),
        "yaml.load can be unsafe on untrusted input; use yaml.safe_load instead.",
    ),
    (
        re.compile(r"requests\.\w+\s*\([^)]*verify\s*=\s*False"),
        "TLS verification disabled (verify=False); this undermines HTTPS security.",
    ),
    (
        re.compile(r"subprocess\.\w+\s*\([^)]*shell\s*=\s*True"),
        "Shell=True enables command injection; avoid or validate inputs strictly.",
    ),
    # Annotation: Chained normalisation which may remove meaningful characters.
    (
        re.compile(r"\.strip\(\)\s*\.\s*lower\(\)"),
        "strip() and lower() may not be necessary for all inputs; they can discard meaningful whitespace or case.",
    ),
    (
        re.compile(r"\.lower\(\)\s*\.\s*strip\(\)"),
        "lower() and strip() in sequence may discard significant characters; confirm the requirement.",
    ),
]


# Annotation: Run a command and capture stdout; print raw output when debugging.
def _run(cmd: List[str], cwd: Path, name: str, timeout: int = 180) -> str:
    if not shutil.which(cmd[0]):
        _dbg(f"{name}: not found on PATH; skipping.")
        return ""
    _dbg(f"{name}: running -> {' '.join(cmd)} (cwd={cwd})")
    try:
        r = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except Exception as e:
        _dbg(f"{name}: failed to execute ({e}); skipping.")
        return ""
    _dbg(f"{name}: raw output start")
    for ln in r.stdout.splitlines():
        _dbg(f"{name}| {ln}")
    _dbg(f"{name}: raw output end")
    return r.stdout


# Annotation: flake8 parser: “file:line:col: CODE message”.
def _from_flake8(stdout: str) -> Dict[int, List[str]]:
    pat = re.compile(r"snippet\.py:(?P<line>\d+):\d+:\s*[A-Z]\d+\s+(?P<msg>.+)")
    hits: Dict[int, List[str]] = {}
    for line in stdout.splitlines():
        m = pat.search(line)
        if m:
            ln = int(m.group("line"))
            msg = m.group("msg").strip()
            hits.setdefault(ln, []).append(msg)
    for ln, msgs in sorted(hits.items()):
        for msg in msgs:
            _dbg(f"flake8: hit line {ln}: {msg}")
    return hits


# Annotation: pyflakes parser: “file:line: message”.
def _from_pyflakes(stdout: str) -> Dict[int, List[str]]:
    pat = re.compile(r"snippet\.py:(?P<line>\d+):\s*(?P<msg>.+)")
    hits: Dict[int, List[str]] = {}
    for line in stdout.splitlines():
        m = pat.search(line)
        if m:
            ln = int(m.group("line"))
            msg = m.group("msg").strip()
            hits.setdefault(ln, []).append(msg)
    for ln, msgs in sorted(hits.items()):
        for msg in msgs:
            _dbg(f"pyflakes: hit line {ln}: {msg}")
    return hits


# Annotation: pydocstyle parser: blocks “path:LINE …” followed by “DXXX: message”.
def _from_pydocstyle(stdout: str) -> Dict[int, List[str]]:
    hits: Dict[int, List[str]] = {}
    cur_line = 1
    hdr = re.compile(r"snippet\.py:(?P<line>\d+)")
    msg = re.compile(r"\bD\d{3}:\s*(?P<msg>.+)")
    for line in stdout.splitlines():
        m = hdr.search(line)
        if m:
            cur_line = int(m.group("line"))
        else:
            mm = msg.search(line)
            if mm:
                hits.setdefault(cur_line, []).append(mm.group("msg").strip())
    for ln, msgs in sorted(hits.items()):
        for m in msgs:
            _dbg(f"pydocstyle: hit line {ln}: {m}")
    return hits


# Annotation: pylint parser: “file:LINE(:COL)?: [CODE(name), func] message”.
def _from_pylint(stdout: str) -> Dict[int, List[str]]:
    pat = re.compile(
        r"snippet\.py:(?P<line>\d+)(?::\d+)?:\s*\[(?P<code>[A-Z]\d{4}[^]]*)\][^]]*\]\s*(?P<msg>.+)"
    )
    hits: Dict[int, List[str]] = {}
    for line in stdout.splitlines():
        m = pat.search(line)
        if m:
            ln = int(m.group("line"))
            txt = f"{m.group('code')}: {m.group('msg').strip()}"
            hits.setdefault(ln, []).append(txt)
    for ln, msgs in sorted(hits.items()):
        for m in msgs:
            _dbg(f"pylint: hit line {ln}: {m}")
    return hits


# Annotation: Bandit parser (JSON): call with -f json, then read results[].line_number + test_id + issue_text.
def _run_bandit_json(cwd: Path, tmpfile: Path) -> Dict[int, List[str]]:
    if not shutil.which("bandit"):
        _dbg("bandit: not found on PATH; skipping.")
        return {}
    stdout = _run(["bandit", "-f", "json", "-q", "-r", str(tmpfile)], cwd, "bandit")
    hits: Dict[int, List[str]] = {}
    try:
        data = json.loads(stdout or "{}")
        for res in data.get("results", []):
            ln = int(res.get("line_number") or 0)
            text = res.get("issue_text") or ""
            tid = res.get("test_id") or ""
            if ln > 0 and text:
                hits.setdefault(ln, []).append(f"{tid}: {text}")
    except Exception as e:
        _dbg(f"bandit(json): parse error ({e})")
    for ln, msgs in sorted(hits.items()):
        for m in msgs:
            _dbg(f"bandit: hit line {ln}: {m}")
    return hits


# Annotation: SonarLint CLI (best-effort): try SARIF, then legacy HTML, then stdout shapes.
def _run_sonarlint(cwd: Path, source_filename: str = "snippet.py") -> Dict[int, List[str]]:
    exe = (
        os.environ.get("SONARLINT_EXE") or shutil.which("sonarlint") or shutil.which("sonarlint17")
    )
    if not exe:
        _dbg("SonarLint: not found on PATH; skipping.")
        return {}
    hits: Dict[int, List[str]] = {}
    sarif_path = cwd / "sonarlint.sarif"
    html_path = cwd / "sonarlint.html"
    stdout = _run([exe, "-f", "sarif", "-o", str(sarif_path), str(cwd)], cwd, "SonarLint(SARIF)")
    try:
        if sarif_path.exists():
            data = json.loads(sarif_path.read_text(encoding="utf-8", errors="ignore"))
            for run in data.get("runs", []):
                for res in run.get("results", []):
                    msg = (res.get("message", {}) or {}).get("text", "") or ""
                    for loc in res.get("locations", []):
                        phys = loc.get("physicalLocation", {}) or {}
                        art = (phys.get("artifactLocation", {}) or {}).get("uri", "") or ""
                        if (
                            art.endswith("/" + source_filename)
                            or art.endswith("\\" + source_filename)
                            or art == source_filename
                        ):
                            region = phys.get("region", {}) or {}
                            ln = int(region.get("startLine") or 0)
                            if ln > 0 and msg:
                                hits.setdefault(ln, []).append(msg.strip())
            if hits:
                for ln, msgs in sorted(hits.items()):
                    for m in msgs:
                        _dbg(f"SonarLint(SARIF): hit line {ln}: {m}")
                return hits
    except Exception as e:
        _dbg(f"SonarLint(SARIF): parse error ({e})")
    stdout2 = _run(
        [exe, "--src", source_filename, "--html-report", str(html_path)], cwd, "SonarLint(HTML)"
    )
    try:
        if html_path.exists():
            txt = html_path.read_text(encoding="utf-8", errors="ignore")
            pat_cells = re.compile(
                r"<td[^>]*>\s*(\d+)\s*</td>.*?<td[^>]*>\s*(?:major|minor|blocker|info|critical)\s*</td>.*?<td[^>]*>\s*(.*?)\s*</td>",
                re.IGNORECASE | re.DOTALL,
            )
            for m in pat_cells.finditer(txt):
                ln = int(m.group(1))
                msg = html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip()
                if ln > 0 and msg:
                    hits.setdefault(ln, []).append(msg)
            if not hits:
                pat_text = re.compile(
                    rf"{re.escape(source_filename)}:(?P<line>\d+):\s*(?P<msg>[^<\n\r]+)"
                )
                for m in pat_text.finditer(txt):
                    ln = int(m.group("line"))
                    msg = m.group("msg").strip()
                    hits.setdefault(ln, []).append(msg)
            if hits:
                for ln, msgs in sorted(hits.items()):
                    for m in msgs:
                        _dbg(f"SonarLint(HTML): hit line {ln}: {m}")
                return hits
    except Exception as e:
        _dbg(f"SonarLint(HTML): parse error ({e})")
    if stdout or stdout2:
        _dbg("SonarLint(stdout): attempting to parse simple 'file:line: message' lines.")
    pat_stdout = re.compile(rf"{re.escape(source_filename)}:(?P<line>\d+):\s*(?P<msg>.+)")
    for line in (stdout + "\n" + stdout2).splitlines():
        m = pat_stdout.search(line)
        if m:
            ln = int(m.group("line"))
            msg = m.group("msg").strip()
            hits.setdefault(ln, []).append(msg)
    if hits:
        for ln, msgs in sorted(hits.items()):
            for m in msgs:
                _dbg(f"SonarLint(stdout): hit line {ln}: {m}")
    else:
        _dbg("SonarLint: no issues found in any mode.")
    return hits


# Annotation: Aggregate reasons from AST, regexes, external tools and SonarLint; expose as a convenience function. Prints per-tool debug as it runs.
def collect_reasons(source_text: str) -> Dict[int, List[str]]:
    # Annotation: Split lines once for regex scanning and to compute 1-based line numbers.
    lines = source_text.splitlines()
    # Annotation: 1) AST heuristics.
    reasons: List[Tuple[int, str]] = []
    try:
        tree = ast.parse(source_text)
        _HeuristicVisitor(reasons).visit(tree)
        if _DEBUG and reasons:
            for ln, msg in reasons:
                _dbg(f"AST: line {ln}: {msg}")
    except Exception as e:
        _dbg(f"AST: parse failed ({e}); continuing with regex/tools.")
    # Annotation: 2) Line regex checks that also fire for fragments.
    for i, ln in enumerate(lines, start=1):
        for pat, msg in _REGEX_REASONS:
            if pat.search(ln):
                reasons.append((i, msg))
                _dbg(f"REGEX: line {i}: {msg}")
    # Annotation: 3) External tools (best effort). We write to a temp file and run tools that are installed.
    tmpdir = Path(tempfile.mkdtemp(prefix="ca_rules_"))
    tmpfile = tmpdir / "snippet.py"
    tmpfile.write_text(source_text, encoding="utf-8")
    _dbg(f"Temp file for tools: {tmpfile}")
    try:
        # Annotation: Run + parse each tool individually.
        fl_out = _run(["flake8", str(tmpfile)], tmpdir, "flake8")
        for ln, msgs in _from_flake8(fl_out).items():
            for msg in msgs:
                reasons.append((ln, msg))
        pf_out = _run(["pyflakes", str(tmpfile)], tmpdir, "pyflakes")
        for ln, msgs in _from_pyflakes(pf_out).items():
            for msg in msgs:
                reasons.append((ln, msg))
        pd_out = _run(["pydocstyle", str(tmpfile)], tmpdir, "pydocstyle")
        for ln, msgs in _from_pydocstyle(pd_out).items():
            for msg in msgs:
                reasons.append((ln, msg))
        pl_out = _run(["pylint", "--output-format=parseable", str(tmpfile)], tmpdir, "pylint")
        for ln, msgs in _from_pylint(pl_out).items():
            for msg in msgs:
                reasons.append((ln, msg))
        for ln, msgs in _run_bandit_json(tmpdir, tmpfile).items():
            for msg in msgs:
                reasons.append((ln, msg))
        for ln, msgs in _run_sonarlint(tmpdir, source_filename="snippet.py").items():
            for msg in msgs:
                reasons.append((ln, msg))
    finally:
        # Annotation: Ensure we remove the temp directory.
        shutil.rmtree(tmpdir, ignore_errors=True)
        _dbg(f"Temp folder removed: {tmpdir}")
    # Annotation: Collate into a mapping line->list[str], preserving insertion order and de-duplicating messages.
    out: Dict[int, List[str]] = {}
    for ln, msg in reasons:
        arr = out.setdefault(ln, [])
        if msg not in arr:
            arr.append(msg)
    # Annotation: Print a concise summary of all reasons per line when debugging.
    if _DEBUG:
        if not out:
            _dbg("No reasons collected from any source.")
        else:
            for ln in sorted(out):
                for msg in out[ln]:
                    _dbg(f"REASONS: line {ln}: {msg}")
    # Annotation: Return the aggregated reasons per line.
    return out


# Annotation: Keyword buckets used to suggest a label directly from tool/AST reasons.
_SAST_HINTS = (
    "unsafe",
    "injection",
    "verify=false",
    "exec",
    "eval",
    "unpickl",
    "yaml.load",
    "shell=true",
    "credential",
    "bandit",
    "deserial",
    "sql",
    "pickle",
    "os.system",
    "subprocess.run",
    "requests.get",
    "disabled",
    "ssl",
    "xss",
)
# Annotation: Best-practice hints focus on style, readability and maintainability.
_BEST_PRACTICE_HINTS = (
    "pep8",
    "docstring",
    "unused",
    "import",
    "naming",
    "format",
    "style",
    "convention",
    "complexity",
    "cyclomatic",
    "shadow",
    "mutable default",
    "reassign",
    "vulture",
    "newline",
)


# Annotation: Choose the best reason for a given label by preferring security-oriented messages for 'sast_risk' and style for 'best_practice'.
def best_reason_for_line(
    line_no: int, label_key: str, reasons_by_line: Dict[int, List[str]], fallback: str
) -> str:
    # Annotation: If no reasons recorded for this line, return the provided fallback.
    if line_no not in reasons_by_line or not reasons_by_line[line_no]:
        return fallback
    # Annotation: For SAST, prioritise messages that look security-related.
    if label_key == "sast_risk":
        for m in reasons_by_line[line_no]:
            if any(x in m.lower() for x in _SAST_HINTS):
                return m
    # Annotation: For best practice, prefer style/readability/docstring nudges.
    if label_key == "best_practice":
        for m in reasons_by_line[line_no]:
            if any(x in m.lower() for x in _BEST_PRACTICE_HINTS):
                return m
    # Annotation: For ml_signal or general case, return the first available reason.
    return reasons_by_line[line_no][0]


# Annotation: Suggest a label purely from the tool/AST reasons on a specific line. Returns 'sast_risk'/'best_practice'/None.
def suggest_label_for_line(line_no: int, reasons_by_line: Dict[int, List[str]]) -> str | None:
    # Annotation: No reasons means no override suggestion.
    if line_no not in reasons_by_line:
        return None
    # Annotation: Security hints trump everything else when present.
    joined = " ".join(reasons_by_line[line_no]).lower()
    if any(k in joined for k in _SAST_HINTS):
        return "sast_risk"
    if any(k in joined for k in _BEST_PRACTICE_HINTS):
        return "best_practice"
    return None


# Annotation: Merge model scores with rule-based suggestions, prioritising tools first when they have any signal; returns (label, reason, score_for_label, used_rules).
def decide_label_and_reason(
    line_no: int,
    model_scores: Dict[str, float],
    reasons_by_line: Dict[int, List[str]],
    model_top: str,
    default_fallback: str,
) -> Tuple[str, str, float, bool]:
    # Annotation: Ask the rules to suggest a label; if present, we override the model and choose the best matching reason.
    rules_label = suggest_label_for_line(line_no, reasons_by_line)
    if rules_label:
        reason = best_reason_for_line(line_no, rules_label, reasons_by_line, default_fallback)
        conf = float(model_scores.get(rules_label, 0.0))
        _dbg(
            f"DECIDE: line {line_no} rules-first -> {rules_label} | reason='{reason}' | model_score={conf:.3f}"
        )
        return rules_label, reason, conf, True
    # Annotation: Otherwise, fall back to the model’s choice but still try to pick a relevant reason for that label.
    reason = best_reason_for_line(line_no, model_top, reasons_by_line, default_fallback)
    conf = float(model_scores.get(model_top, 0.0))
    _dbg(
        f"DECIDE: line {line_no} model-fallback -> {model_top} | reason='{reason}' | model_score={conf:.3f}"
    )
    return model_top, reason, conf, False
