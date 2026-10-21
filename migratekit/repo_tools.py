"""Guarded tools for working on a target repository.

Every path is relative to the repository root. This file is the one place where
the safety rules live; Step 3b exposes these same functions through MCP servers.
"""
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from typing import Protocol

from migratekit.harness.tools import Tool

MAX_READ_CHARS = 6000
MAX_WRITE_BYTES = 200_000
MAX_LIST_ENTRIES = 200
MAX_SEARCH_RESULTS = 40
MAX_TEST_OUTPUT_CHARS = 4000

HIDDEN_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules",
}
SEARCHABLE_SUFFIXES = {".py", ".txt", ".md", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".json"}
PROTECTED_FILENAMES = {"conftest.py", "pytest.ini"}


class GuardrailError(Exception):
    """An action broke a safety rule. The message goes back to the model."""


class SandboxRunner(Protocol):
    def run(self, repo_root: Path, target: str | None) -> tuple[int, str]: ...


class DockerRunner:
    """Runs the repo's tests inside a locked-down container.

    No network, read-only copy of the repo, memory/CPU/process limits,
    all Linux capabilities dropped, and a hard timeout.
    """

    def __init__(
        self,
        image: str = "migratekit-sandbox",
        timeout_s: int = 180,
        memory: str = "1g",
        cpus: str = "1",
    ):
        self.image = image
        self.timeout_s = timeout_s
        self.memory = memory
        self.cpus = cpus

    def run(self, repo_root: Path, target: str | None) -> tuple[int, str]:
        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", self.memory,
            "--cpus", self.cpus,
            "--pids-limit", "256",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--tmpfs", "/tmp",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "--mount", f"type=bind,source={repo_root},target=/work,readonly",
            "-w", "/work",
            self.image,
            "python", "-m", "pytest", "-q", "--maxfail=5", "--tb=short",
            "-p", "no:cacheprovider", "-W", "error::DeprecationWarning",
        ]
        if target:
            cmd.append(target)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_s,
                encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            return 124, f"Tests timed out after {self.timeout_s} seconds."
        except FileNotFoundError:
            return 127, "Docker is not available. Is Docker Desktop running?"
        out = proc.stdout + proc.stderr
        if proc.returncode == 125:
            out += (
                "\n(Docker could not start the container. Check that Docker Desktop is "
                "running and that the image exists: docker build -t migratekit-sandbox sandbox/)"
            )
        return proc.returncode, out


class RepoTools:
    def __init__(self, root: str | Path, runner: SandboxRunner | None = None):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Not a directory: {root}")
        self._runner = runner or DockerRunner()

    # ---- guardrails --------------------------------------------------
    @staticmethod
    def _is_secret(p: Path) -> bool:
        return p.name == ".env" or p.name.startswith(".env.") or p.suffix in {".pem", ".key"}

    def _resolve(self, rel: str) -> Path:
        p = (self.root / (rel or ".")).resolve()
        if not p.is_relative_to(self.root):
            raise GuardrailError("Path is outside the repository.")
        parts = p.relative_to(self.root).parts
        if any(part in HIDDEN_PARTS for part in parts) or self._is_secret(p):
            raise GuardrailError("Access to this path is not allowed.")
        return p

    def _check_writable(self, p: Path) -> None:
        parts = p.relative_to(self.root).parts
        name = p.name
        if (
            "tests" in parts or "test" in parts
            or name in PROTECTED_FILENAMES
            or name.startswith("test_") or name.endswith("_test.py")
        ):
            raise GuardrailError(
                "Test files are read-only. Fix the application code, not the tests."
            )

    @staticmethod
    def _check_content(p: Path, content: str) -> None:
        if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
            raise GuardrailError(f"File too large (limit {MAX_WRITE_BYTES} bytes).")
        if p.suffix == ".py":
            try:
                ast.parse(content)
            except SyntaxError as e:
                raise GuardrailError(f"Refusing to write invalid Python: {e}") from None

    # ---- tools -------------------------------------------------------
    def list_dir(self, path: str = ".") -> str:
        p = self._resolve(path)
        if not p.is_dir():
            raise ValueError(f"Not a directory: {path}")
        names = []
        for child in sorted(p.iterdir()):
            if child.name in HIDDEN_PARTS or self._is_secret(child):
                continue
            names.append(child.name + ("/" if child.is_dir() else ""))
        return "\n".join(names[:MAX_LIST_ENTRIES]) or "(empty)"

    def read_file(self, path: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ValueError(f"Not a file: {path}")
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_READ_CHARS:
            return text[:MAX_READ_CHARS] + f"\n...[truncated, file has {len(text)} characters]"
        return text

    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        self._check_writable(p)
        self._check_content(p, content)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content.encode("utf-8"))
        return f"Wrote {path} ({len(content)} characters)."

    def replace_in_file(self, path: str, old: str, new: str) -> str:
        p = self._resolve(path)
        self._check_writable(p)
        if not p.is_file():
            raise ValueError(f"Not a file: {path}")
        text = p.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            raise ValueError(
                f"'old' must match exactly once, but it matched {count} times. "
                "Include more surrounding text to make it unique."
            )
        updated = text.replace(old, new)
        self._check_content(p, updated)
        p.write_bytes(updated.encode("utf-8"))
        return f"Edited {path}."

    def search(self, pattern: str, path: str = ".") -> str:
        base = self._resolve(path)
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regular expression: {e}") from None
        files = [base] if base.is_file() else sorted(base.rglob("*"))
        hits: list[str] = []
        for f in files:
            rel = f.relative_to(self.root)
            if (
                not f.is_file()
                or f.suffix not in SEARCHABLE_SUFFIXES
                or any(part in HIDDEN_PARTS for part in rel.parts)
                or self._is_secret(f)
            ):
                continue
            for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{rel.as_posix()}:{n}: {line.strip()[:200]}")
                    if len(hits) >= MAX_SEARCH_RESULTS:
                        return "\n".join(hits) + "\n...[more matches not shown]"
        return "\n".join(hits) or "No matches."

    def run_tests(self, target: str = "") -> str:
        rel = None
        if target:
            p = self._resolve(target)
            if not p.exists():
                raise ValueError(f"No such path: {target}")
            rel = p.relative_to(self.root).as_posix()
        code, out = self._runner.run(self.root, rel)
        status = "PASSED" if code == 0 else f"FAILED (exit code {code})"
        return f"{status}\n{out[-MAX_TEST_OUTPUT_CHARS:]}"


def build_tools(rt: RepoTools) -> list[Tool]:
    def obj(props: dict, required: list[str] | None = None) -> dict:
        return {"type": "object", "properties": props, "required": required or []}

    s = {"type": "string"}
    return [
        Tool("list_dir", "List files and folders in a directory of the repository. "
             "Use '.' for the repository root.",
             obj({"path": s}), rt.list_dir),
        Tool("read_file", "Read a text file from the repository.",
             obj({"path": s}, ["path"]), rt.read_file),
        Tool("search", "Search the repository with a regular expression. "
             "Returns matching lines as path:line: text.",
             obj({"pattern": s, "path": s}, ["pattern"]), rt.search),
        Tool("write_file", "Create or overwrite a source file with the full new content. "
             "Test files cannot be changed.",
             obj({"path": s, "content": s}, ["path", "content"]), rt.write_file),
        Tool("replace_in_file", "Replace one exact piece of text in a file. 'old' must "
             "appear exactly once. Test files cannot be changed.",
             obj({"path": s, "old": s, "new": s}, ["path", "old", "new"]), rt.replace_in_file),
        Tool("run_tests", "Run the repository's tests in a sandbox. Optionally pass a test "
             "file or folder as 'target'. Returns PASSED or FAILED plus the output.",
             obj({"target": s}), rt.run_tests),
    ]