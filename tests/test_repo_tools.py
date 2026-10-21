import pytest

from migratekit.repo_tools import GuardrailError, RepoTools, build_tools


class FakeRunner:
    def __init__(self, code=0, out="ok"):
        self.code, self.out, self.calls = code, out, []

    def run(self, repo_root, target):
        self.calls.append(target)
        return self.code, self.out


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("x = 1\ny = 2\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("def test_x(): pass\n")
    (tmp_path / ".env").write_text("SECRET=1\n")
    (tmp_path / ".git").mkdir()
    (tmp_path.parent / "outside.txt").write_text("nope")
    return tmp_path


def make(repo, runner=None):
    return RepoTools(repo, runner=runner or FakeRunner())


def test_cannot_escape_repo(repo):
    with pytest.raises(GuardrailError):
        make(repo).read_file("../outside.txt")


def test_cannot_read_secrets_or_git(repo):
    rt = make(repo)
    with pytest.raises(GuardrailError):
        rt.read_file(".env")
    with pytest.raises(GuardrailError):
        rt.list_dir(".git")


def test_list_dir_hides_secrets(repo):
    out = make(repo).list_dir(".")
    assert "app/" in out and "tests/" in out
    assert ".env" not in out and ".git" not in out


def test_cannot_edit_tests(repo):
    rt = make(repo)
    with pytest.raises(GuardrailError):
        rt.write_file("tests/test_main.py", "def test_x(): assert False\n")
    with pytest.raises(GuardrailError):
        rt.replace_in_file("tests/test_main.py", "pass", "assert True")
    with pytest.raises(GuardrailError):
        rt.write_file("app/test_helpers.py", "x = 1\n")


def test_write_and_read_source(repo):
    rt = make(repo)
    rt.write_file("app/new.py", "z = 3\n")
    assert rt.read_file("app/new.py") == "z = 3\n"


def test_invalid_python_is_refused(repo):
    with pytest.raises(GuardrailError):
        make(repo).write_file("app/bad.py", "def (:\n")


def test_replace_requires_unique_match(repo):
    rt = make(repo)
    with pytest.raises(ValueError):
        rt.replace_in_file("app/main.py", "zzz", "q")
    (repo / "app" / "dup.py").write_text("a = 1\na = 1\n")
    with pytest.raises(ValueError):
        rt.replace_in_file("app/dup.py", "a = 1", "a = 2")
    rt.replace_in_file("app/main.py", "x = 1", "x = 10")
    assert "x = 10" in rt.read_file("app/main.py")


def test_search(repo):
    rt = make(repo)
    assert "app/main.py:2: y = 2" in rt.search(r"y = \d")
    assert rt.search("SECRET") == "No matches."


def test_run_tests_uses_runner_and_formats(repo):
    runner = FakeRunner(code=1, out="x" * 9000)
    out = make(repo, runner).run_tests("tests/test_main.py")
    assert out.startswith("FAILED (exit code 1)")
    assert len(out) < 4200
    assert runner.calls == ["tests/test_main.py"]


def test_run_tests_rejects_bad_target(repo):
    with pytest.raises(GuardrailError):
        make(repo).run_tests("../outside.txt")


def test_build_tools_wires_every_tool(repo):
    names = [t.name for t in build_tools(make(repo))]
    assert names == ["list_dir", "read_file", "search", "write_file", "replace_in_file", "run_tests"]