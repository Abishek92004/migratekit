"""Step 3a check: sandboxed tests plus guardrails. No LLM involved.

Run from the project root:
    python -m scripts.check_step3a
"""
import shutil
import tempfile
from pathlib import Path

from migratekit.repo_tools import GuardrailError, RepoTools

SAMPLE = Path("samples/pydantic_v1_demo")

MODELS_V2 = '''from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


class User(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str
    age: int
    email: Optional[str] = None

    @field_validator("age")
    @classmethod
    def check_age(cls, v):
        if v < 0:
            raise ValueError("age must be non-negative")
        return v


class Order(BaseModel):
    quantity: int
    discount: Optional[float] = None

    @field_validator("quantity")
    @classmethod
    def check_quantity(cls, v):
        if v <= 0:
            raise ValueError("quantity must be positive")
        return v
'''

SERVICE_V2 = '''from .models import Order, User


def load_user(data: dict) -> User:
    return User.model_validate(data)


def user_to_dict(user: User) -> dict:
    return user.model_dump()


def user_to_json(user: User) -> str:
    return user.model_dump_json()


def order_total(order: Order, unit_price: float) -> float:
    total = order.quantity * unit_price
    if order.discount:
        total *= 1 - order.discount
    return round(total, 2)
'''


def head(text: str, n: int = 6) -> str:
    return "\n".join(text.splitlines()[:n])


def main(runner=None) -> None:
    if not SAMPLE.is_dir():
        raise SystemExit(f"Sample repo not found at {SAMPLE}. Run from the project root.")

    work = Path(tempfile.mkdtemp(prefix="migratekit_")) / "repo"
    shutil.copytree(SAMPLE, work)
    rt = RepoTools(work, runner=runner)
    print(f"Working copy: {work}\n")

    print("1) Files in the repo:")
    print(rt.list_dir("."))

    print("\n2) Tests BEFORE migration (should FAIL):")
    before = rt.run_tests()
    print(head(before))
    assert before.startswith("FAILED"), "baseline should fail on Pydantic v2"

    print("\n3) Guardrails (each of these must be blocked):")
    (work / ".env").write_text("SECRET=1\n")
    attempts = {
        "read a file outside the repo": lambda: rt.read_file("../../../etc/passwd"),
        "read .env": lambda: rt.read_file(".env"),
        "edit a test file": lambda: rt.write_file("tests/test_models.py", "x = 1\n"),
        "write invalid Python": lambda: rt.write_file("app/bad.py", "def (:\n"),
    }
    for label, attempt in attempts.items():
        try:
            attempt()
        except GuardrailError as e:
            print(f"   blocked: {label} -> {e}")
        else:
            raise SystemExit(f"GUARDRAIL FAILURE: allowed to {label}")

    print("\n4) Applying a hand-written Pydantic v2 migration...")
    print(rt.write_file("app/models.py", MODELS_V2))
    print(rt.write_file("app/service.py", SERVICE_V2))

    print("\n5) Tests AFTER migration (should PASS):")
    after = rt.run_tests()
    print(head(after))
    assert after.startswith("PASSED"), "migrated code should pass"

    shutil.rmtree(work.parent, ignore_errors=True)
    print("\nStep 3a check complete: sandbox and guardrails work.")


if __name__ == "__main__":
    main()