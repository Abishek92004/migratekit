import pytest
from pydantic import ValidationError

from app.models import Order, User


def test_user_valid():
    u = User(name="Ravi", age=20, email="r@example.com")
    assert u.age == 20


def test_user_email_is_optional():
    u = User(name="Asha", age=30)
    assert u.email is None


def test_user_negative_age_rejected():
    with pytest.raises(ValidationError):
        User(name="Ravi", age=-1)


def test_user_name_is_stripped():
    assert User(name="  Ravi  ", age=20).name == "Ravi"


def test_order_discount_is_optional():
    assert Order(quantity=2).discount is None


def test_order_quantity_must_be_positive():
    with pytest.raises(ValidationError):
        Order(quantity=0)