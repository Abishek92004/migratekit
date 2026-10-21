import json

from app.models import Order, User
from app.service import load_user, order_total, user_to_dict, user_to_json


def test_load_user():
    u = load_user({"name": "Asha", "age": 30})
    assert u.name == "Asha"


def test_user_to_dict():
    d = user_to_dict(User(name="Ravi", age=20))
    assert d == {"name": "Ravi", "age": 20, "email": None}


def test_user_to_json():
    data = json.loads(user_to_json(User(name="Ravi", age=20)))
    assert data["age"] == 20


def test_order_total_with_discount():
    assert order_total(Order(quantity=2, discount=0.5), 10.0) == 10.0


def test_order_total_without_discount():
    assert order_total(Order(quantity=3), 10.0) == 30.0