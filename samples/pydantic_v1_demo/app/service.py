from .models import Order, User


def load_user(data: dict) -> User:
    return User.parse_obj(data)


def user_to_dict(user: User) -> dict:
    return user.dict()


def user_to_json(user: User) -> str:
    return user.json()


def order_total(order: Order, unit_price: float) -> float:
    total = order.quantity * unit_price
    if order.discount:
        total *= 1 - order.discount
    return round(total, 2)