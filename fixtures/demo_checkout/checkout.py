"""Small two-file checkout fixture used by the presentation demo."""

from payment import PaymentGateway


class Checkout:
    def __init__(self, payment: PaymentGateway, log) -> None:
        self.payment = payment
        self.log = log

    def checkout(self, cart: list[int]) -> int:
        self.log.start()
        self._validate(cart)
        total = self._total(cart)
        self.payment.charge(total)
        self.log.end()
        return total

    def _validate(self, cart: list[int]) -> None:
        if not cart:
            raise ValueError("cart must not be empty")

    def _total(self, cart: list[int]) -> int:
        return sum(cart)
