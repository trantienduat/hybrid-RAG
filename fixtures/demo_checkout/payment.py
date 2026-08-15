"""Payment implementation for the two-file checkout demo fixture."""


class PaymentGateway:
    def charge(self, amount: int) -> str:
        self._authorize(amount)
        self._capture(amount)
        return "ok"

    def _authorize(self, amount: int) -> None:
        if amount < 0:
            raise ValueError("amount must not be negative")

    def _capture(self, amount: int) -> None:
        _ = amount
