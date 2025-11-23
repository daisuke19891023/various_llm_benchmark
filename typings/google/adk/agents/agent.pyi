
class Agent:
    name: str
    description: str | None
    model: str | None
    instruction: str | None

    def __init__(
        self,
        *,
        name: str,
        description: str | None = None,
        model: str | None = None,
        instruction: str | None = None,
    ) -> None: ...
