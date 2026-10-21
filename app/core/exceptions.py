class ConfigurationError(RuntimeError):
    pass


class ParsingError(RuntimeError):
    pass


class ModelGatewayError(RuntimeError):
    pass


class ModelOutputError(RuntimeError):
    """The model gateway responded, but the structured output is unusable.

    Raised instead of silently degrading: a model that returns non-JSON or a
    contract-violating payload must fail the request visibly.
    """


class RetrievalError(RuntimeError):
    """Retrieval-layer failure (missing/corrupt parsed artifacts, index
    capacity, discovery failures)."""


class StoreError(RuntimeError):
    """Persistent-store integrity failure (corrupt job records, unwritable
    state files). Never swallowed: a corrupt record must surface, not
    masquerade as "not found"."""


class DiagnosisError(RuntimeError):
    """A diagnosis stage failed (for example the optional AI suggestion call).

    Raised instead of returning a partial diagnosis with the upstream error
    text embedded in a 200 response.
    """
