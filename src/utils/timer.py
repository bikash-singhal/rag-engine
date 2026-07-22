from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timer(latency, field: str):
    """
    Measures execution time (milliseconds) and stores it
    inside the given LatencyReport field.
    """

    start = perf_counter()

    yield

    elapsed = (perf_counter() - start) * 1000

    setattr(
        latency,
        field,
        elapsed,
    )
