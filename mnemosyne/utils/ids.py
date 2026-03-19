from ulid import ULID


def generate_id() -> str:
    """Generate a new ULID string."""
    return str(ULID())
