import hashlib

def hash_password(password: str) -> str:
    """Hashes the provided password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()