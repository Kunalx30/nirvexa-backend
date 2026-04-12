import re


def validate_email(email: str) -> bool:
    """Check if email format is valid."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def validate_password(password: str) -> dict:
    """
    Password must be:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    """
    errors = []

    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    if not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter")
    if not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter")
    if not re.search(r'\d', password):
        errors.append("Password must contain at least one number")
    if not re.search(r'[!@#$%^&*(),.?\":{}|<>]', password):
        errors.append("Password must contain at least one special character")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }


def validate_name(name: str) -> bool:
    """Name must be 2-100 characters, letters and spaces only."""
    name = name.strip()
    return 2 <= len(name) <= 100 and bool(re.match(r'^[a-zA-Z\s]+$', name))