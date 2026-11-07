"""Performs a Luhn check that tells if a number is a real card or random digits"""

def passes_luhn(text:str)->bool:
    if any(not (char.isdigit() or char in " -") for char in text):
        return False
    digits = [int(char) for char in text if char.isdigit()]
    if not 13<= len(digits) <= 19:
        return False

    total = 0
    for position, digit in enumerate(reversed(digits)):
        if position % 2 == 1:  # every second digit from the right is doubled
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0