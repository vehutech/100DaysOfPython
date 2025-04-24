# timeUp.py

import time

def count_timer(end, start=0, *, label="Timer", interval=1, **metadata):
    """
    Counts from start to end (inclusive), pausing 'interval' seconds between numbers.

    Parameters:
    - end (int): The ending count (required, positional)
    - start (int): The starting count (default=0)
    - label (str): A label for the timer (keyword-only)
    - interval (float): Time in seconds between each count (keyword-only)
    - metadata (dict): Arbitrary keyword arguments, for additional context (not used in counting logic)
    """

    print(f"\nStarting '{label}' from {start} to {end}...")
    for i in range(start, end + 1):
        print(f"{label}: {i}")
        time.sleep(interval)
    print(f"'{label}' complete!\n")

    if metadata:
        print("Additional Info:")
        for key, value in metadata.items():
            print(f"  - {key}: {value}")
    print("-" * 30)

# === POSITIONAL ARGUMENTS ===
# These are passed based on order: first is 'end', second is 'start'
count_timer(5, 2)
# Starts from 2, ends at 5 — order matters.

# === DEFAULT ARGUMENT ===
# Only 'end' is provided; 'start' uses its default of 0
count_timer(3)
# Starts from 0, ends at 3 — uses default start.

# === KEYWORD ARGUMENTS ===
# 'label' and 'interval' are specified by name (keyword-only arguments)
count_timer(4, 1, label="Soft-Boiled Egg", interval=2)
# A timer that starts at 1 and counts to 4, waiting 2 seconds between counts

# === MIXED USE ===
# Positional for end and start, keyword for label
count_timer(6, 3, label="Bread Proofing")
# Readable, and parameters are clear.

# === ARBITRARY KEYWORD ARGUMENTS ===
# Additional info passed using **metadata (these are not used in logic)
count_timer(4, 2, label="Coffee Brew", interval=1.5, method="Pour Over", grinder="Medium")
# Useful for logs or future extensions

