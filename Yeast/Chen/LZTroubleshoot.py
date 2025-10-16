import math

# -------------------------------
# 1) Your original version
# -------------------------------
def lz76_phrase_count(s: str) -> int:
    n = len(s)
    if n == 0:
        return 0
    i = 0
    c = 1
    k = 1
    while i + k <= n:
        if s[i:i+k] in s[:i]:
            k += 1
            if i + k - 1 > n:
                c += 1
                break
        else:
            c += 1
            i += k
            k = 1
    return c

def CLZ(x: str) -> float:
    n = len(x)
    if x.count('0') == n or x.count('1') == n:
        return math.log2(n)
    else:
        return math.log2(n) / 2 * (lz76_phrase_count(x) + lz76_phrase_count(x[::-1]))


# -------------------------------
# 2) Correct LZ76-based implementation
# -------------------------------
def lz76_complexity(s: str) -> int:
    i, c, k, n = 0, 1, 1, len(s)
    while True:
        if i + k <= n and s[i:i+k] in s[:i]:
            k += 1
            if i + k - 1 > n:
                c += 1
                break
        else:
            c += 1
            i += k
            if i >= n:
                break
            k = 1
    return c

def normalized_lz_complexity(s: str) -> float:
    n = len(s)
    if n <= 1:
        return 0.0
    c = lz76_complexity(s)
    return c / (n / math.log2(n))


# -------------------------------
# 3) Reference: Antropy library
# -------------------------------
try:
    from antropy import lziv_complexity
    antropy_available = True
except ImportError:
    antropy_available = False


# -------------------------------
# Run comparison
# -------------------------------
seqs = [
    "0000000000000000",   # constant
    "0101010101010101",   # periodic
    "0110101101011010",   # pseudo-random
]

print(f"{'sequence':<20} {'CLZ':>10} {'CLZnew':>12} {'antropy':>12}")
print("-" * 58)

for s in seqs:
    c1 = CLZ(s)
    c2 = normalized_lz_complexity(s)
    if antropy_available:
        c3 = lziv_complexity(list(map(int, s)), normalize=True)
    else:
        c3 = None
    print(f"{s:<20} {c1:10.3f} {c2:12.3f} {c3 if c3 is not None else 'N/A':>12}")

if not antropy_available:
    print("\nNote: antropy not installed. To install, run:\n    pip install antropy")
