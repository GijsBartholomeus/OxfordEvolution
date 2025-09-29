import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import Counter

from SloppyCell.ReactionNetworks import Network, Dynamics

# Load SBML model using SloppyCell
net = Network()
net.loadSBMLFile('chen2004_biomd56.xml')

multipliers = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]

def sample_parameters(net, wildtype=False):
    """Multiply only free global parameters by a random factor, or set all to 1 for wildtype."""
    sampled = {}
    for pid in net.parameters.keys():
        current = net.parameters[pid]
        factor = 1.0 if wildtype else random.choice(multipliers)
        net.parameters[pid] = current * factor
        sampled[pid] = factor
    return sampled

def simulate_and_extract(net, tmax=200, npoints=2001):
    times = np.linspace(0, tmax, npoints)
    try:
        result = Dynamics.integrateNetwork(net, times)
    except Exception as e:
        return None, None
    # Extract CLB2 trajectory
    if 'CLB2' not in result:
        return None, None
    clb2 = result['CLB2']
    return times, clb2

def up_down_encoding(time, signal, nbins=40):
    """Binary string of slope signs at evenly spaced time bins."""
    bits = []
    sample_times = np.linspace(time[0], time[-1], nbins + 1)[:-1]
    dt = sample_times[1] - sample_times[0]
    for t in sample_times:
        val_now = np.interp(t, time, signal)
        val_next = np.interp(t + dt, time, signal)
        slope = val_next - val_now
        bits.append("1" if slope >= 0 else "0")
    return "".join(bits)

def Nw(s):
    """Counts the number of distinct substrings in s using Lempel-Ziv parsing."""
    n = len(s)
    i = 0
    count = 1
    while i < n - 1:
        l = 1
        while i + l <= n and s[i:i+l] in s[:i]:
            l += 1
        i += l
        count += 1
    return count

def CLZ(x):
    """Lempel-Ziv complexity as described in the prompt."""
    n = len(x)
    if x.count('0') == n or x.count('1') == n:
        return math.log2(n)
    else:
        return math.log2(n) / 2 * (Nw(x) + Nw(x[::-1]))

# Main workflow
wildtype = False  # Set to True for wildtype, False for randomized

net_copy = net.copy()
sampled_params = sample_parameters(net_copy, wildtype=wildtype)
time, clb2 = simulate_and_extract(net_copy)

if time is None or clb2 is None:
    print("WARNING: Integration failed for this parameter set!")
else:
    encoding = up_down_encoding(time, clb2, nbins=40)
    plt.plot(time, clb2, label="CLB2")
    plt.xlabel("Time (min)")
    plt.ylabel("Concentration")
    plt.title(f"Chen model (CLB2 trajectory) {'wildtype' if wildtype else 'randomized'}")
    plt.legend()
    plt.show()

    print("Sampled parameters (multipliers):")
    for k, v in sampled_params.items():
        print(f"{k}: {v}")

    print("Up-Down encoding:", encoding[:120], "...")
    print("Lempel-Ziv complexity (CLZ):", CLZ(encoding))

# Sampling loop
N = 5000
encodings = []
complexities = []
skipped_count = 0

for i in range(N):
    net_sample = net.copy()
    sample_parameters(net_sample)
    time, clb2 = simulate_and_extract(net_sample)
    if time is None or clb2 is None:
        skipped_count += 1
        continue
    encoding = up_down_encoding(time, clb2, nbins=40)
    encodings.append(encoding)
    complexities.append(CLZ(encoding))
    if (i+1) % 100 == 0:
        print(f"Completed {i+1} samples | successful: {len(encodings)} | skipped: {skipped_count}")

print(f"\nFinal results: {len(encodings)} successful, {skipped_count} skipped")
print("Example encoding:", encodings[0][:120], "...")
print("Example complexity:", complexities[0])
print("Mean complexity:", np.mean(complexities))
print("Std complexity:", np.std(complexities))

plt.figure()
plt.hist(complexities, bins=30, color='skyblue', edgecolor='black')
plt.xlabel("Lempel-Ziv Complexity (CLZ)")
plt.ylabel("Count")
plt.title("Distribution of CLZ Complexity (CLB2 up-down encoding)")
plt.show()

# Cluster phenotypes and log-log plot
phenotype_counts = Counter(encodings)
frequencies = sorted(phenotype_counts.values(), reverse=True)
ranks = np.arange(1, len(frequencies) + 1)

plt.figure()
plt.loglog(ranks, frequencies, marker='o', linestyle='none')
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.title("Log-Log Plot of Phenotype String Frequency by Rank")
plt.show()

# Wildtype rank
net_wt = net.copy()
sample_parameters(net_wt, wildtype=True)
time_wt, clb2_wt = simulate_and_extract(net_wt)
encoding_wt = up_down_encoding(time_wt, clb2_wt, nbins=40)

sorted_phenotypes = [k for k, v in sorted(phenotype_counts.items(), key=lambda item: item[1], reverse=True)]
if encoding_wt in sorted_phenotypes:
    wt_rank = sorted_phenotypes.index(encoding_wt) + 1
    wt_freq = phenotype_counts[encoding_wt]
    print(f"Wildtype phenotype rank: {wt_rank} (frequency: {wt_freq})")
else:
    print("Wildtype phenotype not found in sampled set.")