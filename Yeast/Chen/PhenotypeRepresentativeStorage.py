"""
Enhanced Phenotype Tracker with Representative Storage
Stores representatives for troubleshooting frequency-complexity diagrams
"""

import numpy as np
import pickle
import os
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Union
import math

class CompactRepresentative:
    """Compact storage for a single phenotype representative"""
    
    def __init__(self, encoding: str, complexity: float, period: float, 
                 genotype: List[float], coarse_data: Tuple[np.ndarray, np.ndarray],
                 frequency: int = 1):
        self.encoding = encoding
        self.complexity = complexity
        self.period = period
        self.genotype = np.array(genotype, dtype=np.float32)  # Use float32 to save memory
        self.coarse_time = np.array(coarse_data[0], dtype=np.float32)
        self.coarse_signal = np.array(coarse_data[1], dtype=np.float32)
        self.frequency = frequency
    
    def get_memory_size(self):
        """Estimate memory usage in bytes"""
        return (
            len(self.encoding) + 
            8 + 8 + 4 +  # complexity, period, frequency
            self.genotype.nbytes + 
            self.coarse_time.nbytes + 
            self.coarse_signal.nbytes
        )

class PhenotypeTrackerWithRepresentatives:
    """Enhanced phenotype tracker that stores representatives for analysis"""
    
    def __init__(self, n_track=5, max_representatives_per_phenotype=9, 
                 min_frequency_for_storage=2, memory_limit_gb=10):
        self.n_track = n_track
        self.max_reps_per_phenotype = max_representatives_per_phenotype
        self.min_freq_for_storage = min_frequency_for_storage
        self.memory_limit_bytes = memory_limit_gb * 1024**3
        
        # Core tracking (same as original)
        self.frequencies = Counter()
        self.complexities = {}
        
        # Representative storage
        self.representatives = defaultdict(list)  # phenotype -> list of CompactRepresentative
        self.complexity_index = defaultdict(set)  # complexity_bin -> set of phenotypes
        self.current_memory_usage = 0
        
        # Statistics
        self.total_samples_seen = 0
        self.stored_representatives_count = 0
        self.memory_overflow_count = 0
        
    def _get_complexity_bin(self, complexity: float, bin_size: float = 0.5) -> float:
        """Bin complexities for efficient lookup"""
        return round(complexity / bin_size) * bin_size
    
    def _should_store_representative(self, phenotype: str, frequency: int) -> bool:
        """Decide whether to store this representative"""
        if frequency < self.min_freq_for_storage:
            return False
        
        current_reps = len(self.representatives[phenotype])
        if current_reps >= self.max_reps_per_phenotype:
            return False
        
        # Check memory limit
        if self.current_memory_usage > self.memory_limit_bytes:
            self.memory_overflow_count += 1
            return False
        
        return True
    
    def _add_representative(self, phenotype: str, complexity: float, period: float,
                          genotype: List[float], coarse_data: Tuple[np.ndarray, np.ndarray],
                          frequency: int):
        """Add a representative for this phenotype"""
        rep = CompactRepresentative(phenotype, complexity, period, genotype, coarse_data, frequency)
        
        self.representatives[phenotype].append(rep)
        self.complexity_index[self._get_complexity_bin(complexity)].add(phenotype)
        self.current_memory_usage += rep.get_memory_size()
        self.stored_representatives_count += 1
    
    def update(self, encoding: str, complexity: float, period: float = None,
               genotype: List[float] = None, coarse_data: Tuple[np.ndarray, np.ndarray] = None):
        """Update tracker with new phenotype (enhanced version)"""
        self.total_samples_seen += 1
        
        # Update core tracking
        self.frequencies[encoding] += 1
        freq = self.frequencies[encoding]
        
        if encoding not in self.complexities:
            self.complexities[encoding] = complexity
        
        # Store representative if conditions are met
        if (self._should_store_representative(encoding, freq) and 
            period is not None and genotype is not None and coarse_data is not None):
            self._add_representative(encoding, complexity, period, genotype, coarse_data, freq)
    
    def get_representatives_by_complexity(self, target_complexity: float, tolerance: float = 0.5,
                                        sort_by: str = "frequency", ascending: bool = False,
                                        max_results: int = 9) -> List[CompactRepresentative]:
        """
        Get representatives for phenotypes with similar complexity
        
        Args:
            target_complexity: Target complexity value
            tolerance: Search within ±tolerance of target
            sort_by: "frequency" or "complexity" 
            ascending: Sort order (False = highest first)
            max_results: Maximum number of representatives to return
        """
        candidates = []
        
        # Search complexity bins within tolerance
        bin_size = 0.5
        min_complexity = target_complexity - tolerance
        max_complexity = target_complexity + tolerance
        
        min_bin = self._get_complexity_bin(min_complexity, bin_size)
        max_bin = self._get_complexity_bin(max_complexity, bin_size)
        
        current_bin = min_bin
        while current_bin <= max_bin:
            if current_bin in self.complexity_index:
                for phenotype in self.complexity_index[current_bin]:
                    complexity = self.complexities[phenotype]
                    if min_complexity <= complexity <= max_complexity:
                        frequency = self.frequencies[phenotype]
                        
                        # Add all representatives for this phenotype
                        for rep in self.representatives[phenotype]:
                            candidates.append((rep, frequency, complexity))
            
            current_bin += bin_size
        
        # Sort candidates
        if sort_by == "frequency":
            candidates.sort(key=lambda x: x[1], reverse=not ascending)
        else:  # sort by complexity
            candidates.sort(key=lambda x: x[2], reverse=not ascending)
        
        # Return top results
        return [rep for rep, _, _ in candidates[:max_results]]
    
    def get_representatives_by_phenotype(self, phenotype: str) -> List[CompactRepresentative]:
        """Get all stored representatives for a specific phenotype"""
        return self.representatives.get(phenotype, [])
    
    def get_complexity_distribution(self) -> Dict[float, int]:
        """Get distribution of complexities (binned)"""
        distribution = defaultdict(int)
        for phenotype, complexity in self.complexities.items():
            freq = self.frequencies[phenotype]
            if freq >= self.min_freq_for_storage:
                bin_val = self._get_complexity_bin(complexity)
                distribution[bin_val] += freq
        return dict(distribution)
    
    def get_memory_stats(self) -> Dict[str, Union[int, float]]:
        """Get memory usage statistics"""
        return {
            'current_memory_mb': self.current_memory_usage / (1024**2),
            'memory_limit_mb': self.memory_limit_bytes / (1024**2),
            'memory_usage_percent': (self.current_memory_usage / self.memory_limit_bytes) * 100,
            'stored_representatives': self.stored_representatives_count,
            'unique_phenotypes_with_reps': len(self.representatives),
            'total_samples_seen': self.total_samples_seen,
            'memory_overflows': self.memory_overflow_count
        }
    
    def save_to_disk(self, filepath: str):
        """Save representatives to disk for later analysis"""
        data = {
            'representatives': dict(self.representatives),
            'complexities': self.complexities,
            'frequencies': self.frequencies,
            'complexity_index': {k: list(v) for k, v in self.complexity_index.items()},
            'config': {
                'n_track': self.n_track,
                'max_reps_per_phenotype': self.max_reps_per_phenotype,
                'min_freq_for_storage': self.min_freq_for_storage,
                'memory_limit_bytes': self.memory_limit_bytes
            },
            'stats': self.get_memory_stats()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        file_size_mb = os.path.getsize(filepath) / (1024**2)
        print(f"Saved {self.stored_representatives_count:,} representatives to {filepath}")
        print(f"File size: {file_size_mb:.1f} MB")
    
    @classmethod
    def load_from_disk(cls, filepath: str):
        """Load representatives from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Recreate tracker
        config = data['config']
        tracker = cls(
            n_track=config['n_track'],
            max_representatives_per_phenotype=config['max_reps_per_phenotype'],
            min_frequency_for_storage=config['min_freq_for_storage'],
            memory_limit_gb=config['memory_limit_bytes'] / (1024**3)
        )
        
        # Restore data
        tracker.representatives = defaultdict(list, data['representatives'])
        tracker.complexities = data['complexities']
        tracker.frequencies = Counter(data['frequencies'])
        tracker.complexity_index = defaultdict(set, {k: set(v) for k, v in data['complexity_index'].items()})
        
        # Recalculate memory usage
        tracker.current_memory_usage = sum(
            rep.get_memory_size() 
            for reps in tracker.representatives.values() 
            for rep in reps
        )
        tracker.stored_representatives_count = sum(len(reps) for reps in tracker.representatives.values())
        
        return tracker

# Utility functions for analysis
def analyze_complexity_range(tracker: PhenotypeTrackerWithRepresentatives, 
                           complexity_min: float, complexity_max: float,
                           num_examples: int = 5):
    """Analyze representatives within a complexity range"""
    print(f"\n=== COMPLEXITY RANGE ANALYSIS: {complexity_min:.1f} - {complexity_max:.1f} ===")
    
    all_candidates = []
    for complexity in np.arange(complexity_min, complexity_max + 0.1, 0.1):
        reps = tracker.get_representatives_by_complexity(
            complexity, tolerance=0.1, sort_by="frequency", max_results=100
        )
        all_candidates.extend(reps)
    
    # Remove duplicates and sort by frequency
    unique_reps = list({rep.encoding: rep for rep in all_candidates}.values())
    unique_reps.sort(key=lambda x: x.frequency, reverse=True)
    
    print(f"Found {len(unique_reps)} unique phenotypes in this range")
    
    for i, rep in enumerate(unique_reps[:num_examples]):
        print(f"\n{i+1}. Complexity: {rep.complexity:.2f}, Frequency: {rep.frequency:,}, Period: {rep.period:.1f}")
        print(f"   Encoding: {rep.encoding[:50]}...")
        print(f"   Genotype sample: {rep.genotype[:5]}")

def complexity_troubleshooter(tracker: PhenotypeTrackerWithRepresentatives,
                            target_complexity: float,
                            mode: str = "highest_freq",
                            num_results: int = 9):
    """
    Main troubleshooting function for frequency-complexity diagrams
    
    Args:
        target_complexity: The complexity you want to investigate
        mode: "highest_freq", "lowest_freq", "most_diverse"
        num_results: Number of representatives to return
    """
    print(f"\n🔍 COMPLEXITY TROUBLESHOOTER: {target_complexity:.2f}")
    print(f"Mode: {mode}, Results: {num_results}")
    print("=" * 60)
    
    if mode == "highest_freq":
        reps = tracker.get_representatives_by_complexity(
            target_complexity, tolerance=0.25, sort_by="frequency", 
            ascending=False, max_results=num_results
        )
    elif mode == "lowest_freq":
        reps = tracker.get_representatives_by_complexity(
            target_complexity, tolerance=0.25, sort_by="frequency", 
            ascending=True, max_results=num_results
        )
    else:  # most_diverse
        reps = tracker.get_representatives_by_complexity(
            target_complexity, tolerance=0.5, sort_by="complexity", 
            ascending=False, max_results=num_results * 2
        )
        # Sample diverse representatives
        if len(reps) > num_results:
            indices = np.linspace(0, len(reps)-1, num_results, dtype=int)
            reps = [reps[i] for i in indices]
    
    if not reps:
        print(f"❌ No representatives found for complexity {target_complexity:.2f}")
        return []
    
    print(f"📊 Found {len(reps)} representatives:")
    for i, rep in enumerate(reps):
        print(f"\n{i+1:2d}. Complexity: {rep.complexity:6.2f} | Frequency: {rep.frequency:8,} | Period: {rep.period:6.1f}")
        print(f"     Encoding: {rep.encoding}")
        print(f"     Genotype: {rep.genotype[:5]} ...")
    
    return reps

if __name__ == "__main__":
    # Example usage
    print("PhenotypeTrackerWithRepresentatives - Memory efficient representative storage")
    print("For troubleshooting frequency-complexity diagrams")