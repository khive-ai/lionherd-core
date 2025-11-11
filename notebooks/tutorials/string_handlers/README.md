# String Handlers Tutorials

Master string similarity algorithms and fuzzy matching patterns for user input validation, deduplication, and approximate matching. Learn when to use different algorithms (Jaro-Winkler, Levenshtein, Soundex) and how to combine them for robust matching.

## Overview

These tutorials teach you to use lionherd-core's string similarity utilities:

- **Similarity Algorithms**: Jaro-Winkler, Levenshtein, Soundex, Hamming
- **Fuzzy Matching**: Handle typos, abbreviations, and phonetic similarities
- **Deduplication**: Identify near-duplicate strings using similarity thresholds
- **Multi-Algorithm Consensus**: Combine algorithms for higher confidence matching

## Prerequisites

- Python 3.11+
- Basic understanding of string similarity concepts (helpful but not required)

## Quick Start

```bash
pip install lionherd-core
jupyter notebook cli_fuzzy_matching.ipynb
```

## Tutorials (4)

| Tutorial | Time | What You'll Learn |
|----------|------|-------------------|
| [**CLI Fuzzy Matching**](./cli_fuzzy_matching.ipynb) | 15-20min | Match user commands to valid options using Jaro-Winkler similarity |
| [**Fuzzy Deduplication**](./fuzzy_deduplication.ipynb) | 15-25min | Detect and merge near-duplicate records using similarity thresholds |
| [**Consensus Matching**](./consensus_matching.ipynb) | 15-20min | Combine multiple algorithms (voting) for confident matching |
| [**Phonetic Matching**](./phonetic_matching.ipynb) | 15-30 min | Match names that sound similar using Soundex algorithm |

## Learning Path

**Recommended order** (1 hour total):

1. **CLI Fuzzy Matching** - Understand Jaro-Winkler and threshold tuning
2. **Fuzzy Deduplication** - Apply similarity to real-world deduplication
3. **Consensus Matching** - Learn multi-algorithm voting patterns
4. **Phonetic Matching** - Handle phonetic similarities (names, search)

**Outcome**: Choose and apply the right similarity algorithm for your use case

## Key Concepts

### Similarity Algorithms

lionherd-core provides 4 algorithms via `string_similarity()`:

| Algorithm | Best For | Example | Similarity |
|-----------|----------|---------|------------|
| **Jaro-Winkler** | Typos, prefix matches | "robert" vs "robrt" | 0.96 |
| **Levenshtein** | Edit distance (insertions, deletions, substitutions) | "kitten" vs "sitting" | 0.57 |
| **Soundex** | Phonetic matching (names) | "Smith" vs "Smyth" | 1.0 (match) |
| **Hamming** | Same-length strings (bit differences) | "karolin" vs "kathrin" | 0.57 |

```python
from lionherd_core.libs.string_handlers import string_similarity

# Jaro-Winkler (good for typos)
string_similarity("robert", "robrt", algo="jaro_winkler")  # 0.96

# Levenshtein (edit distance)
string_similarity("kitten", "sitting", algo="levenshtein")  # 0.57

# Soundex (phonetic)
string_similarity("Smith", "Smyth", algo="soundex")  # 1.0
```

**Tutorial**: [CLI Fuzzy Matching](./cli_fuzzy_matching.ipynb)

### When to Use Each Algorithm

**Jaro-Winkler** (Default recommendation):

- ✅ User input with typos ("deplpy" → "deploy")
- ✅ Command matching ("lis" → "list")
- ✅ Field name variations ("usr_name" → "user_name")
- ❌ Significantly different lengths (biased toward prefixes)

**Levenshtein** (Edit distance):

- ✅ Spell checking
- ✅ Text diff/similarity
- ✅ Any string length combination
- ❌ Phonetic similarities ("Smith" vs "Smyth" = low similarity)

**Soundex** (Phonetic):

- ✅ Name matching ("Catherine" ↔ "Kathryn")
- ✅ Search with pronunciation variants
- ✅ Data entry where spelling varies
- ❌ Non-phonetic typos ("Smit" vs "Smith" = different codes)

**Hamming** (Bit-level):

- ✅ Fixed-length codes/IDs
- ✅ DNA sequences
- ✅ Binary data comparison
- ❌ Variable-length strings (requires same length)

### Threshold Tuning

Similarity scores range from 0.0 (completely different) to 1.0 (identical). Choose thresholds based on tolerance:

| Threshold | Interpretation | Use Case |
|-----------|----------------|----------|
| 0.95+ | Near-exact match | Minor typos only |
| 0.85-0.94 | High similarity | User input fuzzy matching |
| 0.75-0.84 | Moderate similarity | Deduplication with caution |
| 0.60-0.74 | Low similarity | Exploratory matching |
| <0.60 | Likely false positive | Too lenient |

**Tutorial**: [CLI Fuzzy Matching](./cli_fuzzy_matching.ipynb) demonstrates threshold tuning in practice.

## Common Patterns

### Pattern 1: Command Matching (CLI/Chatbots)

Match user input to valid commands:

```python
from lionherd_core.libs.string_handlers import string_similarity

def match_command(user_input: str, valid_commands: list[str], threshold: float = 0.75):
    """Match user input to closest valid command."""
    best_match = None
    best_score = 0.0

    for cmd in valid_commands:
        score = string_similarity(user_input.lower(), cmd.lower(), algo="jaro_winkler")
        if score > best_score and score >= threshold:
            best_score = score
            best_match = cmd

    return best_match, best_score

# Example usage
commands = ["deploy", "rollback", "status", "logs"]
match, score = match_command("deplpy", commands)  # ("deploy", 0.93)
```

**Tutorial**: [CLI Fuzzy Matching](./cli_fuzzy_matching.ipynb)

### Pattern 2: Deduplication

Identify and merge near-duplicate records:

```python
from lionherd_core.libs.string_handlers import string_similarity

def find_duplicates(items: list[dict], key: str, threshold: float = 0.85):
    """Find near-duplicate items based on a key field."""
    duplicates = []

    for i, item1 in enumerate(items):
        for item2 in items[i+1:]:
            score = string_similarity(
                str(item1[key]),
                str(item2[key]),
                algo="jaro_winkler"
            )
            if score >= threshold:
                duplicates.append((item1, item2, score))

    return duplicates

# Example: Find duplicate users
users = [
    {"name": "John Smith", "email": "john@example.com"},
    {"name": "Jon Smith", "email": "jon@example.com"},  # Duplicate
    {"name": "Jane Doe", "email": "jane@example.com"}
]

dupes = find_duplicates(users, key="name", threshold=0.85)
# [({'name': 'John Smith', ...}, {'name': 'Jon Smith', ...}, 0.93)]
```

**Tutorial**: [Fuzzy Deduplication](./fuzzy_deduplication.ipynb)

### Pattern 3: Multi-Algorithm Consensus

Combine algorithms for higher confidence:

```python
from lionherd_core.libs.string_handlers import string_similarity

def consensus_match(str1: str, str2: str, threshold: float = 0.75) -> tuple[bool, dict]:
    """Match using multiple algorithms with voting."""
    algorithms = ["jaro_winkler", "levenshtein", "soundex"]
    scores = {}
    votes = 0

    for algo in algorithms:
        score = string_similarity(str1, str2, algo=algo)
        scores[algo] = score
        if score >= threshold:
            votes += 1

    # Require majority vote (2/3 algorithms agree)
    is_match = votes >= 2

    return is_match, scores

# Example: High-confidence name matching
is_match, scores = consensus_match("Catherine", "Kathryn")
# (True, {"jaro_winkler": 0.78, "levenshtein": 0.62, "soundex": 1.0})
# Two algorithms voted yes (jaro_winkler + soundex), consensus reached
```

**Tutorial**: [Consensus Matching](./consensus_matching.ipynb)

### Pattern 4: Phonetic Name Search

Search names by pronunciation:

```python
from lionherd_core.libs.string_handlers import string_similarity

def phonetic_search(query: str, names: list[str]) -> list[tuple[str, float]]:
    """Search names phonetically (handles spelling variations)."""
    matches = []

    for name in names:
        score = string_similarity(query, name, algo="soundex")
        if score == 1.0:  # Soundex returns 1.0 for phonetic match, 0.0 otherwise
            # Refine with Jaro-Winkler for ranking
            refined_score = string_similarity(query, name, algo="jaro_winkler")
            matches.append((name, refined_score))

    # Sort by refined score
    return sorted(matches, key=lambda x: x[1], reverse=True)

# Example: Find "Smith" variations
names = ["Smith", "Smyth", "Schmidt", "Smit", "Jones"]
results = phonetic_search("Smith", names)
# [("Smith", 1.0), ("Smyth", 0.87), ("Smit", 0.93)]
```

**Tutorial**: [Phonetic Matching](./phonetic_matching.ipynb)

## Production Considerations

### Performance

| Algorithm | Complexity | Typical Speed (1000 comparisons) |
|-----------|------------|----------------------------------|
| Jaro-Winkler | O(n×m) | ~5ms |
| Levenshtein | O(n×m) | ~10ms |
| Soundex | O(n) | ~2ms (encode once, compare codes) |
| Hamming | O(n) | ~1ms (same-length only) |

Where n, m = string lengths.

**Optimization tips**:

- **Pre-encode** Soundex codes for large datasets (avoid re-encoding)
- **Short-circuit** on length differences (if |len(s1) - len(s2)| > threshold, skip)
- **Index** by first character or prefix to reduce comparison count

### Threshold Selection

Test thresholds on real data samples:

```python
def evaluate_threshold(pairs: list[tuple[str, str, bool]], threshold: float):
    """Evaluate precision/recall for a threshold on labeled pairs."""
    tp = fp = tn = fn = 0

    for str1, str2, is_match in pairs:
        score = string_similarity(str1, str2, algo="jaro_winkler")
        predicted_match = score >= threshold

        if predicted_match and is_match:
            tp += 1
        elif predicted_match and not is_match:
            fp += 1
        elif not predicted_match and not is_match:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {"precision": precision, "recall": recall}

# Test thresholds from 0.6 to 0.95
for t in [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    metrics = evaluate_threshold(labeled_pairs, t)
    print(f"Threshold {t}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}")
```

### Error Handling

```python
from lionherd_core.libs.string_handlers import string_similarity

def safe_similarity(str1: str, str2: str, algo: str = "jaro_winkler") -> float:
    """Compute similarity with validation."""
    # Validate inputs
    if not str1 or not str2:
        return 0.0

    # Hamming requires same length
    if algo == "hamming" and len(str1) != len(str2):
        raise ValueError(f"Hamming requires same-length strings (got {len(str1)} and {len(str2)})")

    return string_similarity(str1, str2, algo=algo)
```

## Troubleshooting

### Common Issues

**Issue**: Soundex matches too many unrelated names
**Solution**: Use Soundex for filtering, then refine with Jaro-Winkler for ranking

**Issue**: Jaro-Winkler gives high scores to very different strings with same prefix
**Solution**: Add length difference check before computing similarity

**Issue**: Levenshtein is too slow for large datasets
**Solution**: Use indexing (first character, prefix tree) to reduce comparison count

## Related Resources

- **API Reference**: [libs/string_handlers](../../../docs/api/libs/string_handlers/)
- **String Similarity**: [string_similarity](../../../docs/api/libs/string_handlers/string_similarity.md)
- **Wikipedia**: [Jaro-Winkler](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance), [Levenshtein](https://en.wikipedia.org/wiki/Levenshtein_distance), [Soundex](https://en.wikipedia.org/wiki/Soundex)

## Contributing

Found issues or have suggestions? Open an issue at [lionherd-core GitHub](https://github.com/khive-ai/lionherd-core/issues).
