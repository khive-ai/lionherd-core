# String Similarity

> Flexible string similarity algorithms for fuzzy matching, spell correction, and string comparison

## Overview

The `string_similarity` module provides a comprehensive suite of string similarity algorithms for comparing and matching text. It offers multiple algorithms (Jaro-Winkler, Levenshtein, Cosine, Hamming, SequenceMatcher) with a unified interface for fuzzy string matching, spell correction, and similarity scoring.

**Key Capabilities:**

- **Multiple Algorithms**: 5 built-in similarity algorithms optimized for different use cases
- **Unified Interface**: Single `string_similarity()` function supporting all algorithms
- **Flexible Matching**: Find single best match or all matches above threshold
- **Case-Insensitive**: Optional case-sensitive/insensitive comparison
- **Custom Algorithms**: Support for custom similarity functions via callable
- **Type Safety**: Enum-based algorithm names with string compatibility
- **Rich Results**: Structured match results with scores and indices

**When to Use String Similarity:**

- Fuzzy command/option matching (e.g., CLI argument suggestions)
- Spell correction and typo detection
- Duplicate detection in text collections
- String normalization and canonicalization
- User input validation with suggestions
- Data cleaning and deduplication

**Algorithm Selection Guide:**

- **Jaro-Winkler** (default): Best for short strings, names, and typos with common prefixes
- **Levenshtein**: Edit distance, good for general typo detection and spell correction
- **SequenceMatcher**: Python's difflib ratio, useful for longer text and sequence comparison
- **Hamming**: Same-length strings only, fast for fixed-width codes/identifiers
- **Cosine**: Character set similarity, useful for bag-of-characters comparison

## Module Contents

### Enums

#### `SimilarityAlgo`

Type-safe enum of available similarity algorithms.

```python
class SimilarityAlgo(Enum):
    JARO_WINKLER = "jaro_winkler"
    LEVENSHTEIN = "levenshtein"
    SEQUENCE_MATCHER = "sequence_matcher"
    HAMMING = "hamming"
    COSINE = "cosine"
```

**Usage:**

```python
from lionherd_core.libs.string_handlers import SimilarityAlgo

# Use enum values
algo = SimilarityAlgo.JARO_WINKLER

# Get all valid algorithm names
all_algos = SimilarityAlgo.allowed()
# ('jaro_winkler', 'levenshtein', 'sequence_matcher', 'hamming', 'cosine')
```

### Data Classes

#### `MatchResult`

Represents a string matching result with score and position information.

```python
@dataclass(frozen=True)
class MatchResult:
    word: str      # Matched word from candidate list
    score: float   # Similarity score (0.0 to 1.0)
    index: int     # Original index in candidate list
```

**Attributes:**

- `word` (str): The matching word from the candidate list
- `score` (float): Similarity score between 0 (no similarity) and 1 (identical)
- `index` (int): Original position in the candidate list (for stable ordering)

**Notes:**

Used internally by `string_similarity()` for sorting and filtering results.

### Type Aliases

```python
# Literal type for algorithm names
SIMILARITY_TYPE = Literal[
    "jaro_winkler",
    "levenshtein",
    "sequence_matcher",
    "hamming",
    "cosine",
]

# Type for similarity function signatures
SimilarityFunc = Callable[[str, str], float]
```

### Constants

#### `SIMILARITY_ALGO_MAP`

Maps algorithm names to their implementation functions.

```python
SIMILARITY_ALGO_MAP = {
    "jaro_winkler": jaro_winkler_similarity,
    "levenshtein": levenshtein_similarity,
    "sequence_matcher": sequence_matcher_similarity,
    "hamming": hamming_similarity,
    "cosine": cosine_similarity,
}
```

## Main Function

### `string_similarity()`

Find similar strings using specified similarity algorithm.

**Signature:**

```python
def string_similarity(
    word: str,
    correct_words: Sequence[str],
    algorithm: SIMILARITY_TYPE | SimilarityAlgo | Callable[[str, str], float] = "jaro_winkler",
    threshold: float = 0.0,
    case_sensitive: bool = False,
    return_most_similar: bool = False,
) -> str | list[str] | None: ...
```

**Parameters:**

**word** : str

The input string to find matches for. Automatically converted to string if other type provided.

**correct_words** : Sequence of str

List of candidate strings to compare against. Must not be empty. Elements automatically converted to strings.

**algorithm** : str or SimilarityAlgo or callable, default "jaro_winkler"

Similarity algorithm to use. Options:

- String literal: `"jaro_winkler"`, `"levenshtein"`, `"sequence_matcher"`, `"hamming"`, `"cosine"`
- Enum value: `SimilarityAlgo.JARO_WINKLER`, etc.
- Custom callable: Function taking two strings and returning float score (0.0-1.0)

**threshold** : float, default 0.0

Minimum similarity score for inclusion in results (0.0 to 1.0). Scores below threshold are filtered out.

- 0.0: Return all matches
- 0.5: Return moderately similar matches
- 0.8: Return very similar matches
- 1.0: Return only exact matches

**case_sensitive** : bool, default False

Whether to consider case when matching:

- False: Convert both input and candidates to lowercase before comparison
- True: Exact case matching required

**return_most_similar** : bool, default False

Return format control:

- False: Return list of all matches (sorted by score descending, then index ascending)
- True: Return only the single most similar match

**Returns:**

- str: Most similar match (if `return_most_similar=True`)
- list[str]: All matches above threshold (if `return_most_similar=False`)
- None: No matches found above threshold

**Raises:**

- ValueError: If `correct_words` is empty
- ValueError: If `threshold` not in range [0.0, 1.0]
- ValueError: If `algorithm` string is not a valid algorithm name

**Examples:**

```python
from lionherd_core.libs.string_handlers import string_similarity, SimilarityAlgo

# Basic fuzzy matching
candidates = ["apple", "application", "apply", "banana"]
matches = string_similarity("appl", candidates)
# ['apple', 'application', 'apply'] (all similar to "appl")

# Get single best match
best = string_similarity("appl", candidates, return_most_similar=True)
# 'apple' (highest similarity)

# Use threshold to filter weak matches
matches = string_similarity("appl", candidates, threshold=0.8)
# ['apple', 'apply'] (only high similarity matches)

# Case-sensitive matching
candidates = ["Apple", "apple", "APPLE"]
matches = string_similarity("apple", candidates, case_sensitive=True)
# ['apple'] (exact case match only)

# Different algorithms
matches = string_similarity(
    "colour",
    ["color", "collar", "cool"],
    algorithm=SimilarityAlgo.LEVENSHTEIN,
    threshold=0.7
)
# ['color'] (closest by edit distance)

# Custom similarity function
def custom_similarity(s1: str, s2: str) -> float:
    # Custom logic: only length difference
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1 - abs(len(s1) - len(s2)) / max_len

matches = string_similarity(
    "test",
    ["testing", "best", "rest"],
    algorithm=custom_similarity,
    threshold=0.5
)
# ['best', 'rest'] (similar length)

# No matches scenario
matches = string_similarity("xyz", ["abc", "def"], threshold=0.9)
# None (no high similarity matches)
```

**Notes:**

**Sorting Behavior:**

- Results sorted by score (descending), then original index (ascending)
- Stable ordering ensures consistent results across runs
- Case-sensitive mode: Filters to only exact max score matches after sorting

**Hamming Algorithm Special Handling:**

- Automatically skips candidates with different length than input
- Returns empty list (or None) if no same-length candidates exist

**Algorithm Performance:**

- Jaro-Winkler: O(n×m) where n,m are string lengths
- Levenshtein: O(n×m) with dynamic programming
- SequenceMatcher: Python difflib implementation
- Hamming: O(n) for same-length strings
- Cosine: O(n+m) set-based comparison

## Similarity Functions

### Jaro-Winkler Similarity

#### `jaro_winkler_similarity()`

Calculate Jaro-Winkler similarity with common prefix boost.

**Signature:**

```python
def jaro_winkler_similarity(s: str, t: str, scaling: float = 0.1) -> float: ...
```

**Parameters:**

- `s` (str): First input string
- `t` (str): Second input string
- `scaling` (float, default 0.1): Scaling factor for common prefix adjustment (must be 0.0-0.25)

**Returns:**

- float: Jaro-Winkler similarity score (0.0 to 1.0)

**Raises:**

- ValueError: If scaling factor not in range [0.0, 0.25]

**Examples:**

```python
from lionherd_core.libs.string_handlers import jaro_winkler_similarity

# Similar strings with common prefix
score = jaro_winkler_similarity("martha", "marhta")
# ~0.961 (high similarity, typo detected)

# Common prefix boost
score = jaro_winkler_similarity("application", "apply")
# ~0.727 (prefix "appl" boosts similarity)

# No prefix similarity
score = jaro_winkler_similarity("abc", "xyz")
# 0.0 (no common characters)
```

**Algorithm:**

1. Calculates base Jaro distance
2. Identifies common prefix (up to 4 characters)
3. Applies scaling boost: `jaro + (prefix_len × scaling × (1 - jaro))`

**Use Cases:**

- Name matching (people, places)
- Typo detection with common prefixes
- Short string comparison

#### `jaro_distance()`

Calculate base Jaro distance (used internally by Jaro-Winkler).

**Signature:**

```python
def jaro_distance(s: str, t: str) -> float: ...
```

**Parameters:**

- `s` (str): First input string
- `t` (str): Second input string

**Returns:**

- float: Jaro distance score (0.0 to 1.0)

**Algorithm:**

1. Identifies matching characters within match window: `max(len(s), len(t)) / 2 - 1`
2. Counts transpositions (matched chars in different order)
3. Returns: `(matches/len(s) + matches/len(t) + (matches-transpositions)/matches) / 3`

### Levenshtein Similarity

#### `levenshtein_similarity()`

Calculate normalized Levenshtein (edit distance) similarity.

**Signature:**

```python
def levenshtein_similarity(s1: str, s2: str) -> float: ...
```

**Parameters:**

- `s1` (str): First input string
- `s2` (str): Second input string

**Returns:**

- float: Similarity score (0.0 to 1.0), calculated as `1 - (distance / max_length)`

**Examples:**

```python
from lionherd_core.libs.string_handlers import levenshtein_similarity

# Single character difference
score = levenshtein_similarity("kitten", "sitting")
# 0.571 (3 edits: k→s, e→i, insert g)

# Identical strings
score = levenshtein_similarity("test", "test")
# 1.0 (no edits needed)

# Empty string handling
score = levenshtein_similarity("", "")
# 1.0 (both empty)

score = levenshtein_similarity("abc", "")
# 0.0 (maximum difference)
```

**Use Cases:**

- Spell correction
- Typo detection
- General string comparison

#### `levenshtein_distance()`

Calculate raw Levenshtein edit distance (number of edits).

**Signature:**

```python
def levenshtein_distance(a: str, b: str) -> int: ...
```

**Parameters:**

- `a` (str): First input string
- `b` (str): Second input string

**Returns:**

- int: Minimum number of single-character edits (insertions, deletions, substitutions) needed to transform `a` into `b`

**Algorithm:**

Dynamic programming with O(m×n) time and space complexity.

### Hamming Similarity

#### `hamming_similarity()`

Calculate Hamming similarity (proportion of matching positions).

**Signature:**

```python
def hamming_similarity(s1: str, s2: str) -> float: ...
```

**Parameters:**

- `s1` (str): First input string
- `s2` (str): Second input string (must be same length as `s1`)

**Returns:**

- float: Similarity score (0.0 to 1.0)
- 0.0 if strings have different lengths or either is empty

**Examples:**

```python
from lionherd_core.libs.string_handlers import hamming_similarity

# Same length, some differences
score = hamming_similarity("karolin", "kathrin")
# 0.571 (4 out of 7 positions match)

# Different lengths
score = hamming_similarity("abc", "abcd")
# 0.0 (length mismatch)

# Identical strings
score = hamming_similarity("test", "test")
# 1.0 (all positions match)
```

**Use Cases:**

- Fixed-width code comparison (barcodes, IDs)
- Binary string comparison
- Error detection in fixed-length data

**Constraints:**

- **Requires equal length strings**
- Returns 0.0 for length mismatches

### Cosine Similarity

#### `cosine_similarity()`

Calculate character-set-based cosine similarity.

**Signature:**

```python
def cosine_similarity(s1: str, s2: str) -> float: ...
```

**Parameters:**

- `s1` (str): First input string
- `s2` (str): Second input string

**Returns:**

- float: Cosine similarity score (0.0 to 1.0)
- 0.0 if either string is empty

**Examples:**

```python
from lionherd_core.libs.string_handlers import cosine_similarity

# Similar character sets
score = cosine_similarity("abc", "bca")
# 1.0 (identical character sets, order doesn't matter)

# Partial overlap
score = cosine_similarity("hello", "hola")
# ~0.632 (some common characters: h, l, o)

# No overlap
score = cosine_similarity("abc", "xyz")
# 0.0 (no common characters)
```

**Algorithm:**

1. Converts strings to character sets
2. Calculates intersection size
3. Returns: `|intersection| / sqrt(|set1| × |set2|)`

**Use Cases:**

- Bag-of-characters comparison
- Anagram detection (score = 1.0)
- Character overlap measurement

**Notes:**

- **Order-independent**: "abc" and "cba" have similarity 1.0
- **Set-based**: Character frequency ignored (duplicates don't affect score)

### Sequence Matcher Similarity

#### `sequence_matcher_similarity()`

Calculate similarity using Python's difflib.SequenceMatcher.

**Signature:**

```python
def sequence_matcher_similarity(s1: str, s2: str) -> float: ...
```

**Parameters:**

- `s1` (str): First input string
- `s2` (str): Second input string

**Returns:**

- float: Similarity ratio (0.0 to 1.0) from SequenceMatcher

**Examples:**

```python
from lionherd_core.libs.string_handlers import sequence_matcher_similarity

# Similar sequences
score = sequence_matcher_similarity("abcdef", "abxdef")
# ~0.833 (5 out of 6 characters match)

# Longer text comparison
score = sequence_matcher_similarity(
    "The quick brown fox",
    "The quick brown dog"
)
# ~0.895 (most words identical)
```

**Use Cases:**

- Longer text comparison
- Line-by-line diff operations
- Python standard library compatibility

**Notes:**

- Uses difflib's Gestalt pattern matching
- Good for general-purpose similarity
- Slightly slower than custom implementations

## Usage Patterns

### Pattern 1: CLI Argument Suggestion

```python
from lionherd_core.libs.string_handlers import string_similarity

def suggest_command(user_input: str, valid_commands: list[str]) -> None:
    """Suggest valid command if user provides invalid input."""
    if user_input in valid_commands:
        return  # Valid command

    # Find suggestions above 60% similarity
    suggestions = string_similarity(
        user_input,
        valid_commands,
        threshold=0.6,
        case_sensitive=False
    )

    if suggestions:
        print(f"Unknown command '{user_input}'. Did you mean:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")
    else:
        print(f"Unknown command '{user_input}'")

# Usage
valid_commands = ["start", "stop", "restart", "status", "deploy"]
suggest_command("stat", valid_commands)
# Output:
# Unknown command 'stat'. Did you mean:
#   - start
#   - status
```

### Pattern 2: Spell Correction

```python
from lionherd_core.libs.string_handlers import string_similarity, SimilarityAlgo

def spell_correct(word: str, dictionary: list[str]) -> str:
    """Find closest dictionary word to correct spelling."""
    # Use Levenshtein for edit distance-based correction
    corrected = string_similarity(
        word,
        dictionary,
        algorithm=SimilarityAlgo.LEVENSHTEIN,
        threshold=0.7,
        case_sensitive=False,
        return_most_similar=True
    )

    return corrected if corrected else word

# Usage
dictionary = ["hello", "world", "python", "programming"]
print(spell_correct("helo", dictionary))    # "hello"
print(spell_correct("wrold", dictionary))   # "world"
print(spell_correct("pythno", dictionary))  # "python"
```

### Pattern 3: Deduplication

```python
from lionherd_core.libs.string_handlers import string_similarity

def find_duplicates(items: list[str], threshold: float = 0.9) -> list[tuple[str, str]]:
    """Find near-duplicate pairs in a list."""
    duplicates = []

    for i, item1 in enumerate(items):
        for item2 in items[i+1:]:
            matches = string_similarity(
                item1,
                [item2],
                threshold=threshold,
                case_sensitive=False
            )
            if matches:
                duplicates.append((item1, item2))

    return duplicates

# Usage
names = ["John Smith", "Jon Smith", "Jane Doe", "John Smyth"]
dupes = find_duplicates(names, threshold=0.85)
# [('John Smith', 'Jon Smith'), ('John Smith', 'John Smyth')]
```

### Pattern 4: Fuzzy Lookup

```python
from lionherd_core.libs.string_handlers import string_similarity

class FuzzyDict:
    """Dictionary with fuzzy key matching."""

    def __init__(self, data: dict[str, any], threshold: float = 0.8):
        self.data = data
        self.threshold = threshold

    def get(self, key: str, default=None):
        """Get value with fuzzy key matching."""
        # Try exact match first
        if key in self.data:
            return self.data[key]

        # Try fuzzy match
        match = string_similarity(
            key,
            list(self.data.keys()),
            threshold=self.threshold,
            return_most_similar=True
        )

        return self.data[match] if match else default

# Usage
config = FuzzyDict({
    "database_host": "localhost",
    "database_port": 5432,
    "api_key": "secret123"
})

print(config.get("db_host"))       # "localhost" (fuzzy matched)
print(config.get("api_key"))       # "secret123" (exact match)
print(config.get("unknown", "N/A")) # "N/A" (no match)
```

### Pattern 5: Algorithm Comparison

```python
from lionherd_core.libs.string_handlers import string_similarity, SimilarityAlgo

def compare_algorithms(word: str, candidates: list[str]) -> dict[str, list[str]]:
    """Compare results from different similarity algorithms."""
    results = {}

    algorithms = [
        SimilarityAlgo.JARO_WINKLER,
        SimilarityAlgo.LEVENSHTEIN,
        SimilarityAlgo.COSINE,
        SimilarityAlgo.SEQUENCE_MATCHER,
    ]

    for algo in algorithms:
        matches = string_similarity(
            word,
            candidates,
            algorithm=algo,
            threshold=0.7
        )
        results[algo.value] = matches or []

    return results

# Usage
word = "colour"
candidates = ["color", "collar", "cool", "court", "coal"]
results = compare_algorithms(word, candidates)

for algo, matches in results.items():
    print(f"{algo:20} -> {matches}")
# Output:
# jaro_winkler         -> ['color', 'collar']
# levenshtein          -> ['color', 'collar', 'court']
# cosine               -> ['color', 'coal', 'cool']
# sequence_matcher     -> ['color', 'collar']
```

### Pattern 6: Custom Similarity

```python
from lionherd_core.libs.string_handlers import string_similarity

def weighted_similarity(s1: str, s2: str) -> float:
    """Custom similarity: 80% character overlap + 20% length similarity."""
    # Character overlap component
    set1, set2 = set(s1.lower()), set(s2.lower())
    if not set1 or not set2:
        char_sim = 0.0
    else:
        intersection = set1.intersection(set2)
        char_sim = len(intersection) / ((len(set1) * len(set2)) ** 0.5)

    # Length similarity component
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        len_sim = 1.0
    else:
        len_sim = 1 - abs(len(s1) - len(s2)) / max_len

    # Weighted combination
    return 0.8 * char_sim + 0.2 * len_sim

# Usage
candidates = ["testing", "test", "fest"]
matches = string_similarity(
    "test",
    candidates,
    algorithm=weighted_similarity,
    threshold=0.7
)
print(matches)  # ['test', 'testing', 'fest']
```

## Common Pitfalls

### Pitfall 1: Wrong Algorithm for Use Case

**Issue**: Using Hamming similarity for variable-length strings.

```python
# BAD: Hamming requires equal length
score = hamming_similarity("abc", "abcd")
# Returns 0.0 (should use different algorithm)
```

**Solution**: Choose appropriate algorithm for your use case.

```python
# GOOD: Use Levenshtein for variable-length strings
score = levenshtein_similarity("abc", "abcd")
# Returns 0.75 (meaningful result)
```

### Pitfall 2: Threshold Too High

**Issue**: No matches found due to overly strict threshold.

```python
# BAD: Threshold too high for fuzzy matching
matches = string_similarity(
    "colour",
    ["color", "collar"],
    threshold=0.95  # Too strict
)
# Returns None (no 95%+ matches)
```

**Solution**: Adjust threshold based on use case and testing.

```python
# GOOD: Reasonable threshold for typo tolerance
matches = string_similarity(
    "colour",
    ["color", "collar"],
    threshold=0.75  # More forgiving
)
# Returns ['color', 'collar']
```

### Pitfall 3: Case Sensitivity Mismatch

**Issue**: Unexpected results due to case handling.

```python
# BAD: Case-sensitive with mixed-case input
matches = string_similarity(
    "Apple",
    ["apple", "APPLE", "application"],
    case_sensitive=True
)
# Returns [] (no exact case matches)
```

**Solution**: Use `case_sensitive=False` for user input.

```python
# GOOD: Case-insensitive for flexibility
matches = string_similarity(
    "Apple",
    ["apple", "APPLE", "application"],
    case_sensitive=False
)
# Returns ['apple', 'APPLE', 'application']
```

### Pitfall 4: Performance with Large Candidate Lists

**Issue**: Slow performance when comparing against thousands of candidates.

```python
# BAD: O(n×m) for each candidate
large_dict = load_dictionary()  # 100,000 words
match = string_similarity("test", large_dict, threshold=0.8)
# Takes seconds to complete
```

**Solution**: Pre-filter candidates or use indexing strategies.

```python
# GOOD: Pre-filter by length/prefix
def fast_similarity(word, candidates, threshold=0.8):
    # Filter candidates by reasonable length difference
    max_len_diff = int(len(word) * 0.3)  # 30% length tolerance
    filtered = [
        c for c in candidates
        if abs(len(c) - len(word)) <= max_len_diff
    ]

    # Or filter by common prefix
    prefix = word[:2].lower()
    filtered = [c for c in candidates if c[:2].lower() == prefix]

    return string_similarity(word, filtered, threshold=threshold)
```

### Pitfall 5: Empty Candidate List

**Issue**: Passing empty list raises ValueError.

```python
# BAD: Empty candidate list
matches = string_similarity("test", [])
# ValueError: correct_words must not be empty
```

**Solution**: Validate inputs before calling.

```python
# GOOD: Check for empty list
def safe_similarity(word, candidates, **kwargs):
    if not candidates:
        return None
    return string_similarity(word, candidates, **kwargs)
```

## Design Rationale

### Why Multiple Algorithms?

Different similarity algorithms optimize for different use cases:

1. **Jaro-Winkler**: Optimized for short strings and common prefixes (names, commands)
2. **Levenshtein**: Edit distance, universal for spell correction
3. **Hamming**: Fast for fixed-width data (IDs, codes)
4. **Cosine**: Character set overlap, order-independent
5. **SequenceMatcher**: Python stdlib compatibility, longer text

Providing multiple algorithms lets users choose the best fit for their data characteristics.

### Why Unified Interface?

Single `string_similarity()` function with `algorithm` parameter provides:

1. **Consistent API**: Same parameters across all algorithms
2. **Easy Switching**: Change algorithm without refactoring code
3. **Custom Functions**: Support for user-defined similarity metrics
4. **Discoverability**: All algorithms accessible through one function

### Why Threshold Parameter?

Threshold filtering (0.0-1.0) enables:

1. **Quality Control**: Filter weak matches, reduce false positives
2. **Performance**: Early termination for low-scoring candidates
3. **Tunable Precision/Recall**: Adjust strictness for use case
4. **Zero Threshold Default**: Return all matches, let caller filter

### Why Case-Insensitive Default?

Most fuzzy matching use cases (CLI suggestions, spell correction, user input validation) benefit from case-insensitive comparison:

1. **User Input**: Humans inconsistent with capitalization
2. **Typo Tolerance**: Case errors common in quick typing
3. **Opt-in Strictness**: Case-sensitive available when needed

Default to most common use case, allow override for exact matching.

### Why Return List vs Single?

`return_most_similar` parameter supports two common patterns:

1. **List (default)**: Show all suggestions to user (CLI "did you mean?")
2. **Single**: Auto-correct to best match (spell correction)

Both patterns common enough to warrant first-class support.

## See Also

- **Related Modules**:
  - [extract_json](extract_json.md): Extract JSON from mixed text
  - [fuzzy_json](fuzzy_json.md): Fault-tolerant JSON parsing
  - [to_num](to_num.md): String to number conversion
- **Related Use Cases**:
  - CLI Argument Validation: Use for command/option suggestions
  - Data Cleaning: Deduplication and normalization
  - Search/Autocomplete: Fuzzy search implementations

## Examples

> **Note**: For production-level examples demonstrating advanced integration patterns, see the tutorials linked below.

### Tutorial: CLI Command Suggestion System

**See [Tutorial #90](https://github.com/khive-ai/lionherd-core/issues/90)** for a complete production example of building a CLI command parser with intelligent fuzzy matching, auto-correction, and multi-suggestion UX (~55 lines).

### Tutorial: Fuzzy Data Deduplication

**See [Tutorial #91](https://github.com/khive-ai/lionherd-core/issues/91)** for a production-grade data deduplication system using fuzzy matching with Levenshtein algorithm for CRM/data cleaning use cases (~80 lines).

### Tutorial: Multi-Algorithm Consensus Matching

**See [Tutorial #92](https://github.com/khive-ai/lionherd-core/issues/92)** for an advanced matching pattern using multiple similarity algorithms with voting/consensus for high-confidence results (~45 lines).

### Example 1: Progressive Threshold Search

```python
from lionherd_core.libs.string_handlers import string_similarity

def progressive_search(
    query: str,
    candidates: list[str],
    thresholds: list[float] = [0.95, 0.85, 0.75, 0.60]
) -> tuple[list[str], float]:
    """Search with progressively lower thresholds until matches found."""
    for threshold in thresholds:
        matches = string_similarity(
            query,
            candidates,
            threshold=threshold,
            case_sensitive=False
        )

        if matches:
            return matches, threshold

    return [], 0.0

# Usage
dictionary = ["hello", "world", "python", "programming", "algorithm"]

# High-confidence match
matches, confidence = progressive_search("helo", dictionary)
print(f"Matches at {confidence:.0%} confidence: {matches}")
# Output: Matches at 75% confidence: ['hello']

# Lower-confidence match
matches, confidence = progressive_search("wrld", dictionary)
print(f"Matches at {confidence:.0%} confidence: {matches}")
# Output: Matches at 60% confidence: ['world']
```

### Tutorial: Custom Similarity Algorithms (Phonetic Matching)

**See [Tutorial #93](https://github.com/khive-ai/lionherd-core/issues/93)** for a complete example implementing custom similarity algorithms with string_similarity(), demonstrating Soundex phonetic matching for name variations (~55 lines).
