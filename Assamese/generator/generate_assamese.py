# generate_assamese.py
# Assamese Phase 3 synthetic reasoning dataset generator.
#
# Generates:
#   - train.jsonl
#   - val.jsonl
#   - test.jsonl
#
# Ground-truth answers are generated deterministically from
# hidden synthetic numeric values.
#
# Seed/randomness is controlled by SEED below.


import json
import random
from pathlib import Path


from templates_assamese import (
    DOMAINS,
    EQUAL_DOMAIN_TEMPLATES,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 2026909004

TRAIN_SIZE = 6400
VAL_SIZE = 800
TEST_SIZE = 800

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

random.seed(SEED)


# ============================================================
# ENTITY NAMES
# ============================================================

NAMES = [
    "ৰাহুল",
    "অমিত",
    "বিকাশ",
    "ৰবি",
    "সুমিত",
    "অজয়",
    "দীপক",
    "আকাশ",
    "ৰোহিত",
    "সঞ্জয়",
    "মনোজ",
    "বিশাল",
    "নিলয়",
    "অনিক",
    "সৌৰভ",
    "অভিজিৎ",
    "প্ৰতীক",
    "ধ্ৰুৱ",
    "ময়ূৰ",
    "হৰ্ষ",
    "নিখিল",
    "গৌৰৱ",
    "তুষাৰ",
    "বিবেক",
    "সন্দীপ",
    "মিতেশ",
    "যোগেশ",
    "হিতেশ",
    "চিৰাগ",
    "কেতন",
    "তেজস",
    "বৰুণ",
    "মোহন",
    "কুশল",
    "পংকজ",
    "অনিল",
    "বিনয়",
    "দেৱজিত",
    "সমীৰ",
    "হেমন্ত",
]


# ============================================================
# DOMAINS
# ============================================================

DOMAIN_NAMES = list(DOMAINS.keys())


# ============================================================
# TASK TYPES
# ============================================================

TASK_TYPES = [
    "direct_comparison",
    "greater_smaller",
    "equal",
    "two_hop",
    "three_hop",
    "ordering",
    "reversed_wording",
    "distractor",
]


# ============================================================
# BASIC UTILITIES
# ============================================================

def choose_names(n):
    """
    Choose n unique Assamese names.
    """
    return random.sample(NAMES, n)


def make_values(names):
    """
    Assign unique synthetic numeric values.

    The model never sees these values.
    They are only used to calculate deterministic ground truth.
    """

    values = random.sample(
        range(100, 201),
        len(names),
    )

    return dict(zip(names, values))


def choose_domain():
    """
    Randomly select a reasoning domain.
    """
    return random.choice(DOMAIN_NAMES)


def make_example(
    example_id,
    task,
    difficulty,
    domain,
    statements,
    question,
    answer,
):
    """
    Create a standard dataset example.

    Domain metadata is stored separately from the natural-language
    input so it can be used later for evaluation and analysis.
    """

    return {
        "id": example_id,
        "task": task,
        "difficulty": difficulty,
        "domain": domain,
        "input": "\n".join(
            statements + [question]
        ),
        "answer": answer,
    }


# ============================================================
# COMPARISON STATEMENT
# ============================================================

def comparison_statement(a, b, values, domain):
    """
    Generate a natural Assamese statement representing
    the relation between a and b.
    """

    domain_data = DOMAINS[domain]

    if values[a] > values[b]:

        template = random.choice(
            domain_data["greater"]
        )

        return template.format(
            a=a,
            b=b,
        )

    else:

        template = random.choice(
            domain_data["smaller"]
        )

        return template.format(
            a=a,
            b=b,
        )


# ============================================================
# GREATER QUESTION
# ============================================================

def greater_question(a, b, domain):
    """
    Question asking which entity has the greater property.
    """

    template = random.choice(
        DOMAINS[domain]["greater_question"]
    )

    return template.format(
        a=a,
        b=b,
    )


# ============================================================
# SMALLER QUESTION
# ============================================================

def smaller_question(a, b, domain):
    """
    Question asking which entity has the smaller property.
    """

    template = random.choice(
        DOMAINS[domain]["smaller_question"]
    )

    return template.format(
        a=a,
        b=b,
    )


# ============================================================
# ORDERING QUESTION
# ============================================================

def order_question(domain):
    """
    Generate an ordering question.
    """

    template = random.choice(
        DOMAINS[domain]["ordering"]
    )

    return template


# ============================================================
# TASK 1 — DIRECT COMPARISON
# ============================================================

def generate_direct_comparison(example_id):

    names = choose_names(2)
    values = make_values(names)

    a, b = names
    domain = choose_domain()

    statement = comparison_statement(
        a,
        b,
        values,
        domain,
    )

    question = greater_question(
        a,
        b,
        domain,
    )

    if values[a] > values[b]:
        answer = a
    else:
        answer = b

    return make_example(
        example_id,
        "direct_comparison",
        1,
        domain,
        [statement],
        question,
        answer,
    )


# ============================================================
# TASK 2 — GREATER / SMALLER
# ============================================================

def generate_greater_smaller(example_id):

    names = choose_names(2)
    values = make_values(names)

    a, b = names
    domain = choose_domain()

    statement = comparison_statement(
        a,
        b,
        values,
        domain,
    )

    if values[a] > values[b]:

        question = greater_question(
            a,
            b,
            domain,
        )

        answer = a

    else:

        question = smaller_question(
            a,
            b,
            domain,
        )

        answer = a

    return make_example(
        example_id,
        "greater_smaller",
        1,
        domain,
        [statement],
        question,
        answer,
    )


# ============================================================
# TASK 3 — EQUALITY
# ============================================================

def generate_equal(example_id):

    names = choose_names(2)

    a, b = names

    domain = choose_domain()

    statement = random.choice(
        EQUAL_DOMAIN_TEMPLATES[domain]
    ).format(
        a=a,
        b=b,
    )

    question = (
        f"{a} আৰু {b}ৰ মাজত এই ক্ষেত্ৰত কি সম্পৰ্ক আছে?"
    )

    return make_example(
        example_id,
        "equal",
        1,
        domain,
        [statement],
        question,
        "সমান",
    )


# ============================================================
# TASK 4 — TWO-HOP REASONING
# ============================================================

def generate_two_hop(example_id):

    names = choose_names(3)
    values = make_values(names)

    domain = choose_domain()

    ordered = sorted(
        names,
        key=lambda x: values[x],
        reverse=True,
    )

    a, b, c = ordered

    s1 = comparison_statement(
        a,
        b,
        values,
        domain,
    )

    s2 = comparison_statement(
        b,
        c,
        values,
        domain,
    )

    question = greater_question(
        a,
        c,
        domain,
    )

    return make_example(
        example_id,
        "two_hop",
        2,
        domain,
        [s1, s2],
        question,
        a,
    )


# ============================================================
# TASK 5 — THREE-HOP REASONING
# ============================================================

def generate_three_hop(example_id):

    names = choose_names(4)
    values = make_values(names)

    domain = choose_domain()

    ordered = sorted(
        names,
        key=lambda x: values[x],
        reverse=True,
    )

    a, b, c, d = ordered

    s1 = comparison_statement(
        a,
        b,
        values,
        domain,
    )

    s2 = comparison_statement(
        b,
        c,
        values,
        domain,
    )

    s3 = comparison_statement(
        c,
        d,
        values,
        domain,
    )

    question = greater_question(
        a,
        d,
        domain,
    )

    return make_example(
        example_id,
        "three_hop",
        3,
        domain,
        [s1, s2, s3],
        question,
        a,
    )


# ============================================================
# TASK 6 — ORDERING
# ============================================================

def generate_ordering(example_id):

    names = choose_names(4)
    values = make_values(names)

    domain = choose_domain()

    ordered = sorted(
        names,
        key=lambda x: values[x],
    )

    statements = []

    # Create a chain connecting all entities.
    for i in range(len(ordered) - 1):

        lower = ordered[i]
        higher = ordered[i + 1]

        statements.append(
            comparison_statement(
                higher,
                lower,
                values,
                domain,
            )
        )

    question = order_question(domain)

    # Natural list format rather than symbolic "<".
    answer = ", ".join(ordered)

    return make_example(
        example_id,
        "ordering",
        3,
        domain,
        statements,
        question,
        answer,
    )


# ============================================================
# TASK 7 — REVERSED WORDING
# ============================================================

def generate_reversed_wording(example_id):

    names = choose_names(2)
    values = make_values(names)

    a, b = names
    domain = choose_domain()

    # Deliberately create a statement whose wording
    # requires reversing the relation when answering.

    if values[a] > values[b]:

        # Statement:
        # b is smaller than a.
        statement = random.choice(
            DOMAINS[domain]["smaller"]
        ).format(
            a=b,
            b=a,
        )

        question = greater_question(
            a,
            b,
            domain,
        )

        answer = a

    else:

        # Statement:
        # a is smaller than b.
        statement = random.choice(
            DOMAINS[domain]["smaller"]
        ).format(
            a=a,
            b=b,
        )

        question = greater_question(
            a,
            b,
            domain,
        )

        answer = b

    return make_example(
        example_id,
        "reversed_wording",
        2,
        domain,
        [statement],
        question,
        answer,
    )


# ============================================================
# TASK 8 — DISTRACTOR
# ============================================================

def generate_distractor(example_id):

    names = choose_names(4)
    values = make_values(names)

    domain = choose_domain()

    # Three relevant entities.
    relevant = names[:3]

    ordered = sorted(
        relevant,
        key=lambda x: values[x],
        reverse=True,
    )

    a, b, c = ordered

    # Fourth entity is a distractor.
    d = names[3]

    relevant_1 = comparison_statement(
        a,
        b,
        values,
        domain,
    )

    relevant_2 = comparison_statement(
        b,
        c,
        values,
        domain,
    )

    # Distractor contains the fourth entity.
    distractor_target = random.choice(
        [a, b, c]
    )

    distractor = comparison_statement(
        d,
        distractor_target,
        values,
        domain,
    )

    statements = [
        relevant_1,
        distractor,
        relevant_2,
    ]

    random.shuffle(statements)

    question = greater_question(
        a,
        c,
        domain,
    )

    return make_example(
        example_id,
        "distractor",
        3,
        domain,
        statements,
        question,
        a,
    )


# ============================================================
# GENERATOR DISPATCH
# ============================================================

GENERATORS = {

    "direct_comparison":
        generate_direct_comparison,

    "greater_smaller":
        generate_greater_smaller,

    "equal":
        generate_equal,

    "two_hop":
        generate_two_hop,

    "three_hop":
        generate_three_hop,

    "ordering":
        generate_ordering,

    "reversed_wording":
        generate_reversed_wording,

    "distractor":
        generate_distractor,
}


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(
    size,
    prefix,
    global_seen=None,
):
    """
    Generate a balanced dataset.

    global_seen prevents exact input duplication
    across train/validation/test.
    """

    examples = []

    if global_seen is None:
        global_seen = set()

    task_index = 0
    attempts = 0

    max_attempts = size * 30

    while (
        len(examples) < size
        and attempts < max_attempts
    ):

        # Cycle through task types to maintain
        # approximately balanced task distribution.
        task = TASK_TYPES[
            task_index % len(TASK_TYPES)
        ]

        generator = GENERATORS[task]

        example_id = (
            f"{prefix}_{len(examples) + 1:06d}"
        )

        example = generator(example_id)

        input_text = example["input"]

        if input_text not in global_seen:

            global_seen.add(input_text)
            examples.append(example)

        task_index += 1
        attempts += 1

    if len(examples) < size:

        raise RuntimeError(
            f"Could only generate "
            f"{len(examples)} / {size} "
            f"unique examples."
        )

    # Shuffle examples so task order is not predictable.
    random.shuffle(examples)

    # Re-number IDs after shuffling.
    for i, example in enumerate(
        examples,
        start=1,
    ):

        example["id"] = (
            f"{prefix}_{i:06d}"
        )

    return examples


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(
    data,
    expected_size,
):

    assert len(data) == expected_size

    ids = [
        x["id"]
        for x in data
    ]

    inputs = [
        x["input"]
        for x in data
    ]

    assert len(ids) == len(set(ids)), \
        "Duplicate IDs found!"

    assert len(inputs) == len(set(inputs)), \
        "Duplicate inputs found!"

    for example in data:

        assert "id" in example
        assert "task" in example
        assert "difficulty" in example
        assert "domain" in example
        assert "input" in example
        assert "answer" in example

        assert example["task"] in TASK_TYPES

        assert example["difficulty"] in [
            1,
            2,
            3,
        ]

        assert example["domain"] in DOMAIN_NAMES

        assert example["input"].strip()

        assert example["answer"].strip()

    print(
        f"✓ Validation passed: "
        f"{len(data)} examples"
    )


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(
    data,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for example in data:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"✓ Saved: {path}")


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(
    data,
    name,
):

    task_counts = {}
    difficulty_counts = {}
    domain_counts = {}

    for example in data:

        task = example["task"]
        difficulty = example["difficulty"]
        domain = example["domain"]

        task_counts[task] = (
            task_counts.get(task, 0) + 1
        )

        difficulty_counts[difficulty] = (
            difficulty_counts.get(
                difficulty,
                0,
            ) + 1
        )

        domain_counts[domain] = (
            domain_counts.get(
                domain,
                0,
            ) + 1
        )

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print("\nTask distribution:")

    for task in TASK_TYPES:

        print(
            f"  {task:22s}: "
            f"{task_counts.get(task, 0)}"
        )

    print("\nDifficulty distribution:")

    for difficulty in [1, 2, 3]:

        print(
            f"  Difficulty {difficulty}: "
            f"{difficulty_counts.get(difficulty, 0)}"
        )

    print("\nDomain distribution:")

    for domain in DOMAIN_NAMES:

        print(
            f"  {domain:22s}: "
            f"{domain_counts.get(domain, 0)}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ASSAMESE PHASE 3 DATASET GENERATOR")
    print("=" * 60)

    print(f"\nSeed: {SEED}")

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"\nAvailable names: "
        f"{len(NAMES)}"
    )

    print(
        f"Available domains: "
        f"{len(DOMAIN_NAMES)}"
    )

    print(
        "Domains: "
        + ", ".join(DOMAIN_NAMES)
    )

    print(
        f"\nDataset sizes:"
        f"\n  Train: {TRAIN_SIZE}"
        f"\n  Validation: {VAL_SIZE}"
        f"\n  Test: {TEST_SIZE}"
    )

    # One shared set guarantees
    # no exact input appears in multiple splits.
    global_seen = set()

    # --------------------------------------------------------
    # GENERATE TRAIN
    # --------------------------------------------------------

    print("\nGenerating training set...")

    train = generate_dataset(
        TRAIN_SIZE,
        "as_train",
        global_seen,
    )

    # --------------------------------------------------------
    # GENERATE VALIDATION
    # --------------------------------------------------------

    print("Generating validation set...")

    val = generate_dataset(
        VAL_SIZE,
        "as_val",
        global_seen,
    )

    # --------------------------------------------------------
    # GENERATE TEST
    # --------------------------------------------------------

    print("Generating test set...")

    test = generate_dataset(
        TEST_SIZE,
        "as_test",
        global_seen,
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    print("\nValidating datasets...")

    validate_dataset(
        train,
        TRAIN_SIZE,
    )

    validate_dataset(
        val,
        VAL_SIZE,
    )

    validate_dataset(
        test,
        TEST_SIZE,
    )

    # --------------------------------------------------------
    # CROSS-SPLIT CHECK
    # --------------------------------------------------------

    train_inputs = {
        x["input"]
        for x in train
    }

    val_inputs = {
        x["input"]
        for x in val
    }

    test_inputs = {
        x["input"]
        for x in test
    }

    assert not (
        train_inputs & val_inputs
    ), "Train/validation overlap found!"

    assert not (
        train_inputs & test_inputs
    ), "Train/test overlap found!"

    assert not (
        val_inputs & test_inputs
    ), "Validation/test overlap found!"

    print(
        "✓ No cross-split duplicate inputs"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_jsonl(
        train,
        OUTPUT_DIR / "train.jsonl",
    )

    save_jsonl(
        val,
        OUTPUT_DIR / "val.jsonl",
    )

    save_jsonl(
        test,
        OUTPUT_DIR / "test.jsonl",
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    print_statistics(
        train,
        "TRAIN",
    )

    print_statistics(
        val,
        "VALIDATION",
    )

    print_statistics(
        test,
        "TEST",
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)

    print(
        f"\nTotal examples: "
        f"{len(train) + len(val) + len(test)}"
    )

    print(
        "✓ Assamese Phase 3 dataset ready."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()