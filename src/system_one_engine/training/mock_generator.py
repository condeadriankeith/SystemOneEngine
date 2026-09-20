"""Synthetic multi-task mock data generator for System One Decision Engine.

Generates structured multi-task samples (Choice, Boolean, Score) for unit testing,
benchmarking, and local pipeline verification without requiring large dataset downloads.
"""

import json
import random
from pathlib import Path
from typing import Any


CHOICE_TEMPLATES = [
    {
        "instruction": "Route the customer message to the correct support queue.",
        "context_templates": [
            ("I was double charged on invoice #{id} for the monthly subscription.", "billing"),
            ("Where is shipment #{id}? Tracking has not updated in 4 days.", "logistics"),
            ("Can you tell me if your software supports LDAP SSO authentication?", "technical"),
            ("Your representative was rude and I demand to speak with a supervisor immediately.", "escalations"),
        ],
        "candidates": ["billing", "logistics", "technical", "escalations"],
    },
    {
        "instruction": "Classify the user intent for the banking agent.",
        "context_templates": [
            ("Please send $250 from checking to Alice's account.", "transfer"),
            ("What is my current available balance on credit card #{id}?", "balance"),
            ("I lost my card in London and need it locked right now.", "freeze_card"),
            ("What are your branch opening hours on Saturdays?", "hours"),
        ],
        "candidates": ["transfer", "balance", "freeze_card", "hours"],
    },
]

BOOLEAN_TEMPLATES = [
    {
        "instruction": "Does this message contain urgent escalation language?",
        "context_templates": [
            ("EMERGENCY: The server room is flooding and machines are powering off!", True),
            ("Whenever you get a chance, could you review the attached invoice?", False),
            ("CRITICAL OUTAGE: All user transactions are failing with 500 error codes!", True),
            ("Just wanted to check if you received my previous email from Monday.", False),
        ],
    },
    {
        "instruction": "Does this text violate content moderation safety policies?",
        "context_templates": [
            ("Tell me how to synthesize toxic biological weapons at home.", True),
            ("What is the capital of France and what are some popular tourist spots?", False),
            ("Provide detailed blueprints for attacking electric power substations.", True),
            ("How do I bake sourdough bread with an active starter?", False),
        ],
    },
]

SCORE_TEMPLATES = [
    {
        "instruction": "Rate the customer frustration level on a 4-level scale (0=calm, 3=furious).",
        "criteria": [
            "Calm and polite inquiry.",
            "Slight confusion or mild delay inquiry.",
            "Clear annoyance with demanding tone.",
            "Extreme anger, profanity, or legal threats.",
        ],
        "context_templates": [
            ("Good morning, could you please provide a receipt for my order?", 0),
            ("It has been three days and I have not received my shipping confirmation yet.", 1),
            ("This is the third time I am contacting you. Fix this issue today!", 2),
            ("You thieves stole my money! I am suing your company and reporting you to regulators!", 3),
        ],
    },
    {
        "instruction": "Assess code review complexity from 0 (trivial) to 2 (complex).",
        "criteria": [
            "Trivial fix: typo or single comment adjustment.",
            "Moderate change: single function or localized bugfix.",
            "Complex architectural overhaul or database migration.",
        ],
        "context_templates": [
            ("Fix typo in docstring: 'occured' -> 'occurred'.", 0),
            ("Add retry with exponential backoff to fetch_user_profile HTTP call.", 1),
            ("Refactor entire authentication pipeline to support multi-tenant OAuth2 and database partitioning.", 2),
        ],
    },
]


def generate_mock_dataset(num_samples: int = 500, seed: int = 42) -> list[dict[str, Any]]:
    """Generate a deterministic synthetic multi-task dataset.

    Args:
        num_samples: Total number of samples to produce (default 500).
        seed: Random seed for reproducibility.

    Returns:
        list of multi-task sample dictionaries.
    """
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []

    for idx in range(num_samples):
        task_type = rng.choice(["choice", "boolean", "score"])

        if task_type == "choice":
            template = rng.choice(CHOICE_TEMPLATES)
            ctx_template, target = rng.choice(template["context_templates"])
            ctx = ctx_template.format(id=rng.randint(1000, 9999))
            samples.append({
                "sample_id": f"choice_{idx:04d}",
                "task_type": "choice",
                "instruction": template["instruction"],
                "context": ctx,
                "candidates": list(template["candidates"]),
                "target": target,
                "target_index": template["candidates"].index(target),
            })

        elif task_type == "boolean":
            template = rng.choice(BOOLEAN_TEMPLATES)
            ctx_template, target = rng.choice(template["context_templates"])
            samples.append({
                "sample_id": f"boolean_{idx:04d}",
                "task_type": "boolean",
                "instruction": template["instruction"],
                "context": ctx_template,
                "target": target,
                "target_int": 1 if target else 0,
            })

        else:  # score
            template = rng.choice(SCORE_TEMPLATES)
            ctx_template, target = rng.choice(template["context_templates"])
            samples.append({
                "sample_id": f"score_{idx:04d}",
                "task_type": "score",
                "instruction": template["instruction"],
                "context": ctx_template,
                "criteria": list(template["criteria"]),
                "num_levels": len(template["criteria"]),
                "target": target,
            })

    return samples


def save_mock_dataset(
    filepath: str | Path,
    num_samples: int = 500,
    seed: int = 42,
) -> Path:
    """Generate and save mock multi-task dataset to a JSONL file.

    Args:
        filepath: Destination file path.
        num_samples: Number of samples to generate.
        seed: Random seed.

    Returns:
        Path: Path to written file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = generate_mock_dataset(num_samples=num_samples, seed=seed)

    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return path
