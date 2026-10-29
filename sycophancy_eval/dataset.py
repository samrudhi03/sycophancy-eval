from pathlib import Path

from inspect_ai.dataset import Dataset, Sample, json_dataset

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "questions.jsonl"


def record_to_sample(record: dict) -> Sample:
    return Sample(
        input=record["input"],
        target=record["target"],
        metadata={
            "wrong_answer": record["wrong_answer"],
            "category": record["category"],
            "source": record["source"],
        },
    )


def load_questions(path: Path = DATA_PATH) -> Dataset:
    return json_dataset(str(path), record_to_sample)
