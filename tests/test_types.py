from src.common.types import Sample


def test_sample_minimal_fields():
    s = Sample(
        sample_id="hp_1",
        question="Who directed Inception?",
        gold_passages=["Christopher Nolan directed Inception in 2010."],
        distractor_passages=["Stanley Kubrick directed many films."],
        gold_answer="Christopher Nolan",
        gold_evidence=["Christopher Nolan directed Inception in 2010."],
        retrieved_passages=["Christopher Nolan directed Inception in 2010."],
        is_answerable=True,
    )
    assert s.sample_id == "hp_1"
    assert s.is_answerable is True
    assert len(s.retrieved_passages) == 1
