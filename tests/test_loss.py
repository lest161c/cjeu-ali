import pytest
from src.custom_loss import citation_loss

def test_citation_loss():
    predicted = "References: 31979D0509, 31979D0510"
    retrieved_ids = ["31979D0509", "31979D0511"]
    loss = citation_loss(predicted, retrieved_ids)

    assert loss == 2  # 1 missing, 1 extra
