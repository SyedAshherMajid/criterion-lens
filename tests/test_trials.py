from types import SimpleNamespace

import pytest
import requests

from criterion_lens.trials.criteria_parser import CriterionInput, parse_criteria_text
from criterion_lens.trials.ctgov import (
    ClinicalTrialsRequestError,
    ClinicalTrialsResponseError,
    InvalidNctIdError,
    TrialNotFoundError,
    fetch_eligibility_criteria,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Inclusion Criteria:\n- Age >= 18 years\n- ECOG status 0-1\n"
            "Exclusion Criteria:\n1. Active malignancy",
            [
                CriterionInput("inclusion", "Age >= 18 years"),
                CriterionInput("inclusion", "ECOG status 0-1"),
                CriterionInput("exclusion", "Active malignancy"),
            ],
        ),
        (
            "Inclusion Criteria\n1) Histologically confirmed disease\n"
            "2) Adequate organ function\nExclusion Criteria\n* Prior therapy",
            [
                CriterionInput("inclusion", "Histologically confirmed disease"),
                CriterionInput("inclusion", "Adequate organ function"),
                CriterionInput("exclusion", "Prior therapy"),
            ],
        ),
        (
            "Inclusion Criteria:\nPatients with stage II or III disease\n"
            "and measurable lesions by RECIST.\nExclusion Criteria:\n"
            "Pregnant or breastfeeding participants.",
            [
                CriterionInput(
                    "inclusion",
                    "Patients with stage II or III disease and measurable lesions by RECIST.",
                ),
                CriterionInput("exclusion", "Pregnant or breastfeeding participants."),
            ],
        ),
    ],
)
def test_parse_real_style_criteria(text, expected) -> None:
    assert parse_criteria_text(text) == expected


def test_parse_criteria_requires_typed_heading_and_items() -> None:
    with pytest.raises(ValueError, match="must follow"):
        parse_criteria_text("Age >= 18 years")
    with pytest.raises(ValueError, match="No criteria items"):
        parse_criteria_text("Inclusion Criteria:\nExclusion Criteria:")


def test_fetch_eligibility_criteria_uses_api_v2_and_timeout() -> None:
    observed = {}

    def get(url, *, timeout):
        observed.update(url=url, timeout=timeout)
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "protocolSection": {
                    "eligibilityModule": {"eligibilityCriteria": "Inclusion Criteria:\nAge >= 18"}
                }
            },
        )

    assert fetch_eligibility_criteria("NCT01234567", get=get) == (
        "Inclusion Criteria:\nAge >= 18"
    )
    assert observed == {
        "url": "https://clinicaltrials.gov/api/v2/studies/NCT01234567",
        "timeout": 10,
    }


@pytest.mark.parametrize(
    ("nct_id", "error"),
    [
        ("123", InvalidNctIdError),
        ("NCT1234567", InvalidNctIdError),
        ("NCT12345678", TrialNotFoundError),
    ],
)
def test_fetch_eligibility_criteria_surfaces_input_and_not_found_errors(
    nct_id, error
) -> None:
    def get(*args, **kwargs):
        return SimpleNamespace(status_code=404, json=lambda: {})

    with pytest.raises(error):
        fetch_eligibility_criteria(nct_id, get=get)


def test_fetch_eligibility_criteria_surfaces_network_and_response_errors() -> None:
    def network_error(*args, **kwargs):
        raise requests.Timeout("private detail")

    with pytest.raises(ClinicalTrialsRequestError, match="Could not reach"):
        fetch_eligibility_criteria("NCT01234567", get=network_error)

    with pytest.raises(ClinicalTrialsResponseError, match="does not include"):
        fetch_eligibility_criteria(
            "NCT01234567",
            get=lambda *args, **kwargs: SimpleNamespace(status_code=200, json=lambda: {}),
        )
