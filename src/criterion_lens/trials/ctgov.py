"""Typed, bounded-time access to the ClinicalTrials.gov API v2."""

import re
from collections.abc import Callable
from typing import Any

import requests

_NCT_ID = re.compile(r"^NCT\d{8}$")
_API_URL = "https://clinicaltrials.gov/api/v2/studies/{nct_id}"
DEFAULT_TIMEOUT_SECONDS = 10


class ClinicalTrialsError(RuntimeError):
    """Base error for fetching or parsing a ClinicalTrials.gov study."""


class InvalidNctIdError(ValueError):
    """Raised when an NCT identifier is malformed."""


class TrialNotFoundError(ClinicalTrialsError):
    """Raised when the requested study does not exist."""


class ClinicalTrialsRequestError(ClinicalTrialsError):
    """Raised for network or unsuccessful HTTP responses."""


class ClinicalTrialsResponseError(ClinicalTrialsError):
    """Raised when a successful API response lacks eligibility text."""


def fetch_eligibility_criteria(
    nct_id: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    get: Callable[..., Any] = requests.get,
) -> str:
    """Fetch eligibility text for one NCT study, with explicit network/API errors."""
    if not isinstance(nct_id, str) or not _NCT_ID.fullmatch(nct_id):
        raise InvalidNctIdError("NCT ID must match NCT followed by exactly eight digits")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    try:
        response = get(_API_URL.format(nct_id=nct_id), timeout=timeout)
    except requests.RequestException as error:
        raise ClinicalTrialsRequestError("Could not reach ClinicalTrials.gov") from error

    if response.status_code == 404:
        raise TrialNotFoundError(f"ClinicalTrials.gov study {nct_id} was not found")
    if response.status_code < 200 or response.status_code >= 300:
        raise ClinicalTrialsRequestError(
            f"ClinicalTrials.gov returned HTTP {response.status_code}"
        )
    try:
        payload = response.json()
    except ValueError as error:
        raise ClinicalTrialsResponseError("ClinicalTrials.gov returned invalid JSON") from error
    try:
        text = payload["protocolSection"]["eligibilityModule"]["eligibilityCriteria"]
    except (KeyError, TypeError) as error:
        raise ClinicalTrialsResponseError(
            "Study response does not include eligibility criteria"
        ) from error
    if not isinstance(text, str) or not text.strip():
        raise ClinicalTrialsResponseError("Study has empty eligibility criteria")
    return text.strip()
