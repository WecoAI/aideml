"""Backend for GDM Gemini API"""

import time
import logging

import google.api_core.exceptions
import google.generativeai as genai
from funcy import notnone, once, select_values

from .utils import FunctionSpec, OutputType, backoff_create, opt_messages_to_list

logger = logging.getLogger("aide")

_client = None  # type: ignore

GDM_TIMEOUT_EXCEPTIONS = (
    google.api_core.exceptions.RetryError,
    google.api_core.exceptions.TooManyRequests,
    google.api_core.exceptions.ResourceExhausted,
)


@once
def _setup_gdm_client():
    # We manually define the client. This is normally defined automatically when calling
    # the API, but it isn't thread-safe, so we anticipate its creation here
    raise NotImplementedError("GDM is not implemented yet.")


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    raise NotImplementedError("GDM is not implemented yet.")
