# conftest.py
import sys
import os
from unittest.mock import MagicMock

# Disable ChromaDB telemetry (can cause crashes on Windows)
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

def pytest_configure(config):
    """Mock problematic modules before test collection."""
    mock_torch = MagicMock()
    mock_transformers = MagicMock()

    mock_transformers.GPT2TokenizerFast = MagicMock()
    mock_transformers.PreTrainedTokenizerBase = MagicMock()

    sys.modules['torch'] = mock_torch
    sys.modules['torch.nn'] = MagicMock()
    sys.modules['torch.utils'] = MagicMock()
    sys.modules['torch.utils.data'] = MagicMock()
    sys.modules['transformers'] = mock_transformers
