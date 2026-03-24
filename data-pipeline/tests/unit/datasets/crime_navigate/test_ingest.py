"""
Unit tests for CrimeNavigateIngester.

Focus:
- uses crime_navigate config (dataset name, watermark field, primary key)
- first run (no since) starts from FIRST_RUN_START (2023-01-01)
- API SQL for a month includes OCCURRED_ON_DATE bounds
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.datasets.crime_navigate.ingest import CrimeNavigateIngester


class TestCrimeNavigateIngester:
    @pytest.fixture
    def ingester(self) -> CrimeNavigateIngester:
        return CrimeNavigateIngester()

    def test_basic_metadata(self, ingester: CrimeNavigateIngester) -> None:
        assert ingester.get_dataset_name() == "crime_navigate"
        assert ingester.get_watermark_field() == "OCCURRED_ON_DATE"
        assert ingester.get_primary_key() == "INCIDENT_NUMBER"

    @patch("src.datasets.crime_navigate.ingest.requests.get")
    def test_fetch_data_first_run_starts_from_config(
        self,
        mock_get: MagicMock,
        ingester: CrimeNavigateIngester,
    ) -> None:
        """When since is None, use FIRST_RUN_START from config (2023-01-01)."""
        sample = {
            "success": True,
            "result": {
                "records": [
                    {
                        "INCIDENT_NUMBER": "I2023001",
                        "OCCURRED_ON_DATE": "2023-01-15T12:00:00",
                    }
                ]
            },
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample
        mock_get.return_value = mock_resp

        # Use a small until window in Jan 2023 so only Jan is hit
        until = datetime(2023, 1, 31, tzinfo=UTC)
        df = ingester.fetch_data(since=None, until=until)

        assert not df.empty
        # ensure SQL used OCCURRED_ON_DATE and a >= 2023-01-01 filter
        sql_param = mock_get.call_args.kwargs["params"]["sql"]
        assert "OCCURRED_ON_DATE" in sql_param
        assert "2023-01-01" in sql_param

    @patch("src.datasets.crime_navigate.ingest.requests.get")
    def test_fetch_data_with_watermark_uses_since_date(
        self,
        mock_get: MagicMock,
        ingester: CrimeNavigateIngester,
    ) -> None:
        """When since is provided, month queries are clipped to that date."""
        sample = {
            "success": True,
            "result": {
                "records": [
                    {
                        "INCIDENT_NUMBER": "I2024001",
                        "OCCURRED_ON_DATE": "2024-02-10T08:00:00",
                    }
                ]
            },
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample
        mock_get.return_value = mock_resp

        since = datetime(2024, 2, 5, tzinfo=UTC)
        until = datetime(2024, 2, 28, tzinfo=UTC)
        df = ingester.fetch_data(since=since, until=until)

        assert isinstance(df, pd.DataFrame)
        sql_param = mock_get.call_args.kwargs["params"]["sql"]
        # start bound should be >= 2024-02-05, end bound < 2024-02-29
        assert "2024-02-05" in sql_param
        assert "2024-02-29" in sql_param or "2024-02-28" in sql_param
