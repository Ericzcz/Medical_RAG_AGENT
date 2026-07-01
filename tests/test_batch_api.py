from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import AsyncMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import Project.week4.app.main as main  # noqa: E402


def make_request(redis_client=None):
    app = SimpleNamespace(state=SimpleNamespace(redis=redis_client or object()))
    return SimpleNamespace(app=app)


class RootApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_root_returns_service_entrypoint(self):
        response = await main.root()

        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["service"], "Week4_Agent_API")
        self.assertEqual(response["docs"], "/docs")
        self.assertIn("/health", response["endpoints"])


class BatchApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_local_query_returns_cached_results_without_batch_lookup(self):
        request = make_request()
        req = main.BatchQueryRequest(queries=["q1", "q2"])

        with (
            patch.object(
                main,
                "get_cache",
                new=AsyncMock(side_effect=[{"answer": "a1"}, {"answer": "a2"}]),
            ) as mock_get_cache,
            patch.object(
                main,
                "search_local_knowledge_batch",
                new=AsyncMock(),
            ) as mock_batch_search,
        ):
            response = await main.batch_local_query(req, request)

        self.assertEqual(response.mode, "batch_local")
        self.assertEqual(len(response.items), 2)
        self.assertEqual(response.items[0].answer, "a1")
        self.assertEqual(response.items[0].mode, "cached")
        self.assertEqual(response.items[1].answer, "a2")
        self.assertEqual(response.items[1].mode, "cached")
        self.assertEqual(mock_get_cache.await_count, 2)
        mock_batch_search.assert_not_awaited()

    async def test_batch_local_query_handles_partial_miss_and_error(self):
        request = make_request()
        req = main.BatchQueryRequest(queries=["q1", "q2", "q3"])

        with (
            patch.object(
                main,
                "get_cache",
                new=AsyncMock(
                    side_effect=[
                        {"answer": "cached-a1"},
                        None,
                        None,
                    ]
                ),
            ),
            patch.object(
                main,
                "search_local_knowledge_batch",
                new=AsyncMock(
                    return_value=[
                        {"query": "q2", "answer": "fresh-a2", "error": None},
                        {"query": "q3", "answer": None, "error": "boom"},
                    ]
                ),
            ) as mock_batch_search,
            patch.object(main, "set_cache", new=AsyncMock()) as mock_set_cache,
        ):
            response = await main.batch_local_query(req, request)

        self.assertEqual(response.items[0].mode, "cached")
        self.assertEqual(response.items[0].answer, "cached-a1")

        self.assertEqual(response.items[1].mode, "local")
        self.assertEqual(response.items[1].answer, "fresh-a2")
        self.assertIsNone(response.items[1].error)

        self.assertEqual(response.items[2].mode, "error")
        self.assertIsNone(response.items[2].answer)
        self.assertEqual(response.items[2].error, "boom")

        mock_batch_search.assert_awaited_once_with(["q2", "q3"], main.LOCAL_MODEL)
        mock_set_cache.assert_awaited_once_with(
            request.app.state.redis,
            "q2",
            main.LOCAL_SCOPE,
            main.LOCAL_MODEL,
            {"answer": "fresh-a2"},
        )

    async def test_batch_agent_query_handles_partial_miss_and_error(self):
        request = make_request()
        req = main.BatchQueryRequest(queries=["q1", "q2"])

        with (
            patch.object(
                main,
                "get_cache",
                new=AsyncMock(side_effect=[None, {"answer": "cached-a2"}]),
            ),
            patch.object(
                main,
                "run_agent_batch",
                new=AsyncMock(
                    return_value=[
                        {"query": "q1", "answer": None, "error": "rate limit"},
                    ]
                ),
            ) as mock_batch_agent,
            patch.object(main, "set_cache", new=AsyncMock()) as mock_set_cache,
        ):
            response = await main.batch_agent_query(req, request)

        self.assertEqual(response.mode, "batch_agent")
        self.assertEqual(response.items[0].mode, "error")
        self.assertEqual(response.items[0].error, "rate limit")
        self.assertEqual(response.items[1].mode, "cached")
        self.assertEqual(response.items[1].answer, "cached-a2")

        mock_batch_agent.assert_awaited_once_with(
            queries=["q1"],
            model=main.AGENT_MODEL,
        )
        mock_set_cache.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
