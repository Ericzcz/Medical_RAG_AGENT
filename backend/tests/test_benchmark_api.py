import unittest
from unittest.mock import AsyncMock, patch

import benchmark_api


class MixedBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def test_build_mixed_cases_alternates_local_and_web(self):
        cases = benchmark_api.build_mixed_cases()

        self.assertEqual(len(cases), 10)
        self.assertEqual(
            [case.route for case in cases],
            ["local", "web"] * 5,
        )
        self.assertTrue(
            all(
                case.endpoint == "local_query"
                for case in cases
                if case.route == "local"
            )
        )
        self.assertTrue(
            all(
                case.endpoint == "agent_query"
                for case in cases
                if case.route == "web"
            )
        )

    async def test_mixed_round_uses_each_cases_endpoint_and_cache_scope(self):
        cases = benchmark_api.build_mixed_cases()[:2]
        client = object()

        async def fake_run_one_query(_client, *, endpoint, query):
            return benchmark_api.RequestResult(
                duration=0.1,
                mode="local" if endpoint == "local_query" else "agent",
                status_code=200,
                endpoint=endpoint,
            )

        with (
            patch.object(
                benchmark_api,
                "clear_cache",
                new=AsyncMock(),
            ) as mock_clear_cache,
            patch.object(
                benchmark_api,
                "run_one_query",
                side_effect=fake_run_one_query,
            ) as mock_run_query,
        ):
            results, wall = await benchmark_api.benchmark_mixed_round(
                client,
                cases=cases,
                mode="serial",
            )

        self.assertGreaterEqual(wall, 0)
        self.assertEqual([result.route for result in results], ["local", "web"])
        self.assertEqual(
            [call.kwargs["endpoint"] for call in mock_run_query.await_args_list],
            ["local_query", "agent_query"],
        )
        self.assertEqual(
            [call.kwargs["scope"] for call in mock_clear_cache.await_args_list],
            ["local", "agent"],
        )


if __name__ == "__main__":
    unittest.main()
