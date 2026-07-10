import unittest
from unittest.mock import AsyncMock, patch

import medical_rag_agent.tests.benchmark_api as benchmark_api


class MixedBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def test_medical_question_set_is_default_resume_benchmark_set(self):
        questions = benchmark_api.get_questions("medical")

        self.assertEqual(len(questions), 10)
        self.assertIn("What is Aarskog-Scott syndrome?", questions)
        self.assertIn("What is Addison disease?", questions)

    def test_summarize_results_calculates_resume_metrics(self):
        results = [
            benchmark_api.RequestResult(duration=1.0, mode="local", status_code=200),
            benchmark_api.RequestResult(duration=2.0, mode="local", status_code=200),
            benchmark_api.RequestResult(duration=3.0, mode="cached", status_code=200),
        ]

        stats = benchmark_api.summarize_results(results, wall=3.0)

        self.assertEqual(stats.count, 3)
        self.assertEqual(stats.avg, 2.0)
        self.assertEqual(stats.p50, 2.0)
        self.assertEqual(stats.max_duration, 3.0)
        self.assertEqual(stats.throughput, 1.0)
        self.assertEqual(stats.modes, {"local": 2, "cached": 1})

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

    async def test_uncached_round_clears_cache_before_measuring(self):
        client = object()
        queries = ["q1", "q2"]

        async def fake_run_one_query(_client, *, endpoint, query, session_id=None, user_id=None):
            return benchmark_api.RequestResult(
                duration=0.1,
                mode="local",
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
            results, wall = await benchmark_api.benchmark_round(
                client,
                endpoint="local_query",
                queries=queries,
                scope="local",
                model="gpt-5.5",
                mode="serial",
                cache_mode="uncached",
            )

        self.assertGreaterEqual(wall, 0)
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_clear_cache.await_count, 2)
        self.assertEqual(mock_run_query.await_count, 2)

    async def test_cached_round_warms_cache_before_measuring(self):
        client = object()
        queries = ["q1", "q2"]

        async def fake_run_one_query(_client, *, endpoint, query, session_id=None, user_id=None):
            return benchmark_api.RequestResult(
                duration=0.1,
                mode="cached",
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
            results, wall = await benchmark_api.benchmark_round(
                client,
                endpoint="local_query",
                queries=queries,
                scope="local",
                model="gpt-5.5",
                mode="serial",
                cache_mode="cached",
                session_id="resume_session",
                user_id="resume_user",
            )

        self.assertGreaterEqual(wall, 0)
        self.assertEqual(len(results), 2)
        mock_clear_cache.assert_not_awaited()
        self.assertEqual(mock_run_query.await_count, 4)
        self.assertTrue(
            all(
                call.kwargs["session_id"] == "resume_session"
                for call in mock_run_query.await_args_list
            )
        )
        self.assertTrue(
            all(
                call.kwargs["user_id"] == "resume_user"
                for call in mock_run_query.await_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
