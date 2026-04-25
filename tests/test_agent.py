from agent.core import extract_github_repo, extract_github_user, fallback_query, looks_like_github_task


def test_fallback_query_translates_common_russian_words() -> None:
    query = fallback_query("Найди проекты про ИИ агентов для новичка")

    assert "ai" in query
    assert "agent" in query or "agents" in query
    assert "beginner" in query


def test_extract_github_user_from_profile_url() -> None:
    username = extract_github_user("проанализируй https://github.com/L3iHS?tab=repositories")

    assert username == "L3iHS"


def test_extract_github_repo_from_repo_url() -> None:
    full_name = extract_github_repo("оцени https://github.com/L3iHS/mat_mod_test_ai_agent")

    assert full_name == "L3iHS/mat_mod_test_ai_agent"


def test_fallback_query_ignores_unrelated_question() -> None:
    query = fallback_query("какой пик курса биткойна был")

    assert query == ""


def test_looks_like_github_task_for_repo_question() -> None:
    assert looks_like_github_task("найди проекты по computer vision")
