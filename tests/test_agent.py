from agent.core import extract_github_user, fallback_query


def test_fallback_query_translates_common_russian_words() -> None:
    query = fallback_query("Найди проекты про ИИ агентов для новичка")

    assert "ai" in query
    assert "agent" in query or "agents" in query
    assert "beginner" in query


def test_extract_github_user_from_profile_url() -> None:
    username = extract_github_user("проанализируй https://github.com/L3iHS?tab=repositories")

    assert username == "L3iHS"
