from tools.github_tools import Repository, rank_repositories


def make_repo(
    full_name: str,
    *,
    stars: int,
    description: str,
    pushed_at: str = "2026-04-01T00:00:00Z",
) -> Repository:
    return Repository(
        name=full_name.split("/")[-1],
        full_name=full_name,
        html_url=f"https://github.com/{full_name}",
        description=description,
        language="Python",
        stars=stars,
        forks=10,
        open_issues=1,
        updated_at=pushed_at,
        pushed_at=pushed_at,
        topics=["ai-agents"],
    )


def test_rank_repositories_prefers_beginner_friendly_project() -> None:
    beginner_repo = make_repo(
        "example/ai-agents-course",
        stars=500,
        description="Beginner course with examples and tutorial for AI agents",
    )
    obscure_repo = make_repo(
        "example/internal-agent-core",
        stars=20,
        description="Internal experimental runtime",
    )

    ranked = rank_repositories([obscure_repo, beginner_repo], "AI agents для новичка")

    assert ranked[0].full_name == "example/ai-agents-course"
    assert ranked[0].score > ranked[1].score
