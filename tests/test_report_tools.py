from tools.github_tools import Repository
from tools.report_tools import evaluate_readme_quality, render_markdown_repository_table


def test_evaluate_readme_quality_finds_required_sections() -> None:
    readme = """
Track: A+C

## Запуск
python3 main.py

## Tools
/tools показывает инструменты

## Data
Источник: GitHub API, data/generated

## Проверка
pytest

## Ограничения
Иногда ломается на rate limit
"""

    result = evaluate_readme_quality(readme)

    assert result.verdict in {"confirmed", "partial"}
    assert result.score >= 60
    assert "track" in result.found
    assert "tools" in result.found


def test_render_markdown_repository_table_contains_repositories() -> None:
    repo = Repository(
        name="demo",
        full_name="owner/demo",
        html_url="https://github.com/owner/demo",
        description="",
        language="Python",
        stars=10,
        forks=2,
        open_issues=1,
        updated_at="2026-04-24T00:00:00Z",
        pushed_at="2026-04-24T00:00:00Z",
        topics=[],
        score=42.0,
        score_notes=["пример"],
    )

    table = render_markdown_repository_table([repo])

    assert "Репозиторий" in table
    assert "owner/demo" in table
    assert "42.0" in table
