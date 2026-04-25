from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from tools.github_tools import Repository


@dataclass
class ReadmeEvaluation:
    verdict: str
    score: int
    found: list[str]
    missing: list[str]
    evidence: list[str]


README_CHECKS = {
    "track": ["track:"],
    "installation": ["pip install", "requirements.txt", "python3 -m pip", "pip3 install"],
    "run_command": ["python3 main.py", "python main.py", "запуск", "quick start"],
    "tools": ["tool", "инструмент", "tools", "/tools"],
    "data_sources": ["github api", "data/", "источник", "source"],
    "tests": ["pytest", "тест", "tests/"],
    "reflection": ["reflection.md", "рефлексия"],
    "limitations": ["огранич", "failure", "ломается", "risk"],
}


def evaluate_readme_quality(readme_text: str) -> ReadmeEvaluation:
    """Инструмент: оценивает полноту README по чеклисту тестового"""
    lowered = readme_text.lower()
    found: list[str] = []
    missing: list[str] = []
    evidence: list[str] = []

    for check_name, markers in README_CHECKS.items():
        matched_marker = next((marker for marker in markers if marker in lowered), None)
        if matched_marker:
            found.append(check_name)
            line = find_line_with_marker(readme_text, matched_marker)
            if line:
                evidence.append(f"{check_name}: {line}")
        else:
            missing.append(check_name)

    score = round(len(found) / len(README_CHECKS) * 100)
    if score >= 75:
        verdict = "confirmed"
    elif score >= 45:
        verdict = "partial"
    else:
        verdict = "missing"

    return ReadmeEvaluation(
        verdict=verdict,
        score=score,
        found=found,
        missing=missing,
        evidence=evidence[:6],
    )


def render_markdown_repository_table(repositories: list[Repository]) -> str:
    """Инструмент: строит Markdown-таблицу сравнения репозиториев для отчета"""
    headers = ["#", "Репозиторий", "Язык", "Звезды", "Issues", "Оценка"]
    align_right = [True, False, False, True, True, True]
    rows = [
        [
            str(index),
            f"[{repo.full_name}]({repo.html_url})",
            repo.language or "н/д",
            str(repo.stars),
            str(repo.open_issues),
            f"{repo.score:.2f}",
        ]
        for index, repo in enumerate(repositories[:8], start=1)
    ]

    widths = [
        max(len(headers[column]), 4 if align_right[column] else 3, *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]

    def format_row(values: list[str]) -> str:
        cells = []
        for value, width, right in zip(values, widths, align_right):
            cells.append(value.rjust(width) if right else value.ljust(width))
        return "| " + " | ".join(cells) + " |"

    separator = [
        ("-" * (width - 1) + ":") if right else "-" * width
        for width, right in zip(widths, align_right)
    ]

    return "\n".join([format_row(headers), format_row(separator), *(format_row(row) for row in rows)])


def print_repository_table(repositories: list[Repository]) -> None:
    """Инструмент: выводит таблицу сравнения репозиториев в терминал через rich"""
    table = Table(title="Сравнение репозиториев", show_lines=False)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Репозиторий", style="bold")
    table.add_column("Язык", no_wrap=True)
    table.add_column("Звезды", justify="right")
    table.add_column("Issues", justify="right")
    table.add_column("Оценка", justify="right", style="green")

    for index, repo in enumerate(repositories[:8], start=1):
        table.add_row(
            str(index),
            repo.full_name,
            repo.language or "н/д",
            str(repo.stars),
            str(repo.open_issues),
            str(repo.score),
        )

    Console().print(table)


def write_markdown_report(
    *,
    title: str,
    query: str,
    repositories: list[Repository],
    reports_dir: Path,
    readme_evaluation: ReadmeEvaluation | None = None,
) -> Path:
    """Инструмент: сохраняет Markdown-отчет по результатам анализа"""
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r"[^a-zA-Z0-9а-яА-Я_-]+", "_", title.lower())[:50].strip("_")
    path = reports_dir / f"{safe_title or 'report'}_{stamp}.md"

    lines = [
        f"# {title}",
        "",
        f"- Источник: GitHub API",
        f"- Дата сбора: {datetime.now(timezone.utc).isoformat()}",
        f"- Запрос: `{query}`",
        "",
        "## Сравнение",
        "",
        render_markdown_repository_table(repositories),
    ]

    if readme_evaluation:
        lines.extend(
            [
                "",
                "## Проверка README",
                "",
                f"- Вердикт: `{readme_evaluation.verdict}`",
                f"- Оценка: `{readme_evaluation.score}/100`",
                f"- Найдено: {', '.join(readme_evaluation.found) or 'ничего'}",
                f"- Не хватает: {', '.join(readme_evaluation.missing) or 'ничего'}",
                "",
                "### Evidence",
            ]
        )
        lines.extend(f"- {item}" for item in readme_evaluation.evidence)

    lines.extend(
        [
            "",
            "## JSON",
            "",
            "```json",
            json.dumps([asdict(repo) for repo in repositories], ensure_ascii=False, indent=2),
            "```",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def find_line_with_marker(text: str, marker: str) -> str:
    marker_lower = marker.lower()
    for index, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if marker_lower in line.lower():
            return f"README.md:{index} -> {line[:140]}"
    return ""
