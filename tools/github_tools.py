from __future__ import annotations

import base64
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GITHUB_API = "https://api.github.com"


@dataclass
class Repository:
    name: str
    full_name: str
    html_url: str
    description: str
    language: str | None
    stars: int
    forks: int
    open_issues: int
    updated_at: str
    pushed_at: str
    topics: list[str]
    score: float = 0.0
    score_notes: list[str] | None = None


def _github_get(path: str, token: str | None = None) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "RepoScout-TestTask",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(f"{GITHUB_API}{path}", headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"запрос к GitHub API не удался: {exc.reason}") from exc


def search_github_repositories(
    query: str,
    *,
    limit: int = 10,
    token: str | None = None,
    data_dir: Path | None = None,
) -> list[Repository]:
    """Инструмент 1: ищет реальные репозитории GitHub и сохраняет данные"""
    clean_query = query.strip()
    if "archived:false" not in clean_query:
        clean_query = f"{clean_query} archived:false"

    params = urllib.parse.urlencode(
        {
            "q": clean_query,
            "sort": "stars",
            "order": "desc",
            "per_page": min(max(limit, 1), 20),
        }
    )
    data = _github_get(f"/search/repositories?{params}", token=token)

    repos: list[Repository] = []
    for item in data.get("items", []):
        repos.append(
            Repository(
                name=item.get("name", ""),
                full_name=item.get("full_name", ""),
                html_url=item.get("html_url", ""),
                description=item.get("description") or "",
                language=item.get("language"),
                stars=int(item.get("stargazers_count") or 0),
                forks=int(item.get("forks_count") or 0),
                open_issues=int(item.get("open_issues_count") or 0),
                updated_at=item.get("updated_at", ""),
                pushed_at=item.get("pushed_at", ""),
                topics=list(item.get("topics") or []),
            )
        )

    if data_dir:
        data_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_query = re.sub(r"[^a-zA-Z0-9а-яА-Я_-]+", "_", query.strip())[:60].strip("_")
        output_path = data_dir / f"github_{safe_query or 'query'}_{stamp}.json"
        payload = {
            "source": "GitHub Search API",
            "source_url": f"{GITHUB_API}/search/repositories",
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "query": clean_query,
            "repositories": [asdict(repo) for repo in repos],
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return repos


def list_user_repositories(
    username: str,
    *,
    limit: int = 10,
    token: str | None = None,
    data_dir: Path | None = None,
) -> list[Repository]:
    """Инструмент: получает публичные репозитории пользователя GitHub"""
    clean_username = username.strip()
    params = urllib.parse.urlencode(
        {
            "sort": "updated",
            "direction": "desc",
            "per_page": min(max(limit, 1), 30),
        }
    )
    data = _github_get(f"/users/{urllib.parse.quote(clean_username)}/repos?{params}", token=token)

    repos: list[Repository] = []
    for item in data if isinstance(data, list) else []:
        repos.append(
            Repository(
                name=item.get("name", ""),
                full_name=item.get("full_name", ""),
                html_url=item.get("html_url", ""),
                description=item.get("description") or "",
                language=item.get("language"),
                stars=int(item.get("stargazers_count") or 0),
                forks=int(item.get("forks_count") or 0),
                open_issues=int(item.get("open_issues_count") or 0),
                updated_at=item.get("updated_at", ""),
                pushed_at=item.get("pushed_at", ""),
                topics=list(item.get("topics") or []),
            )
        )

    if data_dir:
        data_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = data_dir / f"github_user_{clean_username}_{stamp}.json"
        payload = {
            "source": "GitHub Users Repositories API",
            "source_url": f"{GITHUB_API}/users/{clean_username}/repos",
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "username": clean_username,
            "repositories": [asdict(repo) for repo in repos],
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return repos


def fetch_repository_readme(full_name: str, *, token: str | None = None) -> str:
    """Дополнительный инструмент: получает README выбранного репозитория"""
    encoded = urllib.parse.quote(full_name, safe="/")
    data = _github_get(f"/repos/{encoded}/readme", token=token)
    content = data.get("content") or ""
    if not content:
        return ""
    try:
        return base64.b64decode(content).decode("utf-8", errors="replace")
    except ValueError:
        return ""


def rank_repositories(repositories: list[Repository], user_goal: str) -> list[Repository]:
    """Инструмент 2: прозрачно ранжирует репозитории для новичка"""
    now = datetime.now(timezone.utc)
    goal_lower = user_goal.lower()
    beginner_words = {
        "tutorial",
        "example",
        "examples",
        "awesome",
        "starter",
        "learn",
        "course",
        "guide",
        "template",
        "demo",
        "workshop",
    }

    ranked: list[Repository] = []
    for repo in repositories:
        notes: list[str] = []
        stars_score = min(math.log10(repo.stars + 1) / 5, 1.0) * 40
        if repo.stars >= 1000:
            notes.append("много звезд, проект заметен в сообществе")
        elif repo.stars >= 100:
            notes.append("есть признаки живого интереса")

        pushed = _parse_github_datetime(repo.pushed_at or repo.updated_at)
        if pushed:
            days_old = max((now - pushed).days, 0)
            recency_score = max(0.0, 25 - min(days_old / 30, 25))
            if days_old <= 180:
                notes.append("обновлялся в последние полгода")
        else:
            recency_score = 0.0

        text = " ".join([repo.name, repo.description, " ".join(repo.topics)]).lower()
        beginner_hits = sorted(word for word in beginner_words if word in text)
        beginner_score = min(len(beginner_hits) * 7, 20)
        if beginner_hits:
            notes.append("похож на учебный/обзорный проект")

        goal_tokens = [token for token in re.findall(r"[a-zA-Zа-яА-Я0-9]+", goal_lower) if len(token) > 2]
        relevance_hits = [token for token in goal_tokens if token in text]
        relevance_score = min(len(relevance_hits) * 5, 15)
        if relevance_hits:
            notes.append("совпадает с формулировкой запроса")

        issue_penalty = min(repo.open_issues / max(repo.stars, 1), 1.0) * 8
        score = round(stars_score + recency_score + beginner_score + relevance_score - issue_penalty, 2)
        ranked.append(
            Repository(
                **{**asdict(repo), "score": score, "score_notes": notes[:3] or ["подходит по базовым метрикам"]}
            )
        )

    return sorted(ranked, key=lambda repo: repo.score, reverse=True)


def _parse_github_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
