from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from typing import Any

from agent.config import PROJECT_ROOT, Settings, get_settings
from agent.llm import ChatMessage, LLMError, OpenAICompatibleClient, extract_json_object
from tools.github_tools import (
    Repository,
    list_user_repositories,
    rank_repositories,
    search_github_repositories,
)


PLANNER_SYSTEM_PROMPT = """Ты управляешь агентом RepoScout.
Задача агента: помогать человеку выбрать GitHub-репозитории для изучения новой темы.

Ответь строго JSON без markdown.
Если нужно искать репозитории, верни:
{"action":"search_github","query":"поисковый запрос для GitHub","limit":8,"goal":"что именно хочет пользователь"}

Если пользователь спрашивает не про поиск репозиториев, верни:
{"action":"answer_directly","answer":"короткий ответ"}

Правила:
- query должен быть на английском, потому что GitHub Search так работает лучше.
- не добавляй в query sort, stars или URL.
- limit от 5 до 10.
"""


ANSWER_SYSTEM_PROMPT = """Ты RepoScout, спокойный инженерный помощник.
У тебя есть реальные данные из GitHub API и результат локального ранжирования.
Отвечай по-русски, без рекламного тона.
Объясни, какие проекты стоит смотреть первыми и почему.
Не скрывай ограничения: stars не равны качеству, GitHub Search может вернуть шум.
"""


AVAILABLE_SKILLS = {
    "beginner_mode": "объяснять проще и меньше перегружать терминами",
    "strict_sources": "отделять факты из GitHub API от интерпретации агента",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="RepoScout: агент для подбора GitHub-репозиториев")
    parser.add_argument("prompt", nargs="*", help="Запрос пользователя")
    parser.add_argument("--offline", action="store_true", help="Не вызывать LLM, использовать простой планировщик")
    parser.add_argument("--limit", type=int, default=8, help="Сколько репозиториев запросить у GitHub")
    args = parser.parse_args()

    settings = get_settings()
    client = None
    if settings.llm_api_key and not args.offline:
        client = OpenAICompatibleClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        )

    user_prompt = " ".join(args.prompt).strip()
    if not user_prompt:
        return run_interactive(client, settings, limit=args.limit)

    return run_request(user_prompt, client, settings, limit=args.limit)


def run_interactive(client: OpenAICompatibleClient | None, settings: Settings, *, limit: int) -> int:
    enabled_skills = set(AVAILABLE_SKILLS)
    print("RepoScout")
    print("Напиши запрос или команду. /help — список команд, /exit — выход.")
    print()

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Выход.")
            return 0

        if not raw:
            continue
        if raw.lower() in {"exit", "quit", "выход"}:
            print("Выход.")
            return 0
        if raw.startswith("/"):
            should_continue = handle_command(raw, enabled_skills, client)
            if not should_continue:
                return 0
            continue

        run_request(raw, client, settings, limit=limit, enabled_skills=enabled_skills)
        print()


def handle_command(
    command: str,
    enabled_skills: set[str],
    client: OpenAICompatibleClient | None = None,
) -> bool:
    parts = command.split()
    name = parts[0].lower()

    if name in {"/exit", "/quit"}:
        print("Выход.")
        return False
    if name == "/help":
        print_help()
        return True
    if name == "/tools":
        print_tools()
        return True
    if name == "/skills":
        print_skills(enabled_skills)
        return True
    if name == "/skill":
        update_skill(parts, enabled_skills)
        return True
    if name == "/llm":
        check_llm(command, client)
        return True

    print("Неизвестная команда. Напиши /help, чтобы увидеть список команд.")
    return True


def print_help() -> None:
    print(
        "\n".join(
            [
                "Команды:",
                "/help — показать список команд",
                "/tools — показать инструменты агента",
                "/skills — показать skill-режимы",
                "/skill on <name> — включить skill",
                "/skill off <name> — выключить skill",
                "/llm <текст> — проверить LLM без инструментов",
                "/exit — выйти",
                "",
                "Обычный текст без / считается запросом к агенту.",
            ]
        )
    )


def print_tools() -> None:
    print(
        "\n".join(
            [
                "Инструменты:",
                "search_github_repositories — ищет репозитории через GitHub API",
                "list_user_repositories — получает публичные репозитории пользователя GitHub",
                "rank_repositories — ранжирует найденные репозитории",
            ]
        )
    )


def print_skills(enabled_skills: set[str]) -> None:
    print("Skill-режимы:")
    for name, description in AVAILABLE_SKILLS.items():
        state = "включен" if name in enabled_skills else "выключен"
        print(f"- {name}: {state}; {description}")


def update_skill(parts: list[str], enabled_skills: set[str]) -> None:
    if len(parts) != 3 or parts[1] not in {"on", "off"}:
        print("Формат: /skill on <name> или /skill off <name>")
        return

    skill_name = parts[2]
    if skill_name not in AVAILABLE_SKILLS:
        print(f"Неизвестный skill: {skill_name}")
        print_skills(enabled_skills)
        return

    if parts[1] == "on":
        enabled_skills.add(skill_name)
        print(f"Skill включен: {skill_name}")
    else:
        enabled_skills.discard(skill_name)
        print(f"Skill выключен: {skill_name}")


def check_llm(command: str, client: OpenAICompatibleClient | None) -> None:
    prompt = command.removeprefix("/llm").strip()
    if not prompt:
        print("Формат: /llm <текст для модели>")
        return
    if client is None:
        print("LLM-клиент не настроен. Проверь .env и не запускай с --offline.")
        return

    try:
        answer = client.chat(
            [
                ChatMessage("system", "Отвечай кратко по-русски."),
                ChatMessage("user", prompt),
            ],
            temperature=0.2,
        )
    except LLMError as exc:
        print(f"LLM не ответила: {exc}")
        return

    print(answer)


def run_request(
    user_prompt: str,
    client: OpenAICompatibleClient | None,
    settings: Settings,
    *,
    limit: int,
    enabled_skills: set[str] | None = None,
) -> int:
    print("RepoScout")
    print(f"Запрос: {user_prompt}")
    print()

    skill_text = load_skill_text(enabled_skills or set())

    github_user = extract_github_user(user_prompt)
    if github_user:
        return run_github_user_request(user_prompt, github_user, client, settings, limit=limit, skill_text=skill_text)

    plan = build_plan(user_prompt, client, fallback_limit=limit, skill_text=skill_text)
    if plan.get("action") == "answer_directly":
        print(plan.get("answer", "Я лучше всего умею искать GitHub-репозитории по теме."))
        return 0

    query = str(plan.get("query") or fallback_query(user_prompt))
    limit = int(plan.get("limit") or limit)
    goal = str(plan.get("goal") or user_prompt)

    print(f"[инструмент] search_github_repositories(запрос={query!r}, лимит={limit})")
    try:
        repositories = search_github_repositories(
            query,
            limit=limit,
            token=settings.github_token,
            data_dir=settings.data_dir,
        )
    except RuntimeError as exc:
        print(f"Не получилось вызвать GitHub API: {exc}")
        print("Проверь интернет, лимит GitHub API или добавь GITHUB_TOKEN в .env.")
        return 1
    if not repositories:
        print("GitHub ничего не вернул. Попробуй переформулировать запрос.")
        return 1

    print(f"[инструмент] rank_repositories(репозитории={len(repositories)}, цель={goal!r})")
    ranked = rank_repositories(repositories, goal)
    print()

    answer = build_answer(user_prompt, query, ranked[:5], client, skill_text=skill_text)
    print(answer)
    return 0


def run_github_user_request(
    user_prompt: str,
    username: str,
    client: OpenAICompatibleClient | None,
    settings: Settings,
    *,
    limit: int,
    skill_text: str,
) -> int:
    print(f"[инструмент] list_user_repositories(username={username!r}, лимит={limit})")
    try:
        repositories = list_user_repositories(
            username,
            limit=limit,
            token=settings.github_token,
            data_dir=settings.data_dir,
        )
    except RuntimeError as exc:
        print(f"Не получилось вызвать GitHub API: {exc}")
        print("Проверь ссылку, интернет, лимит GitHub API или добавь GITHUB_TOKEN в .env.")
        return 1

    if not repositories:
        print("У пользователя не найдено публичных репозиториев.")
        return 1

    print(f"[инструмент] rank_repositories(репозитории={len(repositories)}, цель={user_prompt!r})")
    ranked = rank_repositories(repositories, user_prompt)
    print()

    answer = build_answer(
        user_prompt,
        f"user:{username}",
        ranked[:5],
        client,
        skill_text=skill_text,
    )
    print(answer)
    return 0


def load_skill_text(enabled_skills: set[str]) -> str:
    if not enabled_skills:
        return ""

    chunks: list[str] = []
    for skill_name in sorted(enabled_skills):
        path = PROJECT_ROOT / "skills" / f"{skill_name}.md"
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8").strip())
    if not chunks:
        return ""
    return "Активные skill-режимы:\n" + "\n\n".join(chunks)


def build_plan(
    user_prompt: str,
    client: OpenAICompatibleClient | None,
    *,
    fallback_limit: int,
    skill_text: str = "",
) -> dict[str, Any]:
    if client is None:
        return {
            "action": "search_github",
            "query": fallback_query(user_prompt),
            "limit": fallback_limit,
            "goal": user_prompt,
        }

    try:
        raw = client.chat(
            [
                ChatMessage("system", join_prompt(PLANNER_SYSTEM_PROMPT, skill_text)),
                ChatMessage("user", user_prompt),
            ],
            temperature=0.1,
        )
        plan = extract_json_object(raw)
        if plan.get("action") in {"search_github", "answer_directly"}:
            return plan
    except (LLMError, ValueError, json.JSONDecodeError) as exc:
        print(f"[предупреждение] LLM-планировщик не ответил, используется простой планировщик: {exc}", file=sys.stderr)

    return {
        "action": "search_github",
        "query": fallback_query(user_prompt),
        "limit": fallback_limit,
        "goal": user_prompt,
    }


def build_answer(
    user_prompt: str,
    query: str,
    repositories: list[Repository],
    client: OpenAICompatibleClient | None,
    *,
    skill_text: str = "",
) -> str:
    payload = {
        "user_prompt": user_prompt,
        "github_query": query,
        "repositories": [asdict(repo) for repo in repositories],
    }

    if client is not None:
        try:
            return client.chat(
                [
                    ChatMessage("system", join_prompt(ANSWER_SYSTEM_PROMPT, skill_text)),
                    ChatMessage("user", json.dumps(payload, ensure_ascii=False, indent=2)),
                ],
                temperature=0.25,
            )
        except LLMError as exc:
            print(f"[предупреждение] LLM не собрала ответ, используется шаблон: {exc}", file=sys.stderr)

    return render_template_answer(query, repositories)


def join_prompt(base_prompt: str, extra_text: str) -> str:
    if not extra_text:
        return base_prompt
    return f"{base_prompt.strip()}\n\n{extra_text}"


def render_template_answer(query: str, repositories: list[Repository]) -> str:
    lines = [
        f"Я поискал на GitHub по запросу: {query}",
        "",
        "Короткий список, с чего начать:",
    ]
    for index, repo in enumerate(repositories, start=1):
        notes = "; ".join(repo.score_notes or [])
        lines.extend(
            [
                f"{index}. {repo.full_name} — оценка {repo.score}",
                f"   {repo.html_url}",
                f"   Язык: {repo.language or 'не указан'}, звезд: {repo.stars}, форков: {repo.forks}",
                f"   Почему в списке: {notes}.",
            ]
        )
    lines.extend(
        [
            "",
            "Ограничение: это первичный отбор по публичным метрикам GitHub.",
            "Перед реальным выбором стоит открыть README и посмотреть, насколько проект живой и понятный.",
        ]
    )
    return "\n".join(lines)


def fallback_query(user_prompt: str) -> str:
    lowered = user_prompt.lower()
    replacements = {
        "искусственный интеллект": "artificial intelligence",
        "ии": "ai",
        "агент": "agent",
        "агенты": "agents",
        "нович": "beginner",
        "проект": "project",
        "репозитор": "repository",
        "машинное обучение": "machine learning",
        "математическое моделирование": "mathematical modeling",
        "компьютерное зрение": "computer vision",
    }
    query = lowered
    for source, target in replacements.items():
        query = query.replace(source, target)
    words = re.findall(r"[a-zA-Z0-9+#.-]+", query)
    stop_words = {
        "github",
        "github-project",
        "find",
        "search",
        "project",
        "projects",
        "repository",
        "repositories",
        "repo",
        "repos",
    }
    useful = [word for word in words if word not in stop_words]
    if not useful:
        return "ai agents beginner"
    return " ".join(useful[:8])


def extract_github_user(text: str) -> str | None:
    match = re.search(r"https?://github\.com/([A-Za-z0-9-]+)(?:[/?#]|$)", text)
    if not match:
        return None
    username = match.group(1)
    reserved = {"orgs", "topics", "trending", "marketplace", "features", "settings"}
    if username.lower() in reserved:
        return None
    return username


if __name__ == "__main__":
    raise SystemExit(main())
