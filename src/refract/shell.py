"""`refract shell`: an interactive console for the agent on Windows, macOS and Linux.

Pick a subject, look at its news feed, explain one item, ask follow-up questions, read the trace, get a brief. The
console is built on the standard library only (`cmd`): line editing and history come from `readline` where Python
has it (macOS, Linux) and from the Windows console otherwise. Colours are ANSI and switched on only for a terminal
that shows them; `NO_COLOR` or `--no-color` turns them off.

    refract shell                       # with the model from refract.toml / .env
    refract shell --demo                # no model: recorded answers on the demo data
    refract shell --script session.txt  # commands from a file, one per line (CI, demos)
"""

from __future__ import annotations

import cmd
import os
import re
import shlex
import sys
from pathlib import Path
from typing import IO, Any

from .demo import DemoError
from .facade import Refract
from .i18n import norm_lang
from .render import brief_markdown, chat_markdown, feed_markdown, insight_markdown, trace_markdown
from .schemas import ChatTurn, ReaderRole
from .textutil import DataError

ROLES = [r.value for r in ReaderRole]

UI = {
    "en": {
        "intro": "Refract — news explained for one subject. `help` lists the commands, `exit` leaves.",
        "demo": "Demo mode: model answers are replayed from recordings of real runs; checks and policy are live code.",
        "no_model": "No model configured (LLM_API_KEY / refract.toml): `analyze` and `ask` will fail; try `refract shell --demo`.",
        "pick": "Choose a subject: `use <number or id>`.",
        "no_subject": "No subject yet: `profiles`, then `use <number>`.",
        "no_insight": "No insight yet: `analyze <number or event id>` first.",
        "unknown_profile": "No subject {x!r}; `profiles` lists them.",
        "unknown_event": "No news item {x!r}; `news` or `feed` lists them.",
        "unknown_cmd": "Unknown command {x!r}; `help` lists the commands.",
        "subject": "Subject: {name} ({id}, {kind}); reader role {role}, language {lang}.",
        "role_set": "Reader role: {x}.",
        "lang_set": "Language: {x}.",
        "bad_role": "Roles: {x}.",
        "bad_lang": "Languages: ru, en.",
        "working": "Working on it — Ctrl+C stops.",
        "stopped": "Stopped.",
        "failed": "No insight: {x}",
        "saved": "Saved: {x}",
        "loaded": "Loaded: {x}",
        "no_file": "No such file: {x}",
        "paste_title": "Title (one line):",
        "paste_text": "Text — finish with an empty line:",
        "paste_short": "Give a title and at least a sentence of text.",
        "replayed": "Replayed from the recording {x} (role {role}, language {lang}).",
        "bye": "Bye.",
        "error": "Error: {x}",
        "status": "Data: {data}. Model: {model}. Subject: {subject}. Insight: {insight}.",
        "none": "none",
        "bad_number": "Give a number from the last list or an id.",
        "follow_ups": "Ask about it: `ask <question>`, or just type the question.",
        "demo_marks": "▶ demo — news with a recording for this subject; others need a model.",
    },
    "ru": {
        "intro": "Refract — новости, объяснённые для одного субъекта. `help` — список команд, `exit` — выход.",
        "demo": "Демо-режим: ответы модели воспроизводятся из записей настоящих прогонов; проверки и политика — живой код.",
        "no_model": "Модель не настроена (LLM_API_KEY / refract.toml): `analyze` и `ask` не сработают; попробуйте `refract shell --demo`.",
        "pick": "Выберите субъекта: `use <номер или id>`.",
        "no_subject": "Субъект не выбран: `profiles`, затем `use <номер>`.",
        "no_insight": "Инсайта ещё нет: сначала `analyze <номер или id новости>`.",
        "unknown_profile": "Нет субъекта {x!r}; список — `profiles`.",
        "unknown_event": "Нет новости {x!r}; список — `news` или `feed`.",
        "unknown_cmd": "Нет команды {x!r}; список — `help`.",
        "subject": "Субъект: {name} ({id}, {kind}); роль читателя {role}, язык {lang}.",
        "role_set": "Роль читателя: {x}.",
        "lang_set": "Язык: {x}.",
        "bad_role": "Роли: {x}.",
        "bad_lang": "Языки: ru, en.",
        "working": "Работаю — Ctrl+C остановит.",
        "stopped": "Остановлено.",
        "failed": "Инсайта нет: {x}",
        "saved": "Сохранено: {x}",
        "loaded": "Загружено: {x}",
        "no_file": "Нет файла: {x}",
        "paste_title": "Заголовок (одна строка):",
        "paste_text": "Текст — закончите пустой строкой:",
        "paste_short": "Нужны заголовок и хотя бы одно предложение текста.",
        "replayed": "Воспроизведено из записи {x} (роль {role}, язык {lang}).",
        "bye": "До встречи.",
        "error": "Ошибка: {x}",
        "status": "Данные: {data}. Модель: {model}. Субъект: {subject}. Инсайт: {insight}.",
        "none": "нет",
        "bad_number": "Укажите номер из последнего списка или id.",
        "follow_ups": "Спросите об этом: `ask <вопрос>` или просто напишите вопрос.",
        "demo_marks": "▶ demo — новости с записью для этого субъекта; для остальных нужна модель.",
    },
}

HELP = {
    "en": """Subjects and news
  profiles                 the subjects in the data folder, numbered
  use <n|id>               choose the subject
  profile                  the chosen subject
  news                     every stored news item, numbered
  feed [n]                 news ranked for the subject, strongest links first (default 10)
Analysis
  analyze <n|id>           explain a news item for the subject, with live steps
  paste                    explain a news text you paste (title, then text, then an empty line)
  brief [n]                the n strongest news explained in one short brief (default 3)
  ask <question>           a follow-up question about the last insight; plain text works too
  sources                  every statement of the last insight with its origin and sources
  trace                    what the agent did: steps, models, tokens, cost
  save [folder]            save the last run (default runs/); `refract chat --run` reads it
  load <file>              load a saved run, to read it or ask about it
Settings
  role <role>              reader role: owner, ceo, cfo, chief_accountant, treasurer, entrepreneur, individual, other
  lang <ru|en>             language of the insight and of this console
  status                   data folder, model, subject, last insight
  clear                    clear the screen
  exit                     leave (also quit, Ctrl+D; Ctrl+Z then Enter on Windows)
Numbers refer to the last list shown (profiles, news, feed).""",
    "ru": """Субъекты и новости
  profiles                 субъекты из папки данных, с номерами
  use <n|id>               выбрать субъекта
  profile                  выбранный субъект
  news                     все новости, с номерами
  feed [n]                 новости для субъекта, самые сильные связи сверху (по умолчанию 10)
Анализ
  analyze <n|id>           объяснить новость для субъекта, шаги видны по ходу
  paste                    объяснить вставленный текст (заголовок, текст, пустая строка)
  brief [n]                n самых важных новостей в одной короткой сводке (по умолчанию 3)
  ask <вопрос>             уточняющий вопрос о последнем инсайте; можно просто написать текст
  sources                  каждое утверждение последнего инсайта с происхождением и источниками
  trace                    что сделал агент: шаги, модели, токены, стоимость
  save [папка]             сохранить последний прогон (по умолчанию runs/); его читает `refract chat --run`
  load <файл>              загрузить сохранённый прогон, чтобы прочитать или задать вопрос
Настройки
  role <роль>              роль читателя: owner, ceo, cfo, chief_accountant, treasurer, entrepreneur, individual, other
  lang <ru|en>             язык инсайта и консоли
  status                   папка данных, модель, субъект, последний инсайт
  clear                    очистить экран
  exit                     выход (также quit, Ctrl+D; в Windows Ctrl+Z и Enter)
Номера относятся к последнему показанному списку (profiles, news, feed).""",
}


# ---------------------------------------------------------------------------------------------------- colours


def supports_color(stream: IO[str]) -> bool:
    """ANSI colours for a terminal that shows them. On Windows 10+ the console needs VT processing switched on."""
    if os.getenv("NO_COLOR") or os.getenv("TERM") == "dumb" or not getattr(stream, "isatty", lambda: False)():
        return False
    if os.name != "nt":
        return True
    try:  # ENABLE_VIRTUAL_TERMINAL_PROCESSING on the console of this stream
        import ctypes

        kernel32 = ctypes.windll.kernel32  # pyright: ignore[reportAttributeAccessIssue]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except (AttributeError, OSError):  # pragma: no cover — not a real console
        return False


_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "blue": "34", "cyan": "36"}


class Style:
    def __init__(self, on: bool) -> None:
        self.on = on

    def __call__(self, text: str, *styles: str) -> str:
        if not self.on or not styles:
            return text
        return f"\033[{';'.join(_CODES[s] for s in styles)}m{text}\033[0m"

    def markdown(self, text: str) -> str:
        """Headings bold and coloured, **bold** bold, `code` cyan; the rest as is."""
        if not self.on:
            return text
        out = []
        for line in text.splitlines():
            if line.startswith("#"):
                out.append(self(line.lstrip("# "), "bold", "cyan"))
                continue
            line = re.sub(r"\*\*(.+?)\*\*", lambda m: self(m.group(1), "bold"), line)
            line = re.sub(r"`([^`]+)`", lambda m: self(m.group(1), "cyan"), line)
            out.append(line)
        return "\n".join(out)


# ---------------------------------------------------------------------------------------------------- the console


class Shell(cmd.Cmd):
    def __init__(
        self,
        rf: Refract,
        *,
        stdin: IO[str] | None = None,
        stdout: IO[str] | None = None,
        color: bool | None = None,
        interactive: bool = True,
    ) -> None:
        super().__init__(stdin=stdin, stdout=stdout)
        self.rf = rf
        self.out: IO[str] = stdout or sys.stdout
        self.style = Style(supports_color(self.out) if color is None else color)
        self.interactive = interactive
        self.use_rawinput = interactive and stdin is None
        self.lang = norm_lang(rf.settings.language)
        self.profile_id: str | None = None
        self.role: str | None = None  # None: the default role for the subject's kind
        self.last: list[str] = []  # ids of the last numbered list
        self.last_kind = ""  # "profiles" | "events"
        self.run: Any = None  # the last AgentRun
        self.history: list[ChatTurn] = []
        self.failed = False  # a command failed: the exit code of a script
        self.greeted = False
        self._update_prompt()

    # ------------------------------------------------------------------ plumbing

    def ui(self, key: str, **kw: Any) -> str:
        return UI[self.lang][key].format(**kw)

    def say(self, text: str = "", *styles: str) -> None:
        self.out.write(self.style(text, *styles) + "\n")
        self.out.flush()

    def warn(self, text: str) -> None:
        self.failed = True
        self.say(text, "yellow")

    def _update_prompt(self) -> None:
        subject = self.profile_id or "—"
        if not self.interactive:
            self.prompt = ""
            return
        prompt = self.style(f"refract[{subject}]", "bold", "blue") + "> "
        if self.style.on and _has_readline():  # readline must not count colour codes as width
            prompt = re.sub(r"(\033\[[0-9;]*m)", "\001\\1\002", prompt)
        self.prompt = prompt

    def preloop(self) -> None:
        if not self.interactive or self.greeted:
            return
        self.greeted = True
        self.say(self.ui("intro"), "bold")
        if self.rf.is_demo:
            self.say(self.ui("demo"), "dim")
        elif not self.rf.settings.api_key:
            self.say(self.ui("no_model"), "yellow")
        if self.profile_id:
            self.do_profile("")
        else:
            self.say(self.ui("pick"), "dim")

    def precmd(self, line: str) -> str:
        if not self.interactive and line.strip() not in ("", "EOF") and not line.lstrip().startswith("#"):
            self.say(self.style(f"> {line.strip()}", "dim"))
        return line

    def onecmd(self, line: str) -> bool:
        if line.strip().startswith("#"):  # a comment in a script
            return False
        try:
            return super().onecmd(line)
        except KeyboardInterrupt:
            self.warn(self.ui("stopped"))
        except (DemoError, DataError) as e:  # their messages are written for the reader
            self.warn(str(e))
        except Exception as e:  # noqa: BLE001 — one failed command must not end the session
            self.warn(self.ui("error", x=f"{type(e).__name__}: {e}"))
        return False

    def emptyline(self) -> bool:  # an empty line does not repeat the last command
        return False

    def default(self, line: str) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride] — True ends the loop
        if line in ("EOF", "q"):
            return self.do_exit("")
        word = line.split()[0]
        if (
            self.run is not None
            and self.run.insight is not None
            and (len(line.split()) > 2 or line.endswith("?"))
        ):
            self.do_ask(line)  # plain text after an insight is a question about it
            return False
        self.warn(self.ui("unknown_cmd", x=word))
        return False

    def do_help(self, arg: str) -> None:
        for line in HELP[self.lang].splitlines():
            self.say(line if line.startswith(" ") else self.style(line, "bold"))

    def completenames(self, text: str, *ignored: Any) -> list[str]:
        return [n for n in self.get_names_list() if n.startswith(text)]

    def get_names_list(self) -> list[str]:
        return sorted({n[3:] for n in self.get_names() if n.startswith("do_") and n != "do_EOF"})

    def _pick(self, arg: str, kind: str) -> str | None:
        """A number from the last list of this kind, or an id as is."""
        arg = arg.strip()
        if arg.isdigit() and kind == "profiles" and self.last_kind != kind:  # subjects keep their numbers
            self.last, self.last_kind = list(self.rf.store.profiles), kind
        if arg.isdigit():
            if self.last_kind != kind or not 1 <= int(arg) <= len(self.last):
                self.warn(self.ui("bad_number"))
                return None
            return self.last[int(arg) - 1]
        return arg or None

    def _need_subject(self) -> bool:
        if self.profile_id is None:
            self.warn(self.ui("no_subject"))
            return False
        return True

    def _need_insight(self) -> bool:
        if self.run is None or self.run.insight is None:
            self.warn(self.ui("no_insight"))
            return False
        return True

    def _role(self) -> str:
        if self.role:
            return self.role
        from .eval.harness import DEFAULT_ROLE

        profile = self.rf.store.profiles[self.profile_id or ""]
        return DEFAULT_ROLE.get(profile.kind.value, "other")

    def _replayable(self, event_id: str) -> str:
        """In demo mode, a mark on the news that has a recording for the chosen subject."""
        if self.rf.replay is None:
            return ""
        pairs = {(p, e) for p, e, *_ in self.rf.replay.pairs()}
        return self.style(" ▶ demo", "green") if (self.profile_id, event_id) in pairs else ""

    def _progress(self, ev: dict) -> None:
        if ev.get("status") == "started" and ev.get("label"):
            self.say(f"  … {ev['label']}", "dim")

    # ------------------------------------------------------------------ subjects and news

    def do_profiles(self, arg: str) -> None:
        self.last, self.last_kind = list(self.rf.store.profiles), "profiles"
        for i, p in enumerate(self.rf.store.profiles.values(), 1):
            mark = "*" if p.profile_id == self.profile_id else " "
            self.say(f"{mark}{i:3}  {self.style(p.profile_id, 'bold'):28}  {p.kind.value:16} {p.name or ''}")

    def choose(self, profile_id: str | None) -> None:
        if profile_id:
            self.profile_id = profile_id
            self._update_prompt()

    def do_use(self, arg: str) -> None:
        pid = self._pick(arg, "profiles")
        if pid is None:
            return
        if pid not in self.rf.store.profiles:
            self.warn(self.ui("unknown_profile", x=pid))
            return
        if pid != self.profile_id:
            self.run, self.history = None, []
        self.profile_id = pid
        self._update_prompt()
        self.do_profile("")

    def do_profile(self, arg: str) -> None:
        if not self._need_subject():
            return
        p = self.rf.store.profiles[self.profile_id or ""]
        self.say(self.ui("subject", name=p.name or p.profile_id, id=p.profile_id, kind=p.kind.value,
                         role=self._role(), lang=self.lang))  # fmt: skip
        if p.summary:
            self.say(p.summary, "dim")

    def do_news(self, arg: str) -> None:
        events = sorted(self.rf.store.events.values(), key=lambda e: e.published_at, reverse=True)
        self.last, self.last_kind = [e.event_id for e in events], "events"
        for i, e in enumerate(events, 1):
            self.say(
                f"{i:4}  {e.published_at:%Y-%m-%d} [{e.language}] {e.title}  {self.style(e.event_id, 'dim')}"
                + self._replayable(e.event_id)
            )

    do_events = do_news

    def do_feed(self, arg: str) -> None:
        if not self._need_subject():
            return
        top = int(arg) if arg.strip().isdigit() else 10
        feed = self.rf.feed(self.profile_id or "", top=top, language=self.lang)
        self.last, self.last_kind = [it.event_id for it in feed.items], "events"
        colour = {"direct": "green", "background": "yellow", "none": "dim"}
        for i, it in enumerate(feed.items, 1):
            tag = self.style(f"{it.relevance:10}", colour.get(it.relevance, "dim"))
            self.say(f"{i:4}  {tag} {it.published_at:%Y-%m-%d} {it.title}{self._replayable(it.event_id)}")
            self.say(f"        {it.reason}", "dim")
        if not feed.items:
            self.say(feed_markdown(feed))
        elif self.rf.replay is not None:
            self.say(self.ui("demo_marks"), "dim")

    # ------------------------------------------------------------------ analysis

    def _show_run(self, run: Any) -> None:
        self.run, self.history = run, []
        if run.trace.config.get("replay"):
            self.say(self.ui("replayed", x=run.trace.config["replay"],
                             role=run.trace.role.value if run.trace.role else "—", lang=run.trace.language), "dim")  # fmt: skip
        if run.insight is None:
            self.warn(self.ui("failed", x=run.trace.failure_reason()))
            return
        self.say()
        self.say(self.style.markdown(insight_markdown(run.insight, self.rf.settings.institution)))
        self.say()
        self.say(self.ui("follow_ups"), "dim")

    def do_analyze(self, arg: str) -> None:
        if not self._need_subject():
            return
        eid = self._pick(arg, "events")
        if eid is None:
            return
        if eid not in self.rf.store.events:
            self.warn(self.ui("unknown_event", x=eid))
            return
        self.say(self.ui("working"), "dim")
        run = self.rf.analyze(
            self.profile_id or "", eid, role=self._role(), language=self.lang, on_progress=self._progress
        )
        self._show_run(run)

    def do_paste(self, arg: str) -> None:
        if not self._need_subject():
            return
        self.say(self.ui("paste_title"), "bold")
        title = self._readline().strip()
        self.say(self.ui("paste_text"), "bold")
        lines = []
        while (line := self._readline()).strip():
            lines.append(line.rstrip("\n"))
        text = "\n".join(lines).strip()
        if len(title) < 3 or len(text) < 20:
            self.warn(self.ui("paste_short"))
            return
        self.say(self.ui("working"), "dim")
        event = {"title": title, "text": text, "publisher": "user"}
        run = self.rf.analyze(
            self.profile_id or "", event, role=self._role(), language=self.lang, on_progress=self._progress
        )
        self._show_run(run)

    def _readline(self) -> str:
        if self.use_rawinput:
            try:
                return input() + "\n"
            except EOFError:
                return ""
        return self.stdin.readline()

    def do_brief(self, arg: str) -> None:
        if not self._need_subject():
            return
        top = int(arg) if arg.strip().isdigit() else 3
        self.say(self.ui("working"), "dim")
        brief = self.rf.brief(self.profile_id or "", top=top, role=self.role, language=self.lang)
        self.say(self.style.markdown(brief_markdown(brief)))

    def do_ask(self, arg: str) -> None:
        question = arg.strip()
        if not question or not self._need_insight():
            return
        res = self.rf.chat(self.run.insight.insight_id, question, self.history).response
        self.say()
        self.say(self.style.markdown(chat_markdown(res, self.rf.settings.institution)))
        self.history += [
            ChatTurn(role="user", text=question[:4000]),
            ChatTurn(role="assistant", text=res.answer[:4000]),
        ]

    def do_sources(self, arg: str) -> None:
        if not self._need_insight():
            return
        ins = self.run.insight
        for c in ins.all_claims():
            self.say(f"{self.style(c.claim_id, 'bold'):6} {self.style(c.origin.value, 'cyan'):10} {c.text}")
            if c.source_ids:
                self.say(f"        ← {', '.join(c.source_ids)}", "dim")

    def do_trace(self, arg: str) -> None:
        if self.run is None:
            self.warn(self.ui("no_insight"))
            return
        self.say(self.style.markdown(trace_markdown(self.run.trace)))

    def do_save(self, arg: str) -> None:
        if self.run is None:
            self.warn(self.ui("no_insight"))
            return
        self.say(self.ui("saved", x=self.rf.save_run(self.run, arg.strip() or "runs")))

    def do_load(self, arg: str) -> None:
        path = Path(shlex.split(arg)[0] if arg.strip() else "")
        if not path.is_file():
            self.warn(self.ui("no_file", x=str(path)))
            return
        run = self.rf.load_run(path)
        self.profile_id = run.trace.profile_id
        self._update_prompt()
        self.say(self.ui("loaded", x=path))
        self._show_run(run)

    # ------------------------------------------------------------------ settings

    def do_role(self, arg: str) -> None:
        role = arg.strip()
        if role not in ROLES:
            self.warn(self.ui("bad_role", x=", ".join(ROLES)))
            return
        self.role = role
        self.say(self.ui("role_set", x=role))

    def do_lang(self, arg: str) -> None:
        lang = arg.strip().lower()
        if lang not in ("ru", "en"):
            self.warn(self.ui("bad_lang"))
            return
        self.lang = norm_lang(lang)
        self.say(self.ui("lang_set", x=lang))

    def do_status(self, arg: str) -> None:
        s = self.rf.settings
        model = (
            "demo (recorded)" if self.rf.is_demo else (s.model_id("main") if s.api_key else self.ui("none"))
        )
        insight = (
            self.run.insight.insight_id if self.run is not None and self.run.insight else self.ui("none")
        )
        self.say(self.ui("status", data=self.rf.store.root or "—", model=model,
                         subject=self.profile_id or self.ui("none"), insight=insight))  # fmt: skip

    def do_clear(self, arg: str) -> None:
        if self.style.on:
            self.out.write("\033[2J\033[H")
            self.out.flush()

    def do_exit(self, arg: str) -> bool:
        if self.interactive:
            self.say(self.ui("bye"), "dim")
        return True

    do_quit = do_exit
    do_EOF = do_exit


def run_shell(
    rf: Refract, *, script: str | None = None, color: bool | None = None, profile: str | None = None
) -> int:
    """Interactive on a terminal; commands from `script` (a file, `-` for stdin) otherwise. A script stops at its
    end; its exit code is 1 if any command failed."""
    if script is not None:
        stream = sys.stdin if script == "-" else open(script, encoding="utf-8-sig")  # noqa: SIM115
        try:
            sh = Shell(rf, stdin=stream, color=False if color is None else color, interactive=False)
            sh.choose(profile)
            sh.cmdloop()
        finally:
            if stream is not sys.stdin:
                stream.close()
        return 1 if sh.failed else 0
    interactive = sys.stdin.isatty()
    sh = Shell(rf, color=color, interactive=interactive, stdin=None if interactive else sys.stdin)
    sh.choose(profile)
    if interactive:
        _load_history()
    while True:
        try:
            sh.cmdloop()
            break
        except KeyboardInterrupt:  # Ctrl+C at the prompt: a fresh line, not an exit
            sh.say("")
            sh.greeted = True
    if interactive:
        _save_history()
    return 0 if interactive else (1 if sh.failed else 0)


# ---------------------------------------------------------------------------------------------------- history


def _has_readline() -> bool:
    try:
        import readline  # noqa: F401 — absent on Windows

        return True
    except ImportError:
        return False


_HISTORY = Path.home() / ".refract_history"


def _load_history() -> None:
    try:
        import readline

        readline.read_history_file(_HISTORY)
        readline.set_history_length(1000)
    except (ImportError, OSError):  # no readline (Windows) or no history yet
        pass


def _save_history() -> None:
    try:
        import readline

        readline.write_history_file(_HISTORY)
    except (ImportError, OSError):
        pass
