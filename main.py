# -*- coding: utf-8 -*-
"""
astrob_plugin_time_gap_sense —— 时间流逝感知插件

LLM 没有"时间流逝"概念：两条对话之间现实里过了一小时，它默认按连续对话
debate。本插件在每次 LLM 请求时：
1. 记录本次用户消息的时间戳与文本（按会话 unified_msg_origin 隔离，
   保留最近 N 条作为状态锚点，滚动覆盖）；
2. 计算与上次发言的时间间隔；
3. 注入「当前时间 + 最近对话 + 时间间隔 + 双向状态指令」回提示词：
   - 间隔短：明确保持上一轮状态（防止时间演进过快：刚说不吃就饿、刚喝水就上厕所）
   - 间隔长：允许状态按现实时间自然演变（饥饿、困倦、生理需求等）

缓存提示：时间块每轮都会变化，注入到尾部（tail）可保持前缀静态，
利于 LLM 前缀缓存；默认 inject_target=tail。

与 AstrBot 内置「现实世界时间感知」(datetime_system_prompt) 互补：
内置给"当前时间"，本插件给"时间间隔 + 状态锚点"。
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star
from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import TextPart

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).parent / "time_gap_state.json"
MAX_TEXT_LEN = 50  # 单条消息锚点最大长度


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _human_gap(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes} 分钟"
    hours, rest = divmod(minutes, 60)
    if rest == 0:
        return f"{hours} 小时"
    return f"{hours} 小时 {rest} 分钟"


class Main(Star):
    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.context = context
        self.config = config or {}

    def _recent_count(self) -> int:
        """记录轮数：默认防呆钳制 1-10；解限开关打开后放宽到 1-100。

        超过 10 的锚点通常没有增量价值（模型注意力集中在近期，且旧状态
        本就该随时间消退），解限仅用于确有极端长连续对话需求的场景。
        """
        try:
            raw = int(self.config.get("recent_count", 3) or 3)
        except (TypeError, ValueError):
            logger.warning("time_gap: recent_count 非整数，回退默认 3")
            return 3
        cap = 100 if bool(self.config.get("unlock_recent_count", False)) else 10
        return max(1, min(cap, raw))

    # ---------- 状态读写（按会话隔离） ----------
    def _load_state(self) -> dict:
        try:
            if STATE_FILE.is_file():
                data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception as exc:  # noqa: BLE001 - 读取失败按空状态处理
            logger.warning("time_gap: 状态文件读取失败: %s", exc)
        return {}

    def _save_state(self, state: dict) -> None:
        try:
            STATE_FILE.write_text(
                json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001 - 写入失败不阻塞主流程
            logger.warning("time_gap: 状态文件写入失败: %s", exc)

    # ---------- 事件：LLM 请求前 → 计算间隔并注入 ----------
    @filter.on_llm_request()
    async def inject_time_gap(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        if not bool(self.config.get("enable", True)):
            return
        # 无会话标识说明是插件内部 LLM 调用（如情感分析），跳过
        umo = getattr(event, "unified_msg_origin", None)
        if not umo:
            return

        now = _local_now()
        msg_text = ""
        getter = getattr(event, "get_message_str", None)
        if callable(getter):
            try:
                msg_text = str(getter() or "").strip()
            except Exception:  # noqa: BLE001 - 取文本失败不阻塞
                msg_text = ""
        if len(msg_text) > MAX_TEXT_LEN:
            msg_text = msg_text[:MAX_TEXT_LEN] + "…"

        state = self._load_state()
        entry = state.get(umo) or {}
        raw_entries = entry.get("entries")
        entries = raw_entries if isinstance(raw_entries, list) else []

        lines = []
        lines.append(f"当前时间：{now.strftime('%Y-%m-%d %H:%M')}")
        if entries:
            lines.append("最近对话：")
            for e in entries:
                t = e.get("ts", "")
                disp = "?"
                try:
                    dt = datetime.fromisoformat(t)
                    disp = dt.strftime("%H:%M") if dt.date() == now.date() else dt.strftime("%m-%d %H:%M")
                except Exception:  # noqa: BLE001 - 时间解析失败用占位
                    pass
                lines.append(f"  {disp} 你：「{str(e.get('text', ''))}」")
            last_ts = entries[-1].get("ts", "")
            try:
                last = datetime.fromisoformat(last_ts)
                minutes = max(0, int((now - last).total_seconds() // 60))
                lines.append(f"距你上次发言：{_human_gap(max(minutes, 1))}")
                if minutes >= int(self.config.get("gap_threshold_minutes", 10) or 10):
                    # 长间隔：放开状态随时间演进
                    lines.append(
                        "（状态请按现实时间流逝自然演变：饥饿、困倦、生理需求等"
                        "随时间推进，不要把对话当作连续发生。若上次提到的事是几小时前，"
                        "该事引发的即时状态应已消退，按当前状态自然演绎）"
                    )
                else:
                    # 短间隔：明确锁状态，防止模型"时间演进过快"
                    lines.append(
                        "（距上次对话仅几分钟，现实时间几乎未流逝：请保持上一轮的"
                        "状态、心情与身体感受不变，不要因时间产生新的生理或情绪变化，"
                        "如突然变饿、变困、想去厕所等）"
                    )
            except Exception as exc:  # noqa: BLE001 - 间隔计算失败只注入时间
                logger.warning("time_gap: 间隔计算失败: %s", exc)

        # 更新锚点队列（滚动，保留最近 N 条）
        entries.append({"ts": now.isoformat(timespec="seconds"), "text": msg_text})
        entries = entries[-self._recent_count():]
        state[umo] = {"entries": entries}
        self._save_state(state)

        block = "<time_context>\n" + "\n".join(lines) + "\n</time_context>"
        target = str(self.config.get("inject_target", "tail") or "tail").strip().lower()
        try:
            if target == "system":
                req.system_prompt = (req.system_prompt or "") + "\n" + block
            else:
                # mark_as_temp：只本轮生效，不持久化进会话历史，
                # 避免多轮累积多个 <time_context> 块（AstrBot >= v4.24.0）
                req.extra_user_content_parts.append(TextPart(text=block).mark_as_temp())
        except Exception as exc:  # noqa: BLE001 - 注入失败不影响主流程
            logger.warning("time_gap: 注入失败: %s", exc)
