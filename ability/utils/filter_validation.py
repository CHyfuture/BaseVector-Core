"""
检索过滤表达式校验（B2方案）
允许用户传入 milvus_expr / sql_where，但必须通过基本安全校验。
"""

from __future__ import annotations

import re
from typing import List, Optional


_DEFAULT_FORBIDDEN_SQL_TOKENS = [
    ";",
    "--",
    "/*",
    "*/",
]

_DEFAULT_FORBIDDEN_SQL_KEYWORDS = [
    "drop",
    "delete",
    "update",
    "insert",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
]

_DEFAULT_SQL_CHAR_WHITELIST = r"[a-zA-Z0-9_\s\(\)\=\!\<\>\.\,\:\'\"\%\&\|\+\-\*/]+"


def validate_milvus_expr(
    expr: str,
    *,
    max_len: int = 500,
    forbidden_chars: Optional[List[str]] = None,
) -> str:
    """
    验证Milvus表达式

    Args:
        expr: 表达式字符串
        max_len: 最大长度限制（默认：500）
        forbidden_chars: 禁止字符列表（默认：[";"]）

    Returns:
        验证后的表达式

    Raises:
        ValueError: 如果表达式不合法
    """
    if expr is None:
        return ""
    expr = str(expr).strip()
    if not expr:
        return ""
    if len(expr) > max_len:
        raise ValueError("milvus_expr too long")

    forbidden = forbidden_chars or [";"]
    for char in forbidden:
        if char in expr:
            raise ValueError(f"milvus_expr contains illegal character '{char}'")
    return expr


def validate_sql_where(
    sql_where: str,
    *,
    max_len: int = 500,
    forbidden_tokens: Optional[List[str]] = None,
    forbidden_keywords: Optional[List[str]] = None,
    char_whitelist: Optional[str] = None,
) -> str:
    """
    验证SQL WHERE表达式

    Args:
        sql_where: SQL WHERE表达式字符串
        max_len: 最大长度限制（默认：500）
        forbidden_tokens: 禁止的SQL令牌列表（默认：[";", "--", "/*", "*/"]）
        forbidden_keywords: 禁止的SQL关键字列表（默认：["drop", "delete", ...]）
        char_whitelist: 字符白名单正则表达式（默认：允许常见WHERE表达式字符）

    Returns:
        验证后的表达式

    Raises:
        ValueError: 如果表达式不合法
    """
    if sql_where is None:
        return ""
    s = str(sql_where).strip()
    if not s:
        return ""
    if len(s) > max_len:
        raise ValueError("sql_where too long")

    forbidden_toks = forbidden_tokens or _DEFAULT_FORBIDDEN_SQL_TOKENS
    forbidden_kws = forbidden_keywords or _DEFAULT_FORBIDDEN_SQL_KEYWORDS
    whitelist = char_whitelist or _DEFAULT_SQL_CHAR_WHITELIST

    lowered = s.lower()
    for tok in forbidden_toks:
        if tok in lowered:
            raise ValueError(f"sql_where contains forbidden token: {tok}")
    for kw in forbidden_kws:
        if re.search(rf"\b{re.escape(kw)}\b", lowered):
            raise ValueError(f"sql_where contains forbidden keyword: {kw}")

    # 基础字符白名单（尽量允许常见WHERE表达式，但拒绝明显异常字符）
    if not re.fullmatch(whitelist, s):
        raise ValueError("sql_where contains illegal characters")

    return s

