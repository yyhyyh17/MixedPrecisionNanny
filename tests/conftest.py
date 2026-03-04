"""
公用 fixtures 与工具函数

- make_stats(**overrides)  : 生成一个 fake TensorStats（不依赖 torch，用于 sqlite_writer 测试）
- tmp_writer(tmp_path)     : 返回 (SQLiteWriter, db_path)，测试结束后自动 close
- query(db_path, sql, ...) : 对 db 执行只读查询，方便断言
"""
from __future__ import annotations

import sqlite3
import sys
import os
from types import SimpleNamespace
from typing import Any, List, Tuple

import pytest

# 确保项目根目录在 sys.path，支持从任意目录运行 pytest
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from storage.sqlite_writer import SQLiteWriter


# ─── TensorStats 工厂（不依赖 torch） ──────────────────────────────────────────

def make_stats(**overrides) -> Any:
    """
    创建一个鸭子类型的 TensorStats 对象，用于 SQLiteWriter 单测。
    默认值对应一个正常的 FP16 tensor，无 NaN/Inf/Saturation/Underflow。
    """
    defaults = dict(
        dtype="torch.float16",
        shape=[4, 8],
        numel=32,
        nan_count=0,
        inf_count=0,
        max_val=1.0,
        min_nonzero=0.01,
        mean_val=0.0,
        std_val=0.5,
        p1=-1.0,
        p99=1.0,
        fp16_saturation=0.0,
        fp16_underflow=0.0,
        exact_zero_ratio=0.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ─── SQLiteWriter fixture ───────────────────────────────────────────────────────

@pytest.fixture
def tmp_writer(tmp_path):
    """
    返回 (writer, db_path) tuple。
    writer 在测试结束后自动 close。
    """
    db_path = str(tmp_path / "test_metrics.db")
    writer = SQLiteWriter(db_path)
    yield writer, db_path
    writer.close()


# ─── DB 查询工具 ────────────────────────────────────────────────────────────────

def db_query(db_path: str, sql: str, params: tuple = ()) -> List[Tuple]:
    """对 SQLite DB 执行只读查询，返回所有行。"""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return rows


def db_count(db_path: str, table: str, where: str = "", params: tuple = ()) -> int:
    """返回指定表的行数（可附加 WHERE 条件）。"""
    sql = f"SELECT COUNT(*) FROM {table}"
    if where:
        sql += f" WHERE {where}"
    return db_query(db_path, sql, params)[0][0]
