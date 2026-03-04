"""
SQLite 异步写入器

设计要点：
  - 写操作全部放入队列，由独立 daemon 线程消费，不阻塞训练主循环
  - 攒批后一次性提交事务，减少 I/O 次数
  - 开启 WAL 模式，允许并发读（CLI 查询）不阻塞写
  - flush() / close() 保证程序退出前所有数据落盘
"""
from __future__ import annotations

import os
import queue
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

# 每批最多提交多少条记录
_BATCH_SIZE = 200

# 队列空闲超时（秒）：超时后将已积累的不足一批的记录提前写盘
_FLUSH_IDLE_SEC = 0.5

# 表示关闭信号的哨兵对象
_SENTINEL = object()

# flush 请求的 kind 标识
_FLUSH_KIND = "__flush__"

# ─── DDL ───────────────────────────────────────────────────────────────────────

_CREATE_TABLES_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS layer_stats (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    step         INTEGER NOT NULL,
    phase        TEXT    NOT NULL,   -- 'forward' | 'backward'
    layer_name   TEXT    NOT NULL,
    layer_type   TEXT,
    dtype        TEXT,
    nan_count    INTEGER DEFAULT 0,
    inf_count    INTEGER DEFAULT 0,
    max_val      REAL,
    min_nonzero  REAL,
    mean_val     REAL,
    std_val      REAL,
    p1           REAL,
    p99          REAL,
    fp16_sat     REAL,               -- FP16 饱和率
    fp16_udf     REAL,               -- FP16 下溢率
    exact_zero_ratio REAL,            -- 精确为 0 的比例（BF16/FP16 直接下溢检测）
    ts           REAL                -- Unix timestamp
);

CREATE TABLE IF NOT EXISTS alerts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    step         INTEGER NOT NULL,
    phase        TEXT,
    layer_name   TEXT,
    alert_type   TEXT    NOT NULL,   -- NAN | INF | OVERFLOW | UNDERFLOW | ...
    severity     TEXT    NOT NULL,   -- ERROR | WARNING
    message      TEXT,
    value        REAL,
    ts           REAL
);

CREATE TABLE IF NOT EXISTS loss_scale_history (
    step         INTEGER PRIMARY KEY,
    scale        REAL    NOT NULL,
    overflow     INTEGER DEFAULT 0,  -- 1 表示本 step 发生了 overflow（scale 下降）
    ts           REAL
);

CREATE INDEX IF NOT EXISTS idx_stats_step   ON layer_stats(step);
CREATE INDEX IF NOT EXISTS idx_stats_layer  ON layer_stats(layer_name);
CREATE INDEX IF NOT EXISTS idx_alerts_step  ON alerts(step);
CREATE INDEX IF NOT EXISTS idx_alerts_sev   ON alerts(severity);
"""

_INSERT_STATS = """
INSERT INTO layer_stats
    (step, phase, layer_name, layer_type, dtype,
     nan_count, inf_count, max_val, min_nonzero,
     mean_val, std_val, p1, p99, fp16_sat, fp16_udf, exact_zero_ratio, ts)
VALUES
    (:step, :phase, :layer_name, :layer_type, :dtype,
     :nan_count, :inf_count, :max_val, :min_nonzero,
     :mean_val, :std_val, :p1, :p99, :fp16_sat, :fp16_udf, :exact_zero_ratio, :ts)
"""

_INSERT_ALERT = """
INSERT INTO alerts
    (step, phase, layer_name, alert_type, severity, message, value, ts)
VALUES
    (:step, :phase, :layer_name, :alert_type, :severity, :message, :value, :ts)
"""

_INSERT_SCALE = """
INSERT OR REPLACE INTO loss_scale_history (step, scale, overflow, ts)
VALUES (:step, :scale, :overflow, :ts)
"""


# ─── 写入器 ────────────────────────────────────────────────────────────────────

class SQLiteWriter:
    """
    线程安全的 SQLite 写入器。

    public API：
        write_stats(...)        — 写入一条 layer 统计记录
        write_alert(...)        — 写入一条告警记录
        write_loss_scale(...)   — 写入一条 Loss Scale 记录
        flush(timeout)          — 阻塞直到队列清空（数据全部落盘）
        close()                 — flush 后关闭后台线程
    """

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db_path = db_path

        # 先在主线程建表，确保其他代码可以立即读取结构
        self._init_schema()

        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="nanny-db-writer",
        )
        self._thread.start()

    # ─── 公共写入接口 ────────────────────────────────────────────────────────────

    def write_stats(
        self,
        step: int,
        phase: str,
        layer_name: str,
        layer_type: str,
        stats: Any,          # analyzer.numerical_checker.TensorStats
        ts: Optional[float] = None,
    ) -> None:
        self._q.put(("stats", {
            "step":        step,
            "phase":       phase,
            "layer_name":  layer_name,
            "layer_type":  layer_type,
            "dtype":       stats.dtype,
            "nan_count":   stats.nan_count,
            "inf_count":   stats.inf_count,
            "max_val":     stats.max_val,
            "min_nonzero": stats.min_nonzero,
            "mean_val":    stats.mean_val,
            "std_val":     stats.std_val,
            "p1":          stats.p1,
            "p99":         stats.p99,
            "fp16_sat":    stats.fp16_saturation,
            "fp16_udf":    stats.fp16_underflow,
            "exact_zero_ratio": getattr(stats, "exact_zero_ratio", 0.0),
            "ts":          ts or time.time(),
        }))

    def write_alert(
        self,
        step: int,
        phase: str,
        layer_name: str,
        alert_type: str,
        severity: str,
        message: str,
        value: float,
        ts: Optional[float] = None,
    ) -> None:
        self._q.put(("alert", {
            "step":       step,
            "phase":      phase,
            "layer_name": layer_name,
            "alert_type": alert_type,
            "severity":   severity,
            "message":    message,
            "value":      value,
            "ts":         ts or time.time(),
        }))

    def write_loss_scale(
        self,
        step: int,
        scale: float,
        overflow: bool = False,
    ) -> None:
        self._q.put(("scale", {
            "step":     step,
            "scale":    scale,
            "overflow": int(overflow),
            "ts":       time.time(),
        }))

    def flush(self, timeout: float = 30.0) -> None:
        """阻塞直到所有已入队数据真正提交到 SQLite。

        向队列投递一个 flush 事件；worker 收到后立即 commit 缓冲区，
        并 set() 事件通知调用方。这样保证 flush() 返回时数据已落盘。
        """
        evt = threading.Event()
        self._q.put((_FLUSH_KIND, evt))
        if not evt.wait(timeout=timeout):
            raise TimeoutError(f"SQLiteWriter.flush() timed out after {timeout}s")

    def close(self) -> None:
        """停止后台线程前先把队列清空并落盘。"""
        self.flush()
        self._q.put(_SENTINEL)
        self._thread.join(timeout=15.0)

    # ─── 初始化 ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(_CREATE_TABLES_SQL)
            # 兼容旧 DB：若表已存在且无 exact_zero_ratio 列则追加
            try:
                conn.execute("ALTER TABLE layer_stats ADD COLUMN exact_zero_ratio REAL")
            except sqlite3.OperationalError:
                pass  # 列已存在或表结构已包含
            conn.commit()
        finally:
            conn.close()

    # ─── 后台线程 ──────────────────────────────────────────────────────────────

    def _worker(self) -> None:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous  = NORMAL")

        stats_buf: List[Dict] = []
        alert_buf: List[Dict] = []
        scale_buf: List[Dict] = []

        def _flush() -> None:
            if stats_buf:
                conn.executemany(_INSERT_STATS, stats_buf)
                stats_buf.clear()
            if alert_buf:
                conn.executemany(_INSERT_ALERT, alert_buf)
                alert_buf.clear()
            if scale_buf:
                conn.executemany(_INSERT_SCALE, scale_buf)
                scale_buf.clear()
            conn.commit()

        while True:
            # 尝试从队列取一条记录；超时则把积压的数据先写盘
            try:
                item = self._q.get(timeout=_FLUSH_IDLE_SEC)
            except queue.Empty:
                if stats_buf or alert_buf or scale_buf:
                    try:
                        _flush()
                    except Exception as exc:
                        print(f"[Nanny][WARN] DB idle-flush error: {exc}")
                continue

            # 收到关闭信号
            if item is _SENTINEL:
                try:
                    _flush()
                except Exception as exc:
                    print(f"[Nanny][WARN] DB close-flush error: {exc}")
                self._q.task_done()
                break

            kind, data = item

            # flush 请求：立即提交缓冲区，然后通知调用方
            if kind == _FLUSH_KIND:
                try:
                    _flush()
                except Exception as exc:
                    print(f"[Nanny][WARN] DB flush error: {exc}")
                finally:
                    data.set()  # 通知 flush() 调用方数据已落盘
                self._q.task_done()
                continue

            if kind == "stats":
                stats_buf.append(data)
            elif kind == "alert":
                alert_buf.append(data)
            elif kind == "scale":
                scale_buf.append(data)

            total = len(stats_buf) + len(alert_buf) + len(scale_buf)
            if total >= _BATCH_SIZE:
                try:
                    _flush()
                except Exception as exc:
                    print(f"[Nanny][WARN] DB batch-flush error: {exc}")

            self._q.task_done()

        conn.close()
