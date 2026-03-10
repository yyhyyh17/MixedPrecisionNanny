"""
精度对比可视化 Web 服务

提供：
  - GET  /                — 前端页面
  - GET  /api/reports     — 列出所有已保存的报告
  - GET  /api/report/<id> — 获取指定报告 JSON
  - POST /api/analyze     — 在线分析（需提供模型，仅限同进程调用）

典型用法（命令行启动）：
    python visualization/server.py --report path/to/report.json --port 8501

也可在代码中直接启动：
    from visualization.server import launch_server
    launch_server(report=diff_report, port=8501)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, jsonify, render_template, request

from analyzer.precision_diff import DiffReport

app = Flask(__name__, template_folder="templates")

_REPORT_DIR: str = "./nanny_logs/reports"
_CURRENT_REPORT: Optional[dict] = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reports")
def list_reports():
    """列出 report 目录下所有 JSON 报告文件。"""
    report_dir = app.config.get("REPORT_DIR", _REPORT_DIR)
    if not os.path.isdir(report_dir):
        return jsonify([])
    files = sorted(glob.glob(os.path.join(report_dir, "*.json")), reverse=True)
    reports = []
    for f in files:
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
            reports.append({
                "id": os.path.basename(f).replace(".json", ""),
                "filename": os.path.basename(f),
                "model_name": data.get("model_name", ""),
                "precision": data.get("precision", ""),
                "total_layers": data.get("total_layers", 0),
                "timestamp": data.get("timestamp", 0),
            })
        except Exception:
            continue
    return jsonify(reports)


@app.route("/api/report/<report_id>")
def get_report(report_id: str):
    """获取指定报告的完整 JSON。"""
    if report_id == "current" and _CURRENT_REPORT is not None:
        return jsonify(_CURRENT_REPORT)

    report_dir = app.config.get("REPORT_DIR", _REPORT_DIR)
    path = os.path.join(report_dir, f"{report_id}.json")
    if not os.path.isfile(path):
        return jsonify({"error": f"Report '{report_id}' not found"}), 404
    with open(path, "r") as f:
        data = json.load(f)
    return jsonify(data)


def launch_server(
    report: Optional[DiffReport] = None,
    report_dir: str = "./nanny_logs/reports",
    host: str = "0.0.0.0",
    port: int = 8501,
    debug: bool = False,
) -> None:
    """
    启动可视化服务。

    Args:
        report:     直接传入的 DiffReport（可选，作为 "current" 报告）
        report_dir: 报告文件目录
        host:       绑定地址
        port:       端口
        debug:      Flask debug 模式
    """
    global _CURRENT_REPORT
    if report is not None:
        _CURRENT_REPORT = report.to_dict()
    app.config["REPORT_DIR"] = report_dir
    print(f"[Nanny Visualization] Starting server at http://{host}:{port}")
    if _CURRENT_REPORT is not None:
        print(f"[Nanny Visualization] Current report loaded: {report.model_name} ({report.precision})")
    print(f"[Nanny Visualization] Report directory: {os.path.abspath(report_dir)}")
    app.run(host=host, port=port, debug=debug)


def main():
    parser = argparse.ArgumentParser(description="MixedPrecisionNanny Diff Visualization Server")
    parser.add_argument("--report", type=str, default=None, help="Path to a report JSON file to display")
    parser.add_argument("--report-dir", type=str, default="./nanny_logs/reports", help="Directory containing report JSON files")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    report = None
    if args.report:
        report = DiffReport.from_json(args.report)
        print(f"Loaded report: {args.report}")

    launch_server(
        report=report,
        report_dir=args.report_dir,
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
