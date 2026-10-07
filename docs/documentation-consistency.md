# 文档与实现一致性修正

更新：2026-10-07。依据此前只读审查清单，在 `D:\rag` 的 `codex/branch-sync-review`
执行；开始时 HEAD 为 `362c4ee`、本地 main 为 `9278053`，README 已有暂存修改。
新增修正保留在工作区，没有暂存、提交或推送；README 原有暂存内容保留。

## 已修正

| 项目 | 实现与证据 |
|---|---|
| README 双图合同 | 保留原有两张流程图及暂存设计；更新 `tests/test_doc_layout.py` 的旧单图约束 |
| 开发环境 | `docs/evaluation.md` 安装 `.[dev,service]`，与 CI 服务测试依赖一致；区分安装依赖与离线实验 |
| 消融摘要 | `scripts/sync_ablation_docs.py` 复用本地缓存、原排名与检验合同，同时生成 README 和详细评估的 `ABLATION-SUMMARY`；补齐配对差值区间 |
| 统计解释 | MRL 单证据 R@1 是二元指标，但历史方法确为单尾配对 bootstrap + Holm；未显著不表示等价、充分功效或机制证明 |
| 词法与融合 | 客户端 jieba 对照不等于数据库 analyzer 端到端实测；加权 RRF 不等于 Milvus `WeightedRanker`；同批融合选型后的原始 p 标为探索性 |
| 查询合同 | 架构说明列出 TiDB 与 CRUD 新闻的实际独立查询前缀，注明常量来源及文档侧空前缀 |
| 分块范围 | `corpus_stats.py` 重算确认入库统计对应 450 篇、1,832 块；完整 manifest 为 500 篇 |
| 度量与路线图 | 区分 COSINE 和 IP 的归一化要求；CRUD 的 H+rerank 与 TiDB 分块实验分开；旧成本表标为历史假设 |
| 生成与历史状态 | 补充供应商流式、无显式 token 上限、直连传输及 ASCII 模型身份规则；旧流式失败和前端交接标明历史范围 |
| 本地项目入口 | 修复 Git 忽略的 `AGENTS.md` 的失效记忆链接；v2 为 79 条模型审核、人工 0 条、dev 48 / test 31 |

## 验证

- 新增同步器与文档布局定向测试：31 项通过。
- 全套 Python：1,779 项通过；JUnit：`.research_tmp/docs-consistency-20261007/pytest.xml`。
- `ruff check src tests scripts`、`ruff format --check src tests scripts`、`mypy` 通过。
- `corpus_stats.py` 完成离线重算；完整真实缓存的摘要重算已认证并写入两份文档。
- JUnit 数量同步及 `--check`、M9a/M9b 文档 `--check` 通过。
- 摘要独立 `--check` 使用本机已安装的 Python 3.13.5、原 `.venv` 依赖包完成；
  CRUD/MRL 与 TiDB 全部重算认证通过，输出 `0 document(s) updated`，两份摘要完全一致。
- 两次沙箱重算进入 Windows 系统等待，停止后仅清理了本次进程的临时锁；获准在沙箱外运行
  本地 `verify_offline.py` 后完成首次认证。之后 Python 3.13.3 的独立复验出现解释器级
  `Executing a cache` 崩溃；该失败保留，未当作通过。改用本机现有 3.13.5 后独立复验成功，
  没有安装依赖、修改 `.venv` 或更改历史缓存。校验阶段禁止网络连接与读取 `.env`，未触发这些限制。
- 本轮修改的 Markdown 相对文件链接和 `git diff --check` 通过；开始时的 HEAD、main 与
  README 暂存 blob 均保持原值。没有残留本次进程的产物锁或发布临时文件。
- 全套 pytest 有一项已有 Starlette/httpx 弃用警告；未更改依赖。

本轮仅修改文档、文档同步器及其测试，未改前端和在线服务实现；未新增付费模型调用。
本地缓存和审核结果保留在 Git 忽略目录，公开文档只发布聚合信息。
后续仍按[精简交接](frontend-next-session.md)完善既有 Agent 离线审核与报告，再建立失败回放。
