# TiDB 中文 RAG 精选语料

该目录从 `pingcap/docs-cn` 的固定 commit `26f202bcb1b314ca21f63bdb931f3c90426a14a1` 中分层挑选 500 篇 Markdown 文档，目标是数据规模可控、主题覆盖均衡、来源可追溯。

## 目录结构

- `documents/core_ops/`：运维、部署、配置、排障、备份恢复及数据生态工具。
- `documents/dev_reference/`：应用开发、SQL、AI/向量搜索、最佳实践与 FAQ。
- `documents/temporal_releases/`：版本发布说明，独立检索，不默认混入主语料。
- `selected_manifest.json`：选样策略、主题配额和上游路径。
- `corpus_manifest.jsonl`：每篇本地文件的向量库导入元数据与校验值。
- `download_report.json`：下载数量、字节数及集合统计。
- `select_curated.ps1`：可复现的选样脚本。
- `download_curated.ps1`：断点复用、下载和完整性校验脚本。

## 推荐用法

主向量索引默认只导入 `core_ops` 与 `dev_reference`，共 450 篇。仅在问题明确涉及版本号、升级差异或发布时间时，再路由到 `temporal_releases` 的 50 篇发布说明。

切分时按 Markdown 标题层级切块，建议每块约 400–700 tokens，重叠 60–100 tokens。代码块和表格尽量保持完整，并把 `path`、`source_url`、`source_commit`、`collection`、`theme`、标题层级写入 chunk 元数据。

重新下载或验证：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\download_curated.ps1
```

