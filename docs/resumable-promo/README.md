# EvidenceRAG 可恢复调查宣传片

展示文档调查的暂停、继续，以及服务重启后自动重连、从保存点继续：从切换到“连接服务”开始，经过提问、追问与补充、暂停与继续、服务关闭与重启，到带引用的答案，再用检查器查看单次调用、同类调用、证据原文与运行概览。画面来自 frontend 生产构建与仓库原创合成后端的录制回放，没有调用模型，也没有使用真实语料。

- [播放成片 v02](versions/v02/final/evidencerag-resumable-v02.mp4)（51.5 秒，1920×1080，含配乐）
- [21 帧检查拼图](versions/v02/final/check-sheet.png)、[运镜与品牌转场拼图](versions/v02/final/transitions.png)
- [制作说明、相对 v01 的变化、验证记录与复现命令](versions/v02/README.md)
- [已确认的分镜总览](versions/v02/preview/overview.jpg)
- 归档：[v01 成片](versions/v01/final/evidencerag-resumable-v01.mp4)（43 秒，无检查器段）与[制作说明](versions/v01/README.md)
- 本地入口页 [index.html](index.html)，读取 [status.json](status.json) 显示当前版本状态

Git 保存成片、编辑源码、时间轴和检查记录。预览页、关键帧与无声小样由本目录 `.gitignore` 排除；中间帧、构建产物、音轨与日志在 Git 忽略的 `.research_tmp/resumable-promo/<版本>/`，都可按复现命令在本地重建。
品牌卡图标沿用产品的 Lucide `layers-2`，许可见 [licenses/Lucide-ISC.txt](licenses/Lucide-ISC.txt)。
