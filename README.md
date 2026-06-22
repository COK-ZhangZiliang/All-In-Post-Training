<p align="center">
  <img src="assets/icon.svg" alt="All-In Post-Training icon" width="112" height="112">
</p>

<h1 align="center">All-In Post-Training</h1>

<p align="center">
  一个面向 LLM 后训练研究、工程与 Agentic RL 的可维护全景图。
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-0f766e"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-2563eb">
  <img alt="Status" src="https://img.shields.io/badge/status-initial%20framework-f59e0b">
</p>

## 项目定位

All-In Post-Training 把后训练知识整理成一张可演进的研究地图：从 SFT、RLHF、DPO，到 RLVR、GRPO/DAPO/CISPO，再到 OPD/MOPD、多能力融合、长周期 Agentic RL、沙箱环境与评测体系。首版重点是数据结构、研究种子、静态全景页面和后续迭代计划。

## 当前能力

- 数据驱动知识库：`data/panorama.json` 保存论文、方法、系统、关系边和路线图元数据。
- 离线可运行 CLI：不依赖第三方包即可校验数据并生成静态页面。
- 静态全景图：生成 `site/index.html` 后可直接用浏览器打开，包含搜索、轨道筛选、节点详情和 SVG 关系图。
- 项目治理：`PLAN.md` 记录研究与工程路线，`AGENTS.md` 固化协作、提交和验证规则。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
all-in-post-training validate
all-in-post-training build --out site
python -m http.server 8000 --directory site
```

不安装包也可以直接运行：

```bash
PYTHONPATH=src python3 -m all_in_post_training.cli validate
PYTHONPATH=src python3 -m all_in_post_training.cli build --out site
```

然后打开 `http://localhost:8000`。

## 仓库结构

```text
.
├── AGENTS.md                         # 项目协作、验证和 Git 规则
├── PLAN.md                           # 后训练全景图路线图
├── README.md                         # 项目说明
├── data/panorama.json                # 研究全景图数据源
├── src/all_in_post_training/         # CLI、数据校验和静态站点生成器
├── assets/icon.svg                   # 项目图标
└── tests/                            # 离线单元测试
```

## 研究范围

首版将文档知识和外部调研归纳为六条轨道：

1. 对齐基础：SFT、RLHF、DPO、偏好数据。
2. Reasoning RL：RLVR、GRPO、DAPO、CISPO、可验证奖励。
3. 多能力融合：GKD/OPD、Specialist RL、MOPD、TGPO、SDPO。
4. Agentic RL：多轮环境、turn/step-level credit assignment、dense reward。
5. 工程基础设施：沙箱、rollout、replay、异步调度、prefix tree。
6. 评测与安全：SWE-bench、Tool use、long-horizon eval、安全边界。

## 常用命令

```bash
PYTHONPATH=src python3 -m all_in_post_training.cli validate
PYTHONPATH=src python3 -m all_in_post_training.cli stats
PYTHONPATH=src python3 -m all_in_post_training.cli build --out site
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## License

This project is released under the [MIT License](LICENSE).

