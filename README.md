# CreditMosaic AI

CreditMosaic AI 是一个基于 LLM 新闻信号的跨市场信用风险预警 MVP。项目聚焦“新闻文本如何通过股票市场、信用代理变量和公司风险指标产生异步反应”，将非结构化新闻转化为可建模、可解释、可回测的信用风险信号。

本项目用于科研、教学和金融科技应用演示，不构成投资建议、交易建议或信用评级意见。

## 1. 项目定位

CreditMosaic AI 的第一版目标不是聊天助手，而是一个新闻驱动的信用风险预警平台：

- 输入公司 ticker 或投资组合。
- 采集公司新闻、股价、成交量、基本面和信用代理变量。
- 使用 Longcat/OpenAI-compatible LLM 抽取结构化 `LLMNewsSignal`。
- 使用 FinBERT、Logistic Regression、LightGBM、XGBoost 做基线和主模型对照。
- 展示公司风险、新闻证据、跨市场反应、组合风险贡献和 Markdown 风险报告。

## 2. 当前模块

```text
creditmosaic-ai/
  apps/api/                 FastAPI 后端 API
  db/                       PostgreSQL schema 与迁移
  models/                   已训练模型与评估报告
  pipelines/ingestion/      新闻、行情、基本面、信用代理变量采集
  pipelines/risk/           特征工程、风险标签、模型训练、风险评分
  pipelines/reaction/       跨市场反应和 lead-lag 分析
  services/                 LLM provider、FinBERT、新闻抽取、风险/反应服务
  web/                      Next.js 前端
  docker-compose.yml        本地一键启动入口
```

## 3. 技术栈

| 层级 | 当前实现 |
| --- | --- |
| 前端 | Next.js, React, TypeScript, ECharts |
| 后端 | FastAPI, Python |
| 数据库 | PostgreSQL, DuckDB |
| LLM | Longcat OpenAI-compatible provider, 可扩展 OpenAI/Qwen/DeepSeek |
| NLP baseline | FinBERT |
| 风险模型 | Logistic Regression, LightGBM, XGBoost |
| 可解释性 | Top feature importance, SHAP 支持 |
| 部署 | Docker Compose, 本地优先 |

## 4. 环境变量

复制 `.env.example` 为 `.env`，只在 `.env` 或 shell 环境里放真实密钥。

```powershell
Copy-Item .env.example .env
```

核心配置：

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=creditmosaic
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

API_PORT=8000
WEB_PORT=3000

LONGCAT_API_KEY=
```

`config.yaml` 中默认 LLM provider 是 `longcat`，密钥通过 `${LONGCAT_API_KEY}` 注入，不应写入仓库文件。

## 5. Docker Compose 启动

如果本机已安装 Docker：

```powershell
docker compose up --build
```

服务地址：

- Web: <http://localhost:3000>
- API: <http://localhost:8000>
- OpenAPI: <http://localhost:8000/docs>
- PostgreSQL: `localhost:5432`

PostgreSQL 首次启动时会自动执行 `db/schema.sql`。如果修改 schema 后需要重新初始化数据库，请先确认数据可删除，再移除 compose volume。

```powershell
docker compose down -v
docker compose up --build
```

## 6. 本地开发启动

### 6.1 后端

```powershell
cd F:\Fintech\creditmosaic-ai
.venv\Scripts\python.exe -m uvicorn apps.api.app.main:app --reload --host 127.0.0.1 --port 8000
```

后端健康检查：

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health
```

### 6.2 前端

```powershell
cd F:\Fintech\creditmosaic-ai\web
npm run dev -- --hostname 127.0.0.1 --port 3000
```

前端默认将 `/api/*` 代理到 `NEXT_PUBLIC_API_URL`，未设置时使用 `http://localhost:8000`。

## 7. 数据准备

数据库 schema 位于 `db/schema.sql`，核心表包括：

- `companies`
- `news_items`
- `llm_news_signals`
- `daily_market_data`
- `financial_fundamentals`
- `credit_proxy_data`
- `risk_labels`
- `risk_scores`
- `portfolios`
- `portfolio_analyses`
- `risk_reports`

数据管线入口位于 `pipelines/ingestion/`，目标是完成新闻、股价、基本面、信用代理变量的批量采集和实体匹配。

## 8. 模型与实验

风险建模代码位于 `pipelines/risk/`：

- `risk_labeler.py`: 构造异常收益、异常成交量、波动率跳升、信用代理恶化等标签。
- `feature_engineer.py`: 合并市场特征、财务特征、LLM 新闻特征和 FinBERT 特征。
- `model_trainer.py`: 训练 Logistic Regression、LightGBM、XGBoost。
- `risk_scorer.py`: 输出公司日频风险分数和 Top drivers。

当前模型文件位于 `models/`，评估结果位于 `models/evaluation_report.json`。

## 9. API 概览

主要 API：

| API | 用途 |
| --- | --- |
| `POST /api/news/extract` | 单条新闻 LLM 风险信号抽取 |
| `GET /api/company/{ticker}/risk` | 公司风险总览 |
| `GET /api/company/{ticker}/signals` | 公司新闻信号列表 |
| `POST /api/risk/labels/generate` | 生成风险标签 |
| `POST /api/risk/models/train` | 训练风险模型 |
| `POST /api/risk/scores/generate` | 生成风险评分 |
| `POST /api/reaction/analyze` | 跨市场反应分析 |
| `POST /api/reaction/lag` | lead-lag 滞后分析 |
| `POST /api/portfolio/analyze` | 投资组合风险贡献 |
| `POST /api/portfolio/report` | 组合 Markdown 风险报告 |
| `POST /api/report/generate` | 公司 Markdown 风险报告 |
| `GET /api/alerts` | 风险预警列表 |

## 10. 前端页面

| 页面 | 功能 |
| --- | --- |
| `/` | 项目首页和搜索入口 |
| `/company/[ticker]` | 公司风险总览 |
| `/news/[news_id]` | 新闻信号证据页 |
| `/reaction/[ticker]` | 跨市场反应页 |
| `/portfolio` | 组合风险分析与组合报告 |
| `/report/[ticker]` | 公司风险报告 |

## 11. Demo 闭环

目标演示路径：

```text
输入公司或组合
-> 查看新闻风险信号
-> 查看跨市场反应
-> 查看风险评分和 Top drivers
-> 生成一页式 Markdown 风险报告
```

推荐演示 ticker：`AAPL`、`TSLA`、`NVDA`、`JPM`、`BA`。

## 12. 验证命令

```powershell
cd F:\Fintech\creditmosaic-ai
python -m compileall apps services pipelines -q

@'
from apps.api.app.main import app
schema = app.openapi()
required = [
    "/api/news/extract",
    "/api/company/{ticker}/risk",
    "/api/company/{ticker}/signals",
    "/api/portfolio/analyze",
    "/api/report/generate",
    "/api/alerts",
]
print({"paths": len(schema["paths"]), "missing": [p for p in required if p not in schema["paths"]]})
'@ | .venv\Scripts\python.exe -

cd F:\Fintech\creditmosaic-ai\web
npm run lint
npm run build
```

## 13. 当前限制

- 免费新闻源覆盖不完整，部分新闻可能延迟或缺正文。
- MVP 使用 HYG、LQD、VIX 等信用代理变量，不直接依赖 Bloomberg、TRACE 或 Refinitiv。
- LLM 输出需要 JSON schema 校验和人工抽样复核。
- 风险分数是科研和风险监控信号，不是投资建议。
- 完整科研验收仍需要补齐 10 家以上公司案例和 Model A-E 对照实验。

## 14. 安全与密钥管理

- 不要把 `.env`、真实 API key、数据库密码或访问令牌提交到 Git。
- Longcat key 只通过 `LONGCAT_API_KEY` 环境变量注入。
- `.gitignore` 和 `.dockerignore` 已排除 `.env`、`.venv`、`web/node_modules`、`web/.next`、数据库文件和缓存文件。
- `config.yaml` 只保留 `${LONGCAT_API_KEY}` 形式的变量引用。
- 推送前建议运行一次明文密钥扫描，确认没有 `sk-...`、真实 token 或私密配置进入源码。

## 15. 运行前提与排障

完整运行需要以下服务可用：

- PostgreSQL: `localhost:5432`，数据库名默认 `creditmosaic`。
- FastAPI: `localhost:8000`。
- Next.js: `localhost:3000`。
- Longcat API key: 仅当调用 LLM 抽取或报告生成时需要。

常见问题：

| 现象 | 处理 |
| --- | --- |
| `/api/*` 返回 500 或连接失败 | 确认 FastAPI 后端已启动，且 `NEXT_PUBLIC_API_URL` 指向正确后端 |
| 后端启动时报数据库连接失败 | 确认 PostgreSQL 正在运行，或使用 `docker compose up --build` |
| LLM provider 数量为 0 | 检查 `.env` 或 shell 是否设置 `LONGCAT_API_KEY` |
| 首次 Docker 启动后没有表 | 确认 `db/schema.sql` 已挂载；必要时执行 `docker compose down -v` 后重建 |
| FinBERT 加载慢 | 首次运行会下载模型，属于正常现象 |

## 16. 后续路线

1. 补齐 50 家公司 6 个月可查询样本数据。
2. 完成 Model A-E 对照实验和 LLM 相对 FinBERT 的增量价值分析。
3. 输出 10 家公司案例研究和研究简报。
4. 增加 Prefect 调度和更完整的数据质量报告。
5. 扩展 PDF 报告、watchlist 和风险订阅。
