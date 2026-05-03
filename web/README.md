# CreditMosaic AI Web

这是 CreditMosaic AI 的前端子项目，完整项目说明、Docker 启动、后端 API、数据管线和模型流程请阅读项目根目录的 [README.md](../README.md)。

## 本地启动

```powershell
cd F:\Fintech\creditmosaic-ai\web
npm install
npm run dev -- --hostname 127.0.0.1 --port 3000
```

默认情况下，前端会把 `/api/*` 代理到 `NEXT_PUBLIC_API_URL`；未设置时使用 `http://localhost:8000`。

## 常用命令

```powershell
npm run lint
npm run build
npm run start
```

## 页面

- `/`
- `/company/[ticker]`
- `/news/[news_id]`
- `/reaction/[ticker]`
- `/portfolio`
- `/report/[ticker]`

本前端仅用于科研、教学和风险分析演示，不构成投资建议。
