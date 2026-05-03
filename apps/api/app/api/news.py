"""
News API Module
Provides news query and LLM signal extraction REST endpoints.

Uses dependency injection for database, LLM, and FinBERT access.
The legacy global singleton (NewsService / initialize_news_api) is retained
for backward compatibility with pipeline scripts and direct imports.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import time
import os
from pathlib import Path

from apps.api.app.dependencies import (
    get_db, get_llm_manager, get_news_extractor, get_container,
)
from apps.api.app.schemas.news import (
    NewsExtractRequest, NewsExtractResponse, NewsItemResponse,
    NewsDetailResponse, SignalResponse, BatchExtractRequest,
    BatchExtractResponse, CompareFinBERTResponse,
)

from services.llm_provider import (
    LLMProviderManager, LLMConfig, OpenAIProvider,
    QwenProvider, DeepSeekProvider,
)
from services.news_signal_extractor import NewsSignalExtractor, NewsItem, LLMNewsSignal
from services.finbert_baseline import FinBERTModel, FinBERTComparator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["News"])


# ---------------------------------------------------------------------------
# Legacy singleton (retained for pipeline scripts that import from this module)
# ---------------------------------------------------------------------------

class NewsService:
    """Legacy service class retained for backward compatibility.

    New code should use the DI container (apps.api.app.dependencies)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_manager = None
        self.extractor = None
        self.finbert = None
        self.db = None

    def initialize(self):
        try:
            self._load_local_env()
            self.llm_manager = LLMProviderManager()

            for provider_name, provider_config in self.config.get("llm_providers", {}).items():
                api_key = self._resolve_secret(provider_config.get("api_key", ""))
                if not api_key:
                    logger.warning("Skipping LLM provider '%s': API key is not configured", provider_name)
                    continue

                llm_config = LLMConfig(
                    provider=provider_config.get("provider", provider_name),
                    api_key=api_key,
                    model_name=provider_config.get("model", ""),
                    base_url=provider_config.get("base_url", ""),
                    max_tokens=provider_config.get("max_tokens", 1000),
                    temperature=provider_config.get("temperature", 0.3),
                )
                provider = self._create_provider(llm_config)
                self.llm_manager.add_provider(provider_name, provider)

            if self.config.get("default_provider"):
                default_provider = self.config["default_provider"]
                if default_provider in self.llm_manager.list_providers():
                    self.llm_manager.set_default_provider(default_provider)
                else:
                    logger.warning(
                        "Default LLM provider '%s' is not available; configured providers: %s",
                        default_provider,
                        self.llm_manager.list_providers(),
                    )

            self.extractor = NewsSignalExtractor(self.llm_manager)

            self.finbert = FinBERTModel(self.config.get("finbert_model", "ProsusAI/finbert"))
            self.finbert.initialize()

            from pipelines.ingestion.db_manager import DatabaseManager
            postgres_config = self.config["postgresql"]
            duckdb_path = self.config["duckdb"]["path"]
            self.db = DatabaseManager(postgres_config, duckdb_path)
            self.db.connect()

            logger.info("News service initialized successfully")

        except Exception as e:
            logger.error(f"News service initialization failed: {e}")
            raise

    def _load_local_env(self):
        env_path = Path.cwd() / ".env"
        if not env_path.exists():
            return
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    def _resolve_secret(self, value: str) -> str:
        if value is None:
            return ""
        value = str(value).strip()
        if value.startswith("${") and value.endswith("}"):
            value = os.environ.get(value[2:-1], "").strip()
        placeholder_tokens = ("YOUR_", "TODO", "CHANGE_ME", "PASTE_", "<")
        if not value or any(value.upper().startswith(token) for token in placeholder_tokens):
            return ""
        return value

    def _create_provider(self, config: LLMConfig):
        if config.provider.lower() == "openai":
            return OpenAIProvider(config)
        elif config.provider.lower() == "qwen":
            return QwenProvider(config)
        elif config.provider.lower() == "deepseek":
            return DeepSeekProvider(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

    async def extract_news_signal(
        self, news_item: NewsItem, provider_name: Optional[str] = None, model: Optional[str] = None
    ) -> Optional[LLMNewsSignal]:
        if not self.extractor:
            raise RuntimeError("News extractor not initialized")
        return await self.extractor.extract_signal(news_item, provider_name)

    async def batch_extract_signals(
        self, news_items: List[NewsItem], provider_name: Optional[str] = None, max_concurrent: int = 5
    ) -> List[LLMNewsSignal]:
        if not self.extractor:
            raise RuntimeError("News extractor not initialized")
        return await self.extractor.extract_batch_signals(news_items, provider_name, max_concurrent)

    def save_signal_to_db(self, signal: LLMNewsSignal, news_id: int) -> bool:
        if not self.db:
            raise RuntimeError("Database not initialized")
        try:
            import pandas as pd
            signal_data = {
                "news_id": news_id, "ticker": signal.ticker,
                "sentiment_score": signal.sentiment_score,
                "credit_risk_score": signal.credit_risk_score,
                "event_type": signal.event_type,
                "risk_horizon": signal.risk_horizon,
                "market_impact_type": signal.market_impact_type,
                "evidence_spans": signal.evidence_spans,
                "confidence": signal.confidence,
                "extracted_at": signal.extracted_at,
                "llm_model": signal.llm_model,
            }
            df = pd.DataFrame([signal_data])
            self.db.insert_dataframe_postgres(df, 'llm_news_signals')
            logger.info(f"Signal saved: news_id={news_id}")
            return True
        except Exception as e:
            logger.error(f"Signal save failed: {e}")
            return False

    def insert_news_item(self, news_item: NewsItem) -> int:
        if not self.db:
            raise RuntimeError("Database not initialized")
        import pandas as pd
        df = pd.DataFrame([{
            "ticker": news_item.ticker, "title": news_item.title,
            "body": news_item.body, "source": news_item.source,
            "url": news_item.url, "published_at": news_item.published_at,
            "is_processed": False,
        }])
        self.db.insert_dataframe_postgres(df, 'news_items')
        result = self.db.execute_postgres(
            "SELECT currval(pg_get_serial_sequence('news_items', 'news_id')) as news_id"
        )
        return result[0]['news_id'] if result else 0

    def get_news_from_db(
        self, ticker=None, start_date=None, end_date=None, is_processed=None, limit=100
    ) -> List[Dict[str, Any]]:
        if not self.db:
            raise RuntimeError("Database not initialized")
        conditions, params = [], []
        if ticker:
            conditions.append("ticker = %s"); params.append(ticker)
        if start_date:
            conditions.append("published_at >= %s"); params.append(start_date)
        if end_date:
            conditions.append("published_at <= %s"); params.append(end_date)
        if is_processed is not None:
            conditions.append("is_processed = %s"); params.append(is_processed)
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        results = self.db.execute_postgres(
            f"SELECT * FROM news_items WHERE {where_clause} ORDER BY published_at DESC LIMIT %s",
            tuple(params),
        )
        return results or []

    def get_signals_from_db(
        self, ticker=None, start_date=None, end_date=None,
        min_credit_risk_score=None, event_type=None, limit=100,
    ) -> List[Dict[str, Any]]:
        if not self.db:
            raise RuntimeError("Database not initialized")
        conditions, params = [], []
        if ticker:
            conditions.append("ticker = %s"); params.append(ticker)
        if start_date:
            conditions.append("extracted_at >= %s"); params.append(start_date)
        if end_date:
            conditions.append("extracted_at <= %s"); params.append(end_date)
        if min_credit_risk_score is not None:
            conditions.append("credit_risk_score >= %s"); params.append(min_credit_risk_score)
        if event_type:
            conditions.append("event_type = %s"); params.append(event_type)
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)
        results = self.db.execute_postgres(
            f"SELECT * FROM llm_news_signals WHERE {where_clause} ORDER BY extracted_at DESC LIMIT %s",
            tuple(params),
        )
        return results or []

    def compare_with_finbert(self, text: str, llm_signal=None) -> Dict[str, Any]:
        if not self.finbert:
            raise RuntimeError("FinBERT not initialized")
        comparator = FinBERTComparator(self.finbert)
        return comparator.compare_with_llm(text, llm_signal)


news_service = None


def get_news_service():
    global news_service
    if not news_service:
        raise RuntimeError("News service not initialized")
    return news_service


def initialize_news_api(config: Dict[str, Any]):
    global news_service
    news_service = NewsService(config)
    news_service.initialize()


# ---------------------------------------------------------------------------
# Helper: insert a news item via the DI container's database
# ---------------------------------------------------------------------------

def _insert_news_item(db, news_item: NewsItem) -> int:
    import pandas as pd
    df = pd.DataFrame([{
        "ticker": news_item.ticker, "title": news_item.title,
        "body": news_item.body, "source": news_item.source,
        "url": news_item.url, "published_at": news_item.published_at,
        "is_processed": False,
    }])
    from pipelines.ingestion.db_manager import DatabaseManager
    if isinstance(db, DatabaseManager):
        db.insert_dataframe_postgres(df, 'news_items')
        result = db.execute_postgres(
            "SELECT currval(pg_get_serial_sequence('news_items', 'news_id')) as news_id"
        )
        return result[0]['news_id'] if result else 0
    return 0


def _save_signal_to_db(db, signal: LLMNewsSignal, news_id: int) -> bool:
    try:
        import pandas as pd
        signal_data = {
            "news_id": news_id, "ticker": signal.ticker,
            "sentiment_score": signal.sentiment_score,
            "credit_risk_score": signal.credit_risk_score,
            "event_type": signal.event_type,
            "risk_horizon": signal.risk_horizon,
            "market_impact_type": signal.market_impact_type,
            "evidence_spans": signal.evidence_spans,
            "confidence": signal.confidence,
            "extracted_at": signal.extracted_at,
            "llm_model": signal.llm_model,
        }
        df = pd.DataFrame([signal_data])
        if hasattr(db, 'insert_dataframe_postgres'):
            db.insert_dataframe_postgres(df, 'llm_news_signals')
        logger.info(f"Signal saved: news_id={news_id}")
        return True
    except Exception as e:
        logger.error(f"Signal save failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/extract", response_model=NewsExtractResponse, summary="Extract news signal")
@router.post("/news/extract", response_model=NewsExtractResponse, summary="MVP compatibility: extract news signal")
async def extract_news_signal(
    request: NewsExtractRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_db),
    extractor=Depends(get_news_extractor),
):
    """Submit a news article and receive an LLM-extracted credit risk signal."""
    start_time = time.time()

    try:
        news_item = NewsItem(
            news_id="0", ticker=request.ticker,
            title=request.title, body=request.body,
            source=request.source, url=request.url,
            published_at=request.published_at,
        )

        db_news_id = _insert_news_item(db, news_item)

        signal = await extractor.extract_signal(news_item, request.provider)
        processing_time = time.time() - start_time

        if signal:
            background_tasks.add_task(_save_signal_to_db, db, signal, db_news_id)
            return NewsExtractResponse(
                news_id=db_news_id,
                signal={
                    "sentiment_score": signal.sentiment_score,
                    "credit_risk_score": signal.credit_risk_score,
                    "event_type": signal.event_type,
                    "risk_horizon": signal.risk_horizon,
                    "market_impact_type": signal.market_impact_type,
                    "evidence_spans": signal.evidence_spans,
                    "confidence": signal.confidence,
                    "llm_model": signal.llm_model,
                },
                processing_time=processing_time,
            )
        return NewsExtractResponse(
            news_id=db_news_id,
            error="Signal extraction failed",
            processing_time=processing_time,
        )

    except Exception as e:
        logger.error(f"News signal extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-extract", response_model=BatchExtractResponse, summary="Batch extract signals")
@router.post("/news/batch-extract", response_model=BatchExtractResponse, summary="MVP compatibility: batch extract signals")
async def batch_extract_news_signals(
    request: BatchExtractRequest,
    db=Depends(get_db),
    extractor=Depends(get_news_extractor),
):
    """Submit multiple news articles and extract credit risk signals concurrently."""
    try:
        news_items = []
        for item_data in request.news_items:
            news_item = NewsItem(
                news_id="0", ticker=item_data["ticker"],
                title=item_data["title"], body=item_data["body"],
                source=item_data.get("source", ""), url=item_data.get("url", ""),
                published_at=item_data.get("published_at", datetime.now()),
            )
            db_news_id = _insert_news_item(db, news_item)
            news_item.news_id = str(db_news_id)
            news_items.append(news_item)

        signals = await extractor.extract_batch_signals(
            news_items, request.provider, request.max_concurrent
        )

        successful = 0
        for signal in signals:
            try:
                nid = int(signal.news_id)
            except (ValueError, TypeError):
                nid = 0
            if _save_signal_to_db(db, signal, nid):
                successful += 1

        return BatchExtractResponse(
            total=len(news_items),
            successful=successful,
            failed=len(news_items) - successful,
            results=[{
                "news_id": s.news_id, "ticker": s.ticker,
                "credit_risk_score": s.credit_risk_score,
                "event_type": s.event_type,
                "confidence": s.confidence,
            } for s in signals],
        )

    except Exception as e:
        logger.error(f"Batch extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/news", response_model=List[NewsItemResponse], summary="List news")
def get_news(
    ticker: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    is_processed: Optional[bool] = None,
    limit: int = 100,
    db=Depends(get_db),
):
    """Query news items with optional filters by ticker, date range, and processing status."""
    conditions, params = [], []
    if ticker:
        conditions.append("ticker = %s"); params.append(ticker)
    if start_date:
        conditions.append("published_at >= %s"); params.append(start_date)
    if end_date:
        conditions.append("published_at <= %s"); params.append(end_date)
    if is_processed is not None:
        conditions.append("is_processed = %s"); params.append(is_processed)

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"SELECT * FROM news_items WHERE {where} ORDER BY published_at DESC LIMIT %s"
    params.append(limit)

    with db.cursor() as cur:
        cur.execute(sql, params)
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


@router.get("/news/{news_id}", response_model=NewsDetailResponse, summary="News detail")
def get_news_detail(news_id: int, db=Depends(get_db)):
    """Get a single news article with its LLM signal and market reaction data."""
    sql = """
        SELECT ni.*, lns.signal_id, lns.sentiment_score, lns.credit_risk_score,
               lns.event_type, lns.risk_horizon, lns.market_impact_type,
               lns.evidence_spans, lns.confidence, lns.extracted_at, lns.llm_model
        FROM news_items ni
        LEFT JOIN llm_news_signals lns ON ni.news_id = lns.news_id
        WHERE ni.news_id = %s
    """
    with db.cursor() as cur:
        cur.execute(sql, (news_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="News not found")
        columns = [desc[0] for desc in cur.description]
        news = dict(zip(columns, row))

    nid = news.get("news_id")
    signal = None
    if news.get("signal_id"):
        signal = {
            "signal_id": news.get("signal_id"),
            "sentiment_score": float(news["sentiment_score"]) if news.get("sentiment_score") is not None else None,
            "credit_risk_score": news.get("credit_risk_score"),
            "event_type": news.get("event_type"),
            "risk_horizon": news.get("risk_horizon"),
            "market_impact_type": news.get("market_impact_type"),
            "evidence_spans": news.get("evidence_spans"),
            "confidence": float(news["confidence"]) if news.get("confidence") is not None else None,
            "llm_model": news.get("llm_model"),
        }

    return NewsDetailResponse(
        news_id=nid or news_id,
        ticker=news.get("ticker", ""),
        title=news.get("title", ""),
        body=news.get("body", ""),
        source=news.get("source", ""),
        url=news.get("url", ""),
        published_at=news.get("published_at"),
        is_processed=news.get("is_processed", False),
        signal=signal,
    )


@router.get("/signals", response_model=List[SignalResponse], summary="List signals")
def get_signals(
    ticker: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    min_credit_risk_score: Optional[int] = None,
    event_type: Optional[str] = None,
    limit: int = 100,
    db=Depends(get_db),
):
    """Query LLM news signals with optional filters."""
    conditions, params = [], []
    if ticker:
        conditions.append("ticker = %s"); params.append(ticker)
    if start_date:
        conditions.append("extracted_at >= %s"); params.append(start_date)
    if end_date:
        conditions.append("extracted_at <= %s"); params.append(end_date)
    if min_credit_risk_score is not None:
        conditions.append("credit_risk_score >= %s"); params.append(min_credit_risk_score)
    if event_type:
        conditions.append("event_type = %s"); params.append(event_type)

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"SELECT * FROM llm_news_signals WHERE {where} ORDER BY extracted_at DESC LIMIT %s"
    params.append(limit)

    with db.cursor() as cur:
        cur.execute(sql, params)
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


@router.get("/compare-finbert", response_model=CompareFinBERTResponse, summary="Compare with FinBERT")
def compare_with_finbert(ticker: str, news_id: int, db=Depends(get_db)):
    """Compare LLM signal extraction against the FinBERT sentiment baseline for a news article."""
    container = get_container()
    finbert = container.finbert

    with db.cursor() as cur:
        cur.execute("SELECT * FROM news_items WHERE ticker = %s AND news_id = %s", (ticker, news_id))
        cols = [desc[0] for desc in cur.description]
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="News not found")
        target_news = dict(zip(cols, row))

        cur.execute("SELECT * FROM llm_news_signals WHERE news_id = %s", (news_id,))
        cols2 = [desc[0] for desc in cur.description]
        srow = cur.fetchone()
        target_signal = dict(zip(cols2, srow)) if srow else None

    text = f"{target_news['title']} {target_news['body']}"
    comparator = FinBERTComparator(finbert)
    comparison = comparator.compare_with_llm(text, target_signal)

    return CompareFinBERTResponse(news_id=news_id, ticker=ticker, comparison=comparison)


@router.get("/health", summary="News service health")
def health_check():
    """Check the health of the news service (LLM providers, FinBERT, database)."""
    try:
        container = get_container()
        return {
            "status": "healthy",
            "llm_providers": len(container.llm_manager.list_providers()),
            "finbert_initialized": container.finbert.initialized if container.finbert else False,
            "db_connected": container.db.postgres_conn is not None if container.db else False,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
