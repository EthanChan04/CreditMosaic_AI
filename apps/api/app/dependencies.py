"""
Dependency Injection Module
Provides FastAPI dependencies for database, configuration, and service instances.

Replaces the global singleton pattern with proper DI containers
that support testing, configuration isolation, and lifecycle management.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from functools import lru_cache

import yaml

logger = logging.getLogger(__name__)


class AppContainer:
    """Application dependency container.

    Manages the lifecycle of all shared resources:
    - Configuration
    - Database connections
    - LLM providers
    - Service instances
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._db = None
        self._llm_manager = None
        self._finbert = None
        self._initialized = False

    def load_config(self) -> Dict[str, Any]:
        if self.config:
            return self.config

        if not self.config_path:
            self.config_path = str(Path(__file__).resolve().parents[3] / "config.yaml")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self._load_env_file()
        return self.config

    def _load_env_file(self):
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

    def resolve_secret(self, value: str) -> str:
        if value is None:
            return ""
        value = str(value).strip()
        if value.startswith("${") and value.endswith("}"):
            value = os.environ.get(value[2:-1], "").strip()
        placeholder_tokens = ("YOUR_", "TODO", "CHANGE_ME", "PASTE_", "<")
        if not value or any(value.upper().startswith(t) for t in placeholder_tokens):
            return ""
        return value

    @property
    def db(self):
        if self._db is None:
            self._init_db()
        return self._db

    def _init_db(self):
        from pipelines.ingestion.db_manager import DatabaseManager

        cfg = self.config
        postgres_config = dict(cfg.get("postgresql", {}))
        postgres_config.update({
            "host": os.environ.get("POSTGRES_HOST", postgres_config.get("host", "localhost")),
            "port": os.environ.get("POSTGRES_PORT", str(postgres_config.get("port", "5432"))),
            "database": os.environ.get("POSTGRES_DB", postgres_config.get("database", "creditmosaic")),
            "user": os.environ.get("POSTGRES_USER", postgres_config.get("user", "postgres")),
            "password": os.environ.get("POSTGRES_PASSWORD", postgres_config.get("password", "password")),
        })
        duckdb_path = os.environ.get(
            "DUCKDB_PATH",
            cfg.get("duckdb", {}).get("path", "creditmosaic_analytical.db"),
        )
        self._db = DatabaseManager(postgres_config, duckdb_path)
        self._db.connect()
        logger.info("Database connected via DI container")

    @property
    def llm_manager(self):
        if self._llm_manager is None:
            self._init_llm()
        return self._llm_manager

    def _init_llm(self):
        from services.llm_provider import (
            LLMProviderManager, LLMConfig,
            OpenAIProvider, QwenProvider, DeepSeekProvider,
        )

        self._llm_manager = LLMProviderManager()

        for name, cfg in self.config.get("llm_providers", {}).items():
            api_key = self.resolve_secret(cfg.get("api_key", ""))
            if not api_key:
                logger.warning("Skipping LLM provider '%s': no API key configured", name)
                continue

            llm_config = LLMConfig(
                provider=cfg.get("provider", name),
                api_key=api_key,
                model_name=cfg.get("model", ""),
                base_url=cfg.get("base_url", ""),
                max_tokens=cfg.get("max_tokens", 1000),
                temperature=cfg.get("temperature", 0.3),
            )
            provider_cls = {
                "openai": OpenAIProvider,
                "qwen": QwenProvider,
                "deepseek": DeepSeekProvider,
            }.get(llm_config.provider.lower(), OpenAIProvider)
            self._llm_manager.add_provider(name, provider_cls(llm_config))

        default = self.config.get("default_provider")
        if default and default in self._llm_manager.list_providers():
            self._llm_manager.set_default_provider(default)

        logger.info("LLM manager initialized with %d providers", len(self._llm_manager.list_providers()))

    @property
    def finbert(self):
        if self._finbert is None:
            from services.finbert_baseline import FinBERTModel
            model_name = self.config.get("finbert_model", "ProsusAI/finbert")
            self._finbert = FinBERTModel(model_name)
            self._finbert.initialize()
            logger.info("FinBERT model initialized: %s", model_name)
        return self._finbert

    @property
    def news_extractor(self):
        from services.news_signal_extractor import NewsSignalExtractor
        return NewsSignalExtractor(self.llm_manager)

    @property
    def company_service(self):
        from apps.api.app.services.company_service import CompanyService
        return CompanyService(self.db.postgres_conn)

    @property
    def portfolio_service(self):
        from apps.api.app.services.portfolio_service import PortfolioService
        return PortfolioService(self.db.postgres_conn)

    @property
    def report_service(self):
        from apps.api.app.services.report_service import ReportService
        return ReportService(self.db.postgres_conn, self.llm_manager)

    @property
    def risk_model_service(self):
        from services.risk_model_service import RiskModelService
        return RiskModelService(self.db.postgres_conn)

    @property
    def market_reaction_service(self):
        from services.market_reaction_service import MarketReactionService
        return MarketReactionService(self.db.postgres_conn)

    def close(self):
        if self._db:
            self._db.close()
            self._db = None
        self._llm_manager = None
        self._finbert = None
        logger.info("DI container resources released")

    @property
    def is_healthy(self) -> bool:
        try:
            db_ok = self._db is not None and self._db.postgres_conn is not None
            llm_ok = self._llm_manager is not None and len(self._llm_manager.list_providers()) > 0
            return db_ok and llm_ok
        except Exception:
            return False


# Module-level container singleton (initialized once at startup)
_container: Optional[AppContainer] = None


def get_container() -> AppContainer:
    if _container is None:
        raise RuntimeError("AppContainer not initialized. Call init_app() first.")
    return _container


def init_app(config_path: Optional[str] = None) -> AppContainer:
    global _container
    _container = AppContainer(config_path)
    _container.load_config()
    return _container


# FastAPI dependency callables


def get_db():
    """FastAPI dependency: provides raw PostgreSQL connection."""
    container = get_container()
    return container.db.postgres_conn


def get_config() -> Dict[str, Any]:
    """FastAPI dependency: provides application configuration."""
    return get_container().config


def get_llm_manager():
    """FastAPI dependency: provides LLM provider manager."""
    return get_container().llm_manager


def get_company_service():
    """FastAPI dependency: provides CompanyService instance."""
    return get_container().company_service


def get_portfolio_service():
    """FastAPI dependency: provides PortfolioService instance."""
    return get_container().portfolio_service


def get_report_service():
    """FastAPI dependency: provides ReportService instance."""
    return get_container().report_service


def get_risk_model_service():
    """FastAPI dependency: provides RiskModelService instance."""
    return get_container().risk_model_service


def get_market_reaction_service():
    """FastAPI dependency: provides MarketReactionService instance."""
    return get_container().market_reaction_service


def get_news_extractor():
    """FastAPI dependency: provides NewsSignalExtractor instance."""
    return get_container().news_extractor
