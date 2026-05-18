"""
Shared fixtures for API integration tests.
Creates a FastAPI TestClient with mocked dependencies.
"""

import sys
import contextlib
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from contextlib import ExitStack
from datetime import datetime, date

from fastapi.testclient import TestClient


def _ensure_torch_mock():
    """Mock torch/transformers if not installed, for app import only."""
    if 'torch' not in sys.modules:
        torch_mock = MagicMock()
        sys.modules['torch'] = torch_mock
        sys.modules['torch.nn'] = torch_mock.nn
    if 'transformers' not in sys.modules:
        sys.modules['transformers'] = MagicMock()


@contextlib.asynccontextmanager
async def _noop_lifespan(app):
    """No-op lifespan context manager for testing."""
    yield


@pytest.fixture
def mock_db():
    """Mock database connection for API tests."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


@pytest.fixture
def mock_company_service():
    """Mock CompanyService."""
    svc = MagicMock()
    svc.list_companies.return_value = {
        'items': [
            {
                'ticker': 'AAPL',
                'company_name': 'Apple Inc.',
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'exchange': 'NASDAQ',
                'market_cap': 2800000000000,
                'country': 'US',
                'founded_year': 1976,
                'created_at': '2026-01-01T00:00:00',
                'updated_at': '2026-01-01T00:00:00',
            },
        ],
        'total': 1,
        'page': 1,
        'page_size': 20,
        'total_pages': 1,
    }
    svc.get_companies_by_sector.return_value = [
        {'sector': 'Technology', 'company_count': 3, 'avg_market_cap': 2000000000000},
    ]
    svc.search_companies.return_value = [
        {
            'ticker': 'AAPL',
            'company_name': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'exchange': 'NASDAQ',
            'market_cap': 2800000000000,
            'country': 'US',
            'founded_year': 1976,
            'created_at': '2026-01-01T00:00:00',
            'updated_at': '2026-01-01T00:00:00',
        },
    ]
    svc.get_company_detail.return_value = {
        'ticker': 'AAPL',
        'company_name': 'Apple Inc.',
        'sector': 'Technology',
        'industry': 'Consumer Electronics',
        'exchange': 'NASDAQ',
        'market_cap': 2800000000000,
        'country': 'US',
        'founded_year': 1976,
        'created_at': '2026-01-01T00:00:00',
        'updated_at': '2026-01-01T00:00:00',
        'latest_risk_score': 0.35,
        'risk_level': 'Medium',
        'news_count_30d': 15,
        'high_risk_news_count_30d': 2,
        'latest_price': 185.50,
        'price_change_5d': -0.02,
    }
    svc.get_company_news.return_value = [
        {
            'news_id': 1,
            'title': 'Apple reports earnings',
            'source': 'Reuters',
            'published_at': '2026-04-28T10:00:00',
            'sentiment_score': 0.3,
            'credit_risk_score': 25,
            'event_type': 'neutral_or_irrelevant',
        },
    ]
    return svc


@pytest.fixture
def mock_risk_service():
    """Mock RiskModelService."""
    svc = MagicMock()
    svc.generate_risk_labels.return_value = MagicMock(shape=(100, 12))
    svc.train_models.return_value = {
        'logistic_regression': {'auc': 0.72, 'f1': 0.45},
        'lightgbm': {'auc': 0.81, 'f1': 0.52},
        'xgboost': {'auc': 0.79, 'f1': 0.50},
    }
    svc.score_companies.return_value = MagicMock(shape=(3, 4))
    svc.get_risk_summary.return_value = [
        {
            'ticker': 'AAPL',
            'date': date(2026, 4, 30),
            'risk_score': 0.35,
            'risk_level': 'Medium',
            'model_version': 'lightgbm',
            'top_features': [{'feature': 'volatility_5d_lag6', 'importance': 0.15}],
        },
    ]
    svc.get_company_risk_history.return_value = [
        {
            'ticker': 'AAPL',
            'date': date(2026, 4, 30),
            'risk_score': 0.35,
            'risk_level': 'Medium',
            'model_version': 'lightgbm',
            'top_features': [],
        },
    ]
    svc.get_model_evaluation.return_value = {
        'timestamp': '2026-04-30T00:00:00',
        'feature_count': 45,
        'models': {
            'lightgbm': {'metrics': {'auc': 0.81}},
        },
    }
    return svc


@pytest.fixture
def mock_report_service():
    """Mock ReportService."""
    svc = MagicMock()
    svc.generate_report.return_value = {
        'report_id': 1,
        'ticker': 'AAPL',
        'report_type': 'company_risk',
        'title': 'Risk Report - AAPL',
        'markdown_content': '# Risk Report\n\nSummary...',
        'summary': {'risk_level': 'Medium', 'risk_score': 0.35},
        'model_used': 'longcat',
        'generated_at': '2026-04-30T00:00:00',
    }
    svc.list_reports.return_value = [
        {
            'report_id': 1,
            'ticker': 'AAPL',
            'report_type': 'company_risk',
            'title': 'Risk Report - AAPL',
            'generated_at': '2026-04-30T00:00:00',
        },
    ]
    svc.get_report.return_value = {
        'report_id': 1,
        'ticker': 'AAPL',
        'report_type': 'company_risk',
        'title': 'Risk Report - AAPL',
        'markdown_content': '# Risk Report\n\nSummary...',
        'summary': {'risk_level': 'Medium'},
        'model_used': 'longcat',
        'generated_at': '2026-04-30T00:00:00',
    }
    svc.get_company_latest_report.return_value = {
        'report_id': 1,
        'ticker': 'AAPL',
        'report_type': 'company_risk',
        'title': 'Risk Report - AAPL',
        'markdown_content': '# Risk Report\n\nSummary...',
        'summary': {'risk_level': 'Medium'},
        'model_used': 'longcat',
        'generated_at': '2026-04-30T00:00:00',
    }
    return svc


@pytest.fixture
def mock_portfolio_service():
    """Mock PortfolioService."""
    svc = MagicMock()
    svc.analyze_portfolio.return_value = {
        'total_risk_score': 0.45,
        'risk_level': 'Medium',
        'holdings_risk': [
            {'ticker': 'AAPL', 'weight': 0.5, 'risk_score': 0.35, 'risk_level': 'Medium'},
        ],
        'top_contributors': [{'ticker': 'AAPL', 'contribution': 0.175}],
        'diversification_score': 0.72,
        'recommendation': 'Portfolio risk is moderate.',
    }
    svc.list_portfolios.return_value = []
    svc.analyze_with_correlations.return_value = {
        'correlation_matrix': {},
        'portfolio_volatility': 0.02,
    }
    svc.stress_test.return_value = {'scenarios': []}
    svc.generate_portfolio_report_markdown.return_value = '# Portfolio Report\n\n...'
    return svc


@pytest.fixture
def mock_market_reaction_service():
    """Mock MarketReactionService."""
    svc = MagicMock()
    return svc


@pytest.fixture
def mock_llm_manager():
    """Mock LLM provider manager."""
    mgr = MagicMock()
    mgr.list_providers.return_value = ['longcat']
    return mgr


@pytest.fixture
def mock_news_extractor():
    """Mock NewsSignalExtractor."""
    extractor = MagicMock()
    extractor.extract_signal = MagicMock()
    return extractor


@pytest.fixture
def mock_container(mock_db, mock_company_service, mock_risk_service, mock_llm_manager,
                   mock_report_service, mock_portfolio_service, mock_market_reaction_service):
    """Mock the full AppContainer."""
    container = MagicMock()
    # Set up db with postgres_conn attribute
    db_mock = MagicMock()
    db_mock.postgres_conn = mock_db
    container.db = db_mock
    container.company_service = mock_company_service
    container.risk_model_service = mock_risk_service
    container.llm_manager = mock_llm_manager
    container.llm_manager.list_providers.return_value = ['longcat']
    container.finbert = MagicMock()
    container.finbert.initialized = True
    container.is_healthy = True
    container.report_service = mock_report_service
    container.portfolio_service = mock_portfolio_service
    container.market_reaction_service = mock_market_reaction_service
    return container


@pytest.fixture
def client(mock_container, mock_db, mock_company_service, mock_risk_service,
           mock_llm_manager, mock_news_extractor, mock_report_service,
           mock_portfolio_service, mock_market_reaction_service):
    """Create a FastAPI TestClient with all dependencies mocked."""
    _ensure_torch_mock()
    from apps.api.app.main import app
    from apps.api.app import dependencies
    from apps.api.app.api import company, news, risk, reaction, portfolio, report, alerts

    patches = [
        patch('apps.api.app.dependencies.get_container', return_value=mock_container),
        patch('apps.api.app.main.get_container', return_value=mock_container),
        patch.object(dependencies, 'get_db', return_value=mock_db),
        patch.object(dependencies, 'get_company_service', return_value=mock_company_service),
        patch.object(dependencies, 'get_risk_model_service', return_value=mock_risk_service),
        patch.object(dependencies, 'get_llm_manager', return_value=mock_llm_manager),
        patch.object(dependencies, 'get_news_extractor', return_value=mock_news_extractor),
        patch.object(dependencies, 'get_portfolio_service', return_value=mock_portfolio_service),
        patch.object(dependencies, 'get_report_service', return_value=mock_report_service),
        patch.object(dependencies, 'get_market_reaction_service', return_value=mock_market_reaction_service),
        patch.object(dependencies, 'get_config', return_value={'default_provider': 'longcat'}),
        patch.object(company, 'get_company_service', return_value=mock_company_service),
        patch.object(company, 'get_risk_model_service', return_value=mock_risk_service),
        patch.object(company, 'get_db', return_value=mock_db),
        patch.object(news, 'get_db', return_value=mock_db),
        patch.object(risk, 'get_risk_model_service', return_value=mock_risk_service),
        patch.object(risk, 'get_db', return_value=mock_db),
        patch.object(reaction, 'get_market_reaction_service', return_value=mock_market_reaction_service),
        patch.object(reaction, 'get_db', return_value=mock_db),
        patch.object(portfolio, 'get_portfolio_service', return_value=mock_portfolio_service),
        patch.object(portfolio, 'get_risk_model_service', return_value=mock_risk_service),
        patch.object(report, 'get_report_service', return_value=mock_report_service),
        patch.object(alerts, 'get_db', return_value=mock_db),
    ]

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)

        app.router.lifespan_context = _noop_lifespan
        with TestClient(app) as c:
            yield c
