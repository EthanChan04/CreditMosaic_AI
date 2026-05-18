"""
Integration tests for Company API endpoints.
"""

import pytest


class TestListCompanies:
    """Test GET /api/companies."""

    def test_list_returns_200(self, client):
        response = client.get("/api/companies")
        assert response.status_code == 200

    def test_list_returns_paginated(self, client):
        response = client.get("/api/companies")
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data

    def test_list_items_have_required_fields(self, client):
        response = client.get("/api/companies")
        items = response.json()["items"]
        assert len(items) > 0
        item = items[0]
        assert "ticker" in item
        assert "company_name" in item

    def test_list_with_sector_filter(self, client):
        response = client.get("/api/companies?sector=Technology")
        assert response.status_code == 200

    def test_list_with_search(self, client):
        response = client.get("/api/companies?search=AAPL")
        assert response.status_code == 200

    def test_list_with_pagination(self, client):
        response = client.get("/api/companies?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        # Mock returns fixed page_size; verify structure is correct
        assert "page_size" in data


class TestListSectors:
    """Test GET /api/companies/sectors."""

    def test_sectors_returns_200(self, client):
        response = client.get("/api/companies/sectors")
        assert response.status_code == 200

    def test_sectors_has_list(self, client):
        response = client.get("/api/companies/sectors")
        data = response.json()
        assert "sectors" in data
        assert isinstance(data["sectors"], list)

    def test_sector_has_required_fields(self, client):
        response = client.get("/api/companies/sectors")
        sectors = response.json()["sectors"]
        if sectors:
            s = sectors[0]
            assert "sector" in s
            assert "company_count" in s


class TestSearchCompanies:
    """Test GET /api/companies/search."""

    def test_search_returns_200(self, client):
        response = client.get("/api/companies/search?q=AAPL")
        assert response.status_code == 200

    def test_search_returns_list(self, client):
        response = client.get("/api/companies/search?q=Apple")
        data = response.json()
        assert isinstance(data, list)


class TestCompanyDetail:
    """Test GET /api/companies/{ticker}."""

    def test_detail_returns_200(self, client):
        response = client.get("/api/companies/AAPL")
        assert response.status_code == 200

    def test_detail_has_enriched_fields(self, client):
        response = client.get("/api/companies/AAPL")
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert "latest_risk_score" in data
        assert "risk_level" in data
        assert "news_count_30d" in data

    def test_detail_404_for_unknown(self, client, mock_company_service):
        from fastapi import HTTPException
        mock_company_service.get_company_detail.side_effect = HTTPException(
            status_code=404, detail="Company not found"
        )
        response = client.get("/api/companies/UNKNOWN")
        assert response.status_code == 404


class TestCompanyNews:
    """Test GET /api/companies/{ticker}/news."""

    def test_news_returns_200(self, client, mock_company_service):
        mock_company_service.get_company_news.return_value = [
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
        response = client.get("/api/companies/AAPL/news")
        assert response.status_code == 200

    def test_news_has_ticker_and_news(self, client, mock_company_service):
        mock_company_service.get_company_news.return_value = [
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
        response = client.get("/api/companies/AAPL/news")
        data = response.json()
        assert "ticker" in data
        assert "news" in data


class TestCompanyRiskHistory:
    """Test GET /api/companies/{ticker}/risk-history."""

    def test_risk_history_returns_200(self, client):
        response = client.get("/api/companies/AAPL/risk-history")
        assert response.status_code == 200

    def test_risk_history_has_ticker(self, client):
        response = client.get("/api/companies/AAPL/risk-history")
        data = response.json()
        assert "ticker" in data


class TestMVPCompatRoutes:
    """Test MVP compatibility routes."""

    def test_company_risk_compat(self, client):
        response = client.get("/api/company/AAPL/risk")
        assert response.status_code == 200

    def test_company_signals_compat(self, client):
        response = client.get("/api/company/AAPL/signals")
        assert response.status_code == 200


class TestCompanyUpsert:
    """Test POST /api/companies."""

    def test_upsert_returns_201(self, client, mock_company_service):
        mock_company_service.upsert_company.return_value = {
            'ticker': 'NEW',
            'company_name': 'New Corp',
            'sector': 'Technology',
            'industry': 'Software',
            'exchange': 'NYSE',
            'market_cap': 1000000000,
            'country': 'US',
            'founded_year': 2020,
            'created_at': '2026-01-01T00:00:00',
            'updated_at': '2026-01-01T00:00:00',
        }
        response = client.post("/api/companies", json={
            "ticker": "NEW",
            "company_name": "New Corp",
            "sector": "Technology",
            "industry": "Software",
            "exchange": "NYSE",
            "market_cap": 1000000000,
            "country": "US",
            "founded_year": 2020,
        })
        assert response.status_code == 201


class TestCompanyDelete:
    """Test DELETE /api/companies/{ticker}."""

    def test_delete_returns_200(self, client, mock_company_service):
        mock_company_service.delete_company.return_value = True
        response = client.delete("/api/companies/AAPL")
        assert response.status_code == 200
