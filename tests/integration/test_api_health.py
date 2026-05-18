"""
Integration tests for Health endpoint and middleware.
"""

import pytest


class TestHealthEndpoint:
    """Test GET /health."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "db_connected" in data
        assert "llm_providers" in data
        assert "version" in data

    def test_health_status_healthy(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_db_connected(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["db_connected"] is True

    def test_health_llm_providers(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["llm_providers"] >= 1


class TestMiddleware:
    """Test request timing middleware."""

    def test_process_time_header_present(self, client):
        response = client.get("/health")
        assert "X-Process-Time" in response.headers

    def test_process_time_is_numeric(self, client):
        response = client.get("/health")
        elapsed = float(response.headers["X-Process-Time"])
        assert elapsed >= 0


class TestErrorHandlers:
    """Test global exception handlers."""

    def test_404_for_unknown_route(self, client):
        response = client.get("/api/nonexistent/route")
        assert response.status_code == 404

    def test_422_for_invalid_query_params(self, client):
        """FastAPI returns 422 for validation errors."""
        response = client.get("/api/companies?page=0")
        assert response.status_code == 422

    def test_422_for_invalid_page_size(self, client):
        response = client.get("/api/companies?page_size=500")
        assert response.status_code == 422


class TestOpenAPI:
    """Test OpenAPI schema generation."""

    def test_openapi_schema_available(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "info" in schema

    def test_docs_endpoint(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_required_paths_in_schema(self, client):
        response = client.get("/openapi.json")
        paths = response.json()["paths"]
        required = [
            "/api/companies",
            "/api/companies/sectors",
            "/api/alerts",
            "/health",
        ]
        for path in required:
            assert path in paths, f"Missing path: {path}"
