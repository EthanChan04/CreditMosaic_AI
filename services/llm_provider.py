"""
LLM Provider abstraction layer
Multi-provider support with unified interface for OpenAI, Qwen, DeepSeek
"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    provider: str = ""
    api_key: str = ""
    model_name: str = ""
    base_url: str = ""
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: int = 30


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: Optional[int] = None
    error: Optional[str] = None


class BaseLLMProvider(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    async def generate_completion(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        pass

    @abstractmethod
    async def generate_chat_completion(self, messages: List[Dict]) -> LLMResponse:
        pass


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }

    async def generate_completion(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.generate_chat_completion(messages)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_chat_completion(self, messages: List[Dict]) -> LLMResponse:
        try:
            payload = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.config.timeout
                ) as response:

                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]
                        tokens_used = data.get("usage", {}).get("total_tokens")
                        return LLMResponse(content=content, model=self.config.model_name, tokens_used=tokens_used)
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: {response.status} - {error_text}")
                        return LLMResponse(content="", model=self.config.model_name, error=error_text)

        except Exception as e:
            logger.error(f"OpenAI Provider error: {e}")
            return LLMResponse(content="", model=self.config.model_name, error=str(e))


class QwenProvider(OpenAIProvider):
    """Qwen (Tongyi Qianwen) provider via DashScope OpenAI-compatible API"""

    def __init__(self, config: LLMConfig):
        if not config.base_url:
            config.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        super().__init__(config)


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek provider via OpenAI-compatible API"""

    def __init__(self, config: LLMConfig):
        if not config.base_url:
            config.base_url = "https://api.deepseek.com/v1"
        super().__init__(config)


class LLMProviderFactory:
    @staticmethod
    def create_provider(provider_type: str, config: LLMConfig) -> BaseLLMProvider:
        provider_map = {
            "openai": OpenAIProvider,
            "qwen": QwenProvider,
            "deepseek": DeepSeekProvider,
        }
        key = provider_type.lower()
        if key in provider_map:
            return provider_map[key](config)
        else:
            raise ValueError(f"Unsupported LLM provider type: {provider_type}")


class LLMProviderManager:
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.default_provider: Optional[str] = None

    def add_provider(self, name: str, provider: BaseLLMProvider, set_as_default: bool = False):
        self.providers[name] = provider
        if set_as_default or self.default_provider is None:
            self.default_provider = name

    def set_default_provider(self, name: str):
        if name not in self.providers:
            raise ValueError(f"Provider '{name}' not registered")
        self.default_provider = name

    def get_provider(self, name: str = None) -> Optional[BaseLLMProvider]:
        if name:
            return self.providers.get(name)
        return self.providers.get(self.default_provider)

    def list_providers(self) -> List[str]:
        return list(self.providers.keys())

    async def generate_completion(
        self,
        messages: List[Dict],
        provider_name: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1500
    ) -> LLMResponse:
        provider = self.get_provider(provider_name)
        if not provider:
            raise RuntimeError(f"No LLM provider available (requested: {provider_name})")

        original_temp = provider.config.temperature
        original_max_tokens = provider.config.max_tokens
        provider.config.temperature = temperature
        provider.config.max_tokens = max_tokens

        try:
            return await provider.generate_chat_completion(messages)
        finally:
            provider.config.temperature = original_temp
            provider.config.max_tokens = original_max_tokens


class LLMResponseValidator:
    LLM_NEWS_SIGNAL_SCHEMA = {
        "type": "object",
        "required": ["sentiment_score", "credit_risk_score", "event_type", "risk_horizon",
                     "market_impact_type", "evidence_spans", "confidence"],
        "properties": {
            "sentiment_score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
            "credit_risk_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "event_type": {
                "type": "string",
                "enum": ["liquidity_pressure", "debt_refinancing", "earnings_deterioration",
                         "litigation", "regulatory", "rating_change", "management_change",
                         "supply_chain", "fraud_or_accounting", "neutral_or_irrelevant"]
            },
            "risk_horizon": {"type": "string", "enum": ["1w", "1m", "3m", "12m"]},
            "market_impact_type": {
                "type": "string",
                "enum": ["equity_leading", "credit_leading", "two_market_shock", "low_impact"]
            },
            "evidence_spans": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        }
    }

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown code fences and extract the first JSON object from text."""
        text = text.strip()
        if not text:
            return text

        fence_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        m = re.search(fence_pattern, text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        elif not text.startswith('{') and not text.startswith('['):
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                text = m.group(0)

        return text.strip()

    @staticmethod
    def validate_credit_risk_signal(output: str) -> Optional[Dict[str, Any]]:
        try:
            cleaned = LLMResponseValidator._extract_json(output)
            data = json.loads(cleaned)

            required_fields = LLMResponseValidator.LLM_NEWS_SIGNAL_SCHEMA["required"]
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field: {field}")
                    return None

            sentiment = data["sentiment_score"]
            if not isinstance(sentiment, (int, float)) or sentiment < -1.0 or sentiment > 1.0:
                logger.error(f"sentiment_score out of range [-1.0, 1.0]: {sentiment}")
                return None

            credit_risk = data["credit_risk_score"]
            if not isinstance(credit_risk, int) or credit_risk < 0 or credit_risk > 100:
                logger.error(f"credit_risk_score out of range [0, 100]: {credit_risk}")
                return None

            valid_event_types = LLMResponseValidator.LLM_NEWS_SIGNAL_SCHEMA["properties"]["event_type"]["enum"]
            if data["event_type"] not in valid_event_types:
                logger.error(f"Invalid event_type: {data['event_type']}")
                return None

            valid_horizons = LLMResponseValidator.LLM_NEWS_SIGNAL_SCHEMA["properties"]["risk_horizon"]["enum"]
            if data["risk_horizon"] not in valid_horizons:
                logger.error(f"Invalid risk_horizon: {data['risk_horizon']}")
                return None

            valid_impact_types = LLMResponseValidator.LLM_NEWS_SIGNAL_SCHEMA["properties"]["market_impact_type"]["enum"]
            if data["market_impact_type"] not in valid_impact_types:
                logger.error(f"Invalid market_impact_type: {data['market_impact_type']}")
                return None

            evidence_spans = data["evidence_spans"]
            if not isinstance(evidence_spans, list) or len(evidence_spans) == 0:
                logger.error("evidence_spans must be non-empty array")
                return None

            confidence = data["confidence"]
            if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
                logger.error(f"confidence out of range [0.0, 1.0]: {confidence}")
                return None

            return data

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return None
