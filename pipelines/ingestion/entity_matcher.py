"""
公司实体匹配服务
将新闻中的公司名称映射到标准ticker
"""

import re
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
import yfinance as yf

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EntityMatch:
    """实体匹配结果"""
    ticker: str
    company_name: str
    confidence: float
    match_method: str
    source_text: str

class EntityMatcher:
    """实体匹配器"""

    def __init__(self, company_list: List[Dict]):
        """
        初始化实体匹配器

        Args:
            company_list: 公司列表，每个公司包含ticker, company_name, sector等信息
        """
        self.companies = company_list
        self.name_to_ticker = {comp['company_name'].lower(): comp['ticker'] for comp in company_list}
        self.ticker_to_company = {comp['ticker']: comp for comp in company_list}

        # 构建名称变体映射
        self.name_variants = self._build_name_variants()

    def _build_name_variants(self) -> Dict[str, str]:
        """构建公司名称变体映射"""
        variants = {}

        for company in self.companies:
            ticker = company['ticker']
            name = company['company_name']

            # 添加各种变体
            variants[name.lower()] = ticker
            variants[ticker.lower()] = ticker

            # 去除公司后缀的变体
            name_no_suffix = re.sub(r'\s+(Inc|Corp|Ltd|LLC|PLC)\.?$', '', name, flags=re.IGNORECASE)
            if name_no_suffix.lower() != name.lower():
                variants[name_no_suffix.lower()] = ticker

            # 常见简称
            if ',' in name:
                short_name = name.split(',')[0]
                variants[short_name.lower()] = ticker

        return variants

    def match_from_text(self, text: str, min_confidence: float = 0.6) -> List[EntityMatch]:
        """
        从文本中提取并匹配公司实体

        Args:
            text: 输入文本
            min_confidence: 最小置信度阈值

        Returns:
            匹配到的公司实体列表
        """
        matches = []

        # 1. 精确匹配（ticker和完整公司名）
        exact_matches = self._find_exact_matches(text)
        matches.extend(exact_matches)

        # 2. 模糊匹配（处理拼写变体）
        fuzzy_matches = self._find_fuzzy_matches(text, min_confidence)
        matches.extend(fuzzy_matches)

        # 3. 基于上下文的匹配
        context_matches = self._find_context_matches(text, min_confidence)
        matches.extend(context_matches)

        # 去重和排序
        unique_matches = self._deduplicate_matches(matches)
        sorted_matches = sorted(unique_matches, key=lambda x: x.confidence, reverse=True)

        return sorted_matches

    def _find_exact_matches(self, text: str) -> List[EntityMatch]:
        """Find exact matches using word-boundary-aware matching for short tickers."""
        matches = []
        text_lower = text.lower()

        for variant, ticker in self.name_variants.items():
            if len(variant) <= 3 and variant == ticker.lower():
                if not re.search(r'\b' + re.escape(variant) + r'\b', text_lower):
                    continue
            elif variant not in text_lower:
                continue

            company = self.ticker_to_company[ticker]
            start_pos = text_lower.find(variant)
            match_text = text[start_pos:start_pos + len(variant)]

            match = EntityMatch(
                ticker=ticker,
                company_name=company['company_name'],
                confidence=1.0,
                match_method='exact',
                source_text=match_text
            )
            matches.append(match)

        return matches

    def _find_fuzzy_matches(self, text: str, min_confidence: float) -> List[EntityMatch]:
        """查找模糊匹配"""
        matches = []
        words = re.findall(r'\b\w+\b', text)

        for company in self.companies:
            company_name = company['company_name']
            ticker = company['ticker']

            # 检查公司名称的各个部分
            name_parts = company_name.split()

            for i, word in enumerate(words):
                best_ratio = 0
                best_part = ""

                for part in name_parts:
                    ratio = SequenceMatcher(None, word.lower(), part.lower()).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_part = part

                if best_ratio >= min_confidence:
                    # 检查是否是连续匹配
                    if i + len(name_parts) <= len(words):
                        potential_match = " ".join(words[i:i + len(name_parts)])
                        full_ratio = SequenceMatcher(
                            None,
                            potential_match.lower(),
                            company_name.lower()
                        ).ratio()

                        if full_ratio >= min_confidence:
                            match = EntityMatch(
                                ticker=ticker,
                                company_name=company_name,
                                confidence=full_ratio,
                                match_method='fuzzy',
                                source_text=potential_match
                            )
                            matches.append(match)

        return matches

    def _find_context_matches(self, text: str, min_confidence: float) -> List[EntityMatch]:
        """基于上下文的匹配"""
        matches = []

        # 查找行业关键词
        industry_patterns = {
            'technology': ['tech', 'software', 'hardware', 'semiconductor', 'internet'],
            'finance': ['bank', 'financial', 'investment', 'insurance'],
            'healthcare': ['pharma', 'biotech', 'health', 'medical'],
            'energy': ['oil', 'gas', 'energy', 'renewable'],
            'consumer': ['retail', 'consumer', 'food', 'beverage']
        }

        # 查找地理位置
        location_keywords = ['inc', 'corp', 'ltd', 'llc', 'usa', 'us', 'america', 'international']

        for company in self.companies:
            score = 0
            matched_keywords = []

            # 行业匹配
            if company.get('industry'):
                industry = company['industry'].lower()
                for ind_keyword in industry_patterns:
                    if ind_keyword in industry:
                        matched_keywords.append(ind_keyword)
                        score += 0.3

            # ticker附近关键词匹配
            ticker_lower = company['ticker'].lower()
            if ticker_lower in text.lower():
                # 检查ticker周围的上下文
                ticker_pos = text.lower().find(ticker_lower)
                context_window = text[max(0, ticker_pos - 20):min(len(text), ticker_pos + len(ticker_lower) + 20)]

                for keyword in location_keywords:
                    if keyword in context_window.lower():
                        score += 0.2
                        matched_keywords.append(keyword)

            if score >= min_confidence:
                match = EntityMatch(
                    ticker=company['ticker'],
                    company_name=company['company_name'],
                    confidence=score,
                    match_method='context',
                    source_text=','.join(matched_keywords)
                )
                matches.append(match)

        return matches

    def _deduplicate_matches(self, matches: List[EntityMatch]) -> List[EntityMatch]:
        """去重匹配结果"""
        seen_tickers = set()
        unique_matches = []

        for match in matches:
            if match.ticker not in seen_tickers:
                seen_tickers.add(match.ticker)
                unique_matches.append(match)

        return unique_matches

    def validate_ticker(self, ticker: str) -> bool:
        """验证ticker是否有效"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return 'symbol' in info
        except:
            return False

    def get_company_info(self, ticker: str) -> Optional[Dict]:
        """获取公司详细信息"""
        return self.ticker_to_company.get(ticker)

class TickerListManager:
    """Ticker列表管理器"""

    @staticmethod
    def create_default_ticker_list() -> List[Dict]:
        """创建默认ticker列表（美股主要公司）"""
        return [
            {'ticker': 'AAPL', 'company_name': 'Apple Inc.', 'sector': 'Technology'},
            {'ticker': 'MSFT', 'company_name': 'Microsoft Corporation', 'sector': 'Technology'},
            {'ticker': 'GOOGL', 'company_name': 'Alphabet Inc.', 'sector': 'Technology'},
            {'ticker': 'AMZN', 'company_name': 'Amazon.com Inc.', 'sector': 'Consumer Cyclical'},
            {'ticker': 'TSLA', 'company_name': 'Tesla Inc.', 'sector': 'Consumer Cyclical'},
            {'ticker': 'NVDA', 'company_name': 'NVIDIA Corporation', 'sector': 'Technology'},
            {'ticker': 'META', 'company_name': 'Meta Platforms Inc.', 'sector': 'Technology'},
            {'ticker': 'JPM', 'company_name': 'JPMorgan Chase & Co.', 'sector': 'Financial Services'},
            {'ticker': 'JNJ', 'company_name': 'Johnson & Johnson', 'sector': 'Healthcare'},
            {'ticker': 'V', 'company_name': 'Visa Inc.', 'sector': 'Financial Services'},
            {'ticker': 'PG', 'company_name': 'Procter & Gamble Company', 'sector': 'Consumer Defensive'},
            {'ticker': 'UNH', 'company_name': 'UnitedHealth Group Inc.', 'sector': 'Healthcare'},
            {'ticker': 'HD', 'company_name': 'Home Depot Inc.', 'sector': 'Consumer Cyclical'},
            {'ticker': 'MA', 'company_name': 'Mastercard Inc.', 'sector': 'Financial Services'},
            {'ticker': 'DIS', 'company_name': 'Walt Disney Company', 'sector': 'Communication Services'},
            {'ticker': 'BAC', 'company_name': 'Bank of America Corp', 'sector': 'Financial Services'},
            {'ticker': 'XOM', 'company_name': 'Exxon Mobil Corp', 'sector': 'Energy'},
            {'ticker': 'WMT', 'company_name': 'Walmart Inc.', 'sector': 'Consumer Defensive'},
            {'ticker': 'NFLX', 'company_name': 'Netflix Inc.', 'sector': 'Communication Services'},
            {'ticker': 'PFE', 'company_name': 'Pfizer Inc.', 'sector': 'Healthcare'},
            {'ticker': 'ADBE', 'company_name': 'Adobe Inc.', 'sector': 'Technology'},
            {'ticker': 'CRM', 'company_name': 'Salesforce Inc.', 'sector': 'Technology'},
            {'ticker': 'AVGO', 'company_name': 'Broadcom Inc.', 'sector': 'Technology'},
            {'ticker': 'CSCO', 'company_name': 'Cisco Systems Inc.', 'sector': 'Technology'},
            {'ticker': 'COST', 'company_name': 'Costco Wholesale Corp', 'sector': 'Consumer Defensive'},
            {'ticker': 'TMUS', 'company_name': 'T-Mobile US Inc.', 'sector': 'Communication Services'},
            {'ticker': 'ABT', 'company_name': 'Abbott Laboratories', 'sector': 'Healthcare'},
            {'ticker': 'INTC', 'company_name': 'Intel Corporation', 'sector': 'Technology'},
            {'ticker': 'WFC', 'company_name': 'Wells Fargo & Company', 'sector': 'Financial Services'},
            {'ticker': 'INTU', 'company_name': 'Intuit Inc.', 'sector': 'Technology'},
            {'ticker': 'VZ', 'company_name': 'Verizon Communications Inc.', 'sector': 'Communication Services'},
            {'ticker': 'T', 'company_name': 'AT&T Inc.', 'sector': 'Communication Services'},
            {'ticker': 'QCOM', 'company_name': 'Qualcomm Incorporated', 'sector': 'Technology'},
            {'ticker': 'TXN', 'company_name': 'Texas Instruments Incorporated', 'sector': 'Technology'},
            {'ticker': 'AMD', 'company_name': 'Advanced Micro Devices Inc.', 'sector': 'Technology'},
            {'ticker': 'LIN', 'company_name': 'Linde plc', 'sector': 'Basic Materials'},
            {'ticker': 'ORCL', 'company_name': 'Oracle Corporation', 'sector': 'Technology'},
            {'ticker': 'ACN', 'company_name': 'Accenture plc', 'sector': 'Technology'},
            {'ticker': 'IBM', 'company_name': 'International Business Machines Corp', 'sector': 'Technology'},
            {'ticker': 'GE', 'company_name': 'General Electric Company', 'sector': 'Industrials'},
            {'ticker': 'CAT', 'company_name': 'Caterpillar Inc.', 'sector': 'Industrials'},
            {'ticker': 'BA', 'company_name': 'The Boeing Company', 'sector': 'Industrials'},
            {'ticker': 'LMT', 'company_name': 'Lockheed Martin Corporation', 'sector': 'Industrials'},
            {'ticker': 'RTX', 'company_name': 'RTX Corporation', 'sector': 'Industrials'},
            {'ticker': 'UNP', 'company_name': 'Union Pacific Corporation', 'sector': 'Industrials'},
            {'ticker': 'UPS', 'company_name': 'United Parcel Service Inc.', 'sector': 'Industrials'},
            {'ticker': 'LOW', 'company_name': "Lowe's Companies Inc.", 'sector': 'Consumer Cyclical'},
            {'ticker': 'SBUX', 'company_name': 'Starbucks Corporation', 'sector': 'Consumer Cyclical'},
            {'ticker': 'MDT', 'company_name': 'Medtronic plc', 'sector': 'Healthcare'},
            {'ticker': 'TMO', 'company_name': 'Thermo Fisher Scientific Inc.', 'sector': 'Healthcare'},
        ]

    @staticmethod
    def save_ticker_list(companies: List[Dict], filepath: str):
        """保存ticker列表到CSV"""
        df = pd.DataFrame(companies)
        df.to_csv(filepath, index=False)
        logger.info(f"保存 {len(companies)} 家公司到 {filepath}")

    @staticmethod
    def load_ticker_list(filepath: str) -> List[Dict]:
        """从CSV加载ticker列表"""
        df = pd.read_csv(filepath)
        return df.to_dict('records')

# 使用示例
if __name__ == "__main__":
    # 创建ticker列表
    companies = TickerListManager.create_default_ticker_list()

    # 创建实体匹配器
    matcher = EntityMatcher(companies)

    # 测试文本
    test_texts = [
        "Apple Inc. announced its earnings today, beating analyst expectations.",
        "The stock of AAPL is trading higher after the announcement.",
        "Microsoft Corp and Google parent Alphabet reported strong cloud growth.",
        "Tesla's Q4 results disappointed investors, sending TSLA shares down.",
        "JPMorgan Chase is facing regulatory challenges in the US market."
    ]

    # 测试匹配
    for text in test_texts:
        print(f"\n文本: {text}")
        matches = matcher.match_from_text(text)
        for match in matches:
            print(f"  匹配: {match.ticker} ({match.company_name}) - 置信度: {match.confidence:.2f} - 方法: {match.match_method}")

    # 验证ticker
    print(f"\n验证ticker:")
    for ticker in ['AAPL', 'INVALID', 'TSLA']:
        is_valid = matcher.validate_ticker(ticker)
        print(f"  {ticker}: {'有效' if is_valid else '无效'}")