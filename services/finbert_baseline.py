"""
FinBERT基线情绪分析模型
提供与传统金融NLP模型的对比基准
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy.special import softmax

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FinBERTResult:
    """FinBERT分析结果"""
    text: str
    positive_score: float
    negative_score: float
    neutral_score: float
    sentiment_label: str
    sentiment_score: float  # -1到1
    confidence: float

class FinBERTModel:
    """FinBERT模型封装"""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        初始化FinBERT模型

        Args:
            model_name: 模型名称或路径
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.initialized = False

        logger.info(f"FinBERT使用设备: {self.device}")

    def initialize(self) -> bool:
        """
        初始化模型和分词器

        Returns:
            初始化是否成功
        """
        try:
            logger.info(f"正在加载FinBERT模型: {self.model_name}")

            # 加载分词器和模型
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            # 移动到设备
            self.model.to(self.device)
            self.model.eval()

            self.initialized = True
            logger.info("FinBERT模型加载成功")
            return True

        except Exception as e:
            logger.error(f"FinBERT模型加载失败: {e}")
            return False

    def analyze_sentiment(self, text: str, max_length: int = 512) -> FinBERTResult:
        """
        分析文本情绪

        Args:
            text: 输入文本
            max_length: 最大长度

        Returns:
            FinBERTResult对象
        """
        if not self.initialized:
            logger.warning("FinBERT未初始化，返回默认值")
            return FinBERTResult(
                text=text,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0,
                sentiment_label="neutral",
                sentiment_score=0.0,
                confidence=1.0
            )

        try:
            # 编码输入
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(self.device)

            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.cpu().numpy()[0]
                probabilities = softmax(logits)

            # FinBERT输出: [positive, negative, neutral]
            positive_score = float(probabilities[0])
            negative_score = float(probabilities[1])
            neutral_score = float(probabilities[2])

            # 确定情绪标签
            sentiment_label = ["positive", "negative", "neutral"][np.argmax(probabilities)]

            # 计算情绪分数 (-1到1)
            # positive为正，negative为负，neutral为0
            if sentiment_label == "positive":
                sentiment_score = positive_score
            elif sentiment_label == "negative":
                sentiment_score = -negative_score
            else:
                sentiment_score = 0.0

            # 计算置信度
            confidence = float(np.max(probabilities))

            result = FinBERTResult(
                text=text,
                positive_score=positive_score,
                negative_score=negative_score,
                neutral_score=neutral_score,
                sentiment_label=sentiment_label,
                sentiment_score=sentiment_score,
                confidence=confidence
            )

            logger.debug(f"FinBERT分析结果: {sentiment_label} | 置信度: {confidence:.3f}")
            return result

        except Exception as e:
            logger.error(f"FinBERT情绪分析失败: {e}")
            return FinBERTResult(
                text=text,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0,
                sentiment_label="neutral",
                sentiment_score=0.0,
                confidence=1.0
            )

    def analyze_batch(self, texts: List[str], batch_size: int = 8) -> List[FinBERTResult]:
        """
        批量分析文本情绪

        Args:
            texts: 文本列表
            batch_size: 批次大小

        Returns:
            FinBERTResult列表
        """
        if not self.initialized:
            logger.warning("FinBERT未初始化，返回默认值")
            return [
                FinBERTResult(
                    text=text,
                    positive_score=0.0,
                    negative_score=0.0,
                    neutral_score=1.0,
                    sentiment_label="neutral",
                    sentiment_score=0.0,
                    confidence=1.0
                )
                for text in texts
            ]

        results = []

        try:
            # 分批处理
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                logger.info(f"处理批次: {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

                # 编码批次
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)

                # 推理
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits.cpu().numpy()
                    probabilities = softmax(logits, axis=1)

                # 处理每个结果
                for j, probs in enumerate(probabilities):
                    positive_score = float(probs[0])
                    negative_score = float(probs[1])
                    neutral_score = float(probs[2])

                    sentiment_label = ["positive", "negative", "neutral"][np.argmax(probs)]

                    if sentiment_label == "positive":
                        sentiment_score = positive_score
                    elif sentiment_label == "negative":
                        sentiment_score = -negative_score
                    else:
                        sentiment_score = 0.0

                    confidence = float(np.max(probs))

                    result = FinBERTResult(
                        text=batch_texts[j],
                        positive_score=positive_score,
                        negative_score=negative_score,
                        neutral_score=neutral_score,
                        sentiment_label=sentiment_label,
                        sentiment_score=sentiment_score,
                        confidence=confidence
                    )
                    results.append(result)

        except Exception as e:
            logger.error(f"FinBERT批量分析失败: {e}")
            # 返回部分结果
            return results

        return results

    def get_sentiment_features(self, texts: List[str]) -> Dict[str, List[float]]:
        """
        获取用于模型训练的情绪特征

        Args:
            texts: 文本列表

        Returns:
            特征字典
        """
        results = self.analyze_batch(texts)

        features = {
            "finbert_positive": [r.positive_score for r in results],
            "finbert_negative": [r.negative_score for r in results],
            "finbert_neutral": [r.neutral_score for r in results],
            "finbert_sentiment_score": [r.sentiment_score for r in results],
            "finbert_confidence": [r.confidence for r in results],
            "finbert_is_positive": [1 if r.sentiment_label == "positive" else 0 for r in results],
            "finbert_is_negative": [1 if r.sentiment_label == "negative" else 0 for r in results],
            "finbert_is_neutral": [1 if r.sentiment_label == "neutral" else 0 for r in results]
        }

        return features

class FinBERTComparator:
    """FinBERT对比分析器"""

    def __init__(self, finbert_model: FinBERTModel, llm_extractor=None):
        self.finbert = finbert_model
        self.llm_extractor = llm_extractor

    def compare_with_llm(self, text: str, llm_signal=None) -> Dict[str, Any]:
        """
        对比FinBERT和LLM分析结果

        Args:
            text: 文本
            llm_signal: LLM信号（如果为None，则使用LLM提取器）

        Returns:
            对比结果
        """
        # FinBERT分析
        finbert_result = self.finbert.analyze_sentiment(text)

        # LLM分析
        if llm_signal is None and self.llm_extractor:
            # 这里需要异步调用，简化处理
            logger.warning("需要异步调用LLM提取器")
            llm_signal = None

        comparison = {
            "finbert": {
                "sentiment_score": finbert_result.sentiment_score,
                "sentiment_label": finbert_result.sentiment_label,
                "confidence": finbert_result.confidence,
                "probabilities": {
                    "positive": finbert_result.positive_score,
                    "negative": finbert_result.negative_score,
                    "neutral": finbert_result.neutral_score
                }
            }
        }

        if llm_signal:
            comparison["llm"] = {
                "sentiment_score": llm_signal.sentiment_score,
                "credit_risk_score": llm_signal.credit_risk_score,
                "event_type": llm_signal.event_type,
                "confidence": llm_signal.confidence
            }

            # 对比分析
            comparison["comparison"] = {
                "sentiment_difference": abs(finbert_result.sentiment_score - llm_signal.sentiment_score),
                "sentiment_agreement": self._calculate_sentiment_agreement(
                    finbert_result.sentiment_score, llm_signal.sentiment_score
                ),
                "confidence_comparison": {
                    "finbert": finbert_result.confidence,
                    "llm": llm_signal.confidence,
                    "higher": "finbert" if finbert_result.confidence > llm_signal.confidence else "llm"
                }
            }

        return comparison

    def _calculate_sentiment_agreement(self, score1: float, score2: float) -> str:
        """计算情绪一致性"""
        diff = abs(score1 - score2)
        if diff < 0.2:
            return "high"
        elif diff < 0.4:
            return "medium"
        else:
            return "low"

    def batch_compare(self, texts: List[str], llm_signals: List = None) -> List[Dict[str, Any]]:
        """
        批量对比分析

        Args:
            texts: 文本列表
            llm_signals: LLM信号列表

        Returns:
            对比结果列表
        """
        finbert_results = self.finbert.analyze_batch(texts)
        comparisons = []

        for i, finbert_result in enumerate(finbert_results):
            llm_signal = llm_signals[i] if llm_signals else None

            comparison = {
                "index": i,
                "text": finbert_result.text,
                "finbert": {
                    "sentiment_score": finbert_result.sentiment_score,
                    "sentiment_label": finbert_result.sentiment_label,
                    "confidence": finbert_result.confidence
                }
            }

            if llm_signal:
                comparison["llm"] = {
                    "sentiment_score": llm_signal.sentiment_score,
                    "credit_risk_score": llm_signal.credit_risk_score,
                    "confidence": llm_signal.confidence
                }

                comparison["comparison"] = {
                    "sentiment_difference": abs(finbert_result.sentiment_score - llm_signal.sentiment_score),
                    "sentiment_agreement": self._calculate_sentiment_agreement(
                        finbert_result.sentiment_score, llm_signal.sentiment_score
                    )
                }

            comparisons.append(comparison)

        return comparisons

    def generate_comparison_report(self, comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成对比报告

        Args:
            comparisons: 对比结果列表

        Returns:
            报告数据
        """
        if not comparisons:
            return {"error": "没有对比数据"}

        # 计算统计指标
        total = len(comparisons)
        high_agreement = len([c for c in comparisons if c.get("comparison", {}).get("sentiment_agreement") == "high"])
        medium_agreement = len([c for c in comparisons if c.get("comparison", {}).get("sentiment_agreement") == "medium"])
        low_agreement = len([c for c in comparisons if c.get("comparison", {}).get("sentiment_agreement") == "low"])

        sentiment_diffs = [c.get("comparison", {}).get("sentiment_difference", 0) for c in comparisons]
        avg_sentiment_diff = sum(sentiment_diffs) / len(sentiment_diffs) if sentiment_diffs else 0

        # FinBERT情绪分布
        finbert_labels = [c["finbert"]["sentiment_label"] for c in comparisons]
        finbert_distribution = {
            "positive": finbert_labels.count("positive"),
            "negative": finbert_labels.count("negative"),
            "neutral": finbert_labels.count("neutral")
        }

        return {
            "total_samples": total,
            "sentiment_agreement": {
                "high": high_agreement,
                "medium": medium_agreement,
                "low": low_agreement,
                "agreement_rate": (high_agreement + medium_agreement) / total if total > 0 else 0
            },
            "avg_sentiment_difference": avg_sentiment_diff,
            "finbert_distribution": finbert_distribution,
            "conclusion": self._generate_conclusion(avg_sentiment_diff, high_agreement, total)
        }

    def _generate_conclusion(self, avg_diff: float, high_agreement: int, total: int) -> str:
        """生成结论"""
        agreement_rate = high_agreement / total if total > 0 else 0

        if avg_diff < 0.2 and agreement_rate > 0.7:
            return "FinBERT和LLM在情绪判断上高度一致，LLM信号具有可靠性"
        elif avg_diff < 0.3 and agreement_rate > 0.5:
            return "FinBERT和LLM在情绪判断上基本一致，但存在一定差异"
        else:
            return "FinBERT和LLM在情绪判断上存在较大差异，需要进一步验证"

# 使用示例
if __name__ == "__main__":
    # 初始化FinBERT
    finbert = FinBERTModel()
    success = finbert.initialize()

    if success:
        # 测试文本
        test_texts = [
            "Apple reports strong earnings growth of 25% in Q4, beating all expectations.",
            "Tesla faces major lawsuit over workplace safety violations, potential fines up to $500M.",
            "Microsoft announces debt refinancing at favorable rates, extending maturity to 2030."
        ]

        # 批量分析
        results = finbert.analyze_batch(test_texts)

        for i, result in enumerate(results):
            print(f"\n文本 {i+1}:")
            print(f"  情绪: {result.sentiment_label}")
            print(f"  分数: {result.sentiment_score:.3f}")
            print(f"  置信度: {result.confidence:.3f}")
            print(f"  分布: 正:{result.positive_score:.3f}, 负:{result.negative_score:.3f}, 中:{result.neutral_score:.3f}")

        # 特征提取示例
        features = finbert.get_sentiment_features(test_texts)
        print(f"\n特征提取完成，共 {len(features)} 个特征组")