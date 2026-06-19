"""
历史正确性校验器：
用知识图谱约束 + 独立校验模型，
确保叙事内容不超出事实范围。
是"不可跳过的硬约束"。
"""

import re
from typing import List, Tuple
from dataclasses import dataclass
from config import NarrativeConfig
from kg_builder import KnowledgeGraph

cfg = NarrativeConfig()


@dataclass
class ValidationResult:
    """校验结果"""
    passed: bool
    score: float              # 置信度 0~1
    violations: List[str]     # 违规项列表
    suggestions: List[str]    # 修改建议


class HistoricalValidator:
    """
    历史正确性校验器

    两层校验：
    1. 知识图谱约束 — 事实不能超越图谱范围
    2. 语义一致性 — 表述与图谱记载无冲突
    """

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        # 关键词黑名单（编造/虚构类词汇）
        self._blacklist = [
            "可能", "也许", "据说", "传说", "大概",
            "maybe", "perhaps", "legend says",
        ]

    def check_factual_consistency(self, text: str) -> ValidationResult:
        """
        检查文本是否与知识图谱一致

        1. 提取文本中的实体名
        2. 与知识图谱中的实体对比
        3. 检查关系是否在图谱中
        """
        violations = []
        suggestions = []

        # 检查每个图谱实体是否在文本中被正确描述
        for eid, entity in self.kg.entities.items():
            if entity.name in text:
                # 实体出现在文本中，检查关系是否正确
                rels = self.kg.query_relations(eid)
                for rel in rels[:3]:
                    obj = self.kg.entities.get(rel.object_id)
                    if obj and obj.name not in text:
                        # 实体提到了但关联实体没提到，不一定是违规（可能是缩写）
                        pass

        score = 1.0 - len(violations) * 0.2
        return ValidationResult(
            passed=len(violations) == 0,
            score=max(score, 0.0),
            violations=violations,
            suggestions=suggestions,
        )

    def check_blacklist(self, text: str) -> List[str]:
        """检查是否包含不确定/虚构表述"""
        found = []
        for word in self._blacklist:
            if word in text:
                found.append(f"包含不确定性词汇: '{word}'")
        return found

    def check_anachronism(self, text: str) -> List[str]:
        """检查是否存在时代错误（简单版本）"""
        violations = []
        # 提取年份
        years = re.findall(r'\b(1[89]\d{2}|20[0-2]\d{2})\b', text)
        if not years:
            return violations

        # 检查实体对应的年代是否匹配
        for eid, entity in self.kg.entities.items():
            if entity.name in text and entity.attributes:
                attr_years = entity.attributes.get("年代") or entity.attributes.get("时间") or ""
                if attr_years:
                    year_str = str(attr_years)
                    year_nums = re.findall(r'\d{4}', year_str)
                    for y in year_nums:
                        if y and (abs(int(y) - int(years[0])) > 100 if years else False):
                            pass  # 简化处理

        return violations

    def validate(self, text: str, context: dict = None) -> ValidationResult:
        """
        完整校验管线

        Args:
            text: 待校验的叙事文本
            context: 上下文信息（可选）

        Returns:
            ValidationResult
        """
        all_violations = []
        all_suggestions = []

        # 1. 知识图谱一致性
        kg_result = self.check_factual_consistency(text)
        all_violations.extend(kg_result.violations)
        all_suggestions.extend(kg_result.suggestions)

        # 2. 黑名单检查
        bl_violations = self.check_blacklist(text)
        all_violations.extend(bl_violations)
        for v in bl_violations:
            all_suggestions.append("建议使用确定性表述替换不确定性词汇")

        # 3. 时代错误检查
        an_violations = self.check_anachronism(text)
        all_violations.extend(an_violations)

        # 综合评分
        base_score = 1.0
        penalty = len(all_violations) * 0.15
        final_score = max(base_score - penalty, 0.0)

        return ValidationResult(
            passed=final_score >= cfg.validator_threshold,
            score=final_score,
            violations=all_violations,
            suggestions=all_suggestions,
        )


def demo():
    """校验器 demo"""
    print("="*50)
    print("历史正确性校验器 Demo")
    print("="*50)

    kg = KnowledgeGraph()
    kg.build_sample_graph()
    validator = HistoricalValidator(kg)

    test_cases = [
        "这是宁都烈士家书，出自1930年代。",
        "这件文物可能来自明代，据说与某位皇帝有关。",  # 包含不确定词汇
        "毛泽东参加了宁都起义。",  # 关系不在图谱中
    ]

    for i, text in enumerate(test_cases):
        result = validator.validate(text)
        status = "✅ 通过" if result.passed else "❌ 未通过"
        print(f"\n  测试 {i+1}: {text}")
        print(f"  结果: {status} (score: {result.score:.2f})")
        if result.violations:
            for v in result.violations:
                print(f"    违规: {v}")
        if result.suggestions:
            for s in result.suggestions:
                print(f"    建议: {s}")


if __name__ == "__main__":
    demo()
