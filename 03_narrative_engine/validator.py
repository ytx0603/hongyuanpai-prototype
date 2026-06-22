# -*- coding: utf-8 -*-
# 校验器：确保 AI 生成的叙事不乱编
# 三层检查：属性一致性 / 时代错误 / 不确定词汇
#
# TODO: 后面加一个基于 sentence-BERT 的语义冲突检测，现在这个规则太简单了
# TODO: 黑名单应该放到外部配置文件里，方便文保专家自己加词

import re
from typing import List
from dataclasses import dataclass
from config import NarrativeConfig
from kg_builder import KnowledgeGraph

cfg = NarrativeConfig()


@dataclass
class ValidationResult:
    passed: bool
    score: float
    violations: List[str]
    suggestions: List[str]


class HistoricalValidator:
    # 校验器主类
    # 策略：宁可误报也不能漏报（漏了会让 AI 瞎编的历史混过去）
    def __init__(self, kg):
        self.kg = kg
        # 不确定词黑名单 —— 这些词出现就扣分
        self._bl = ["可能", "也许", "据说", "传说", "大概", "maybe", "perhaps", "legend says"]

    def check_factual_consistency(self, text):
        # 检查文本中的实体属性跟 KG 记录的有没有冲突
        # 比如 KG 说材质=纸本，文本里写"金属制品" → 违规
        bad, fix = [], []
        for eid, ent in self.kg.entities.items():
            if ent.name not in text:
                continue
            attrs = ent.attributes or {}
            kg_mat = attrs.get("材质", "")
            if not kg_mat:
                continue
            # 遍历已知材质词，看有没有跟 KG 矛盾的
            for m in ["金属", "木质", "纸本", "石质", "陶瓷", "布料"]:
                if m in text and m != kg_mat:
                    pos = text.find(ent.name)
                    ctx = text[pos:pos+80] if pos >= 0 else text
                    if m in ctx:
                        bad.append(f"材质冲突: {ent.name} KG={kg_mat} text={m}")
                        fix.append(f"建议: {ent.name} 材质改为 {kg_mat}")
        sc = 1.0 - len(bad) * 0.25
        return ValidationResult(len(bad) == 0, max(sc, 0), bad, fix)

    def check_blacklist(self, text):
        return [f"不确定词: {w}" for w in self._bl if w in text]

    def check_anachronism(self, text):
        # 时代穿越检测：提取年份，跟 KG 记录比对，差太多就报
        bad = []
        yrs = re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', text)
        if not yrs:
            return bad
        for eid, ent in self.kg.entities.items():
            if ent.name not in text:
                continue
            kgp = (ent.attributes or {}).get("年代") or (ent.attributes or {}).get("时间") or ""
            if not kgp:
                continue
            ky = re.findall(r'\d{4}', str(kgp))
            for k in ky:
                for t in yrs:
                    if abs(int(t) - int(k)) > 80:
                        bad.append(f"年代冲突: {ent.name} ~{k}, text mentions {t}")
        return bad

    def validate(self, text, ctx=None):
        # 完整校验管线：三个检查合起来打分
        fv = self.check_factual_consistency(text).violations
        bv = self.check_blacklist(text)
        av = self.check_anachronism(text)

        all_v = fv + bv + av
        all_s = self.check_factual_consistency(text).suggestions
        if bv:
            all_s.append("把不确定词换成确定表述")

        penalty = len(all_v) * 0.15
        score = max(1.0 - penalty, 0.0)
        return ValidationResult(
            passed=score >= cfg.validator_threshold,
            score=score, violations=all_v, suggestions=all_s,
        )


def demo():
    print("="*50)
    print("Validator demo")
    print("="*50)
    kg = KnowledgeGraph()
    kg.build_sample_graph()
    v = HistoricalValidator(kg)
    tests = [
        "这是宁都烈士家书，出自1930年代。",
        "这件文物可能来自明代，据说与某位皇帝有关。",
        "宁都烈士家书是金属制品。",
    ]
    for i, t in enumerate(tests):
        r = v.validate(t)
        print(f"  [{i+1}] {t[:30]}... -> {'OK' if r.passed else 'FAIL'} ({r.score:.2f})")
        for x in r.violations:
            print(f"      violation: {x}")


if __name__ == "__main__":
    demo()
