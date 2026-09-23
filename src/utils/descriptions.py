# -*- coding: utf-8 -*-
"""
类别文本描述字典。

get_text_features (model.py) 遍历每类的描述句 list、CLIP 编码后取平均作为类别文本原型。
当前管线只使用下方两个原始单词字典（与 DNP_Original 基线一致）。

历史注记：LLM 描述句增强实验线的 RICH_*/NEG_NORMAL_*/get_descriptions（--desc_variant A/C/AC）
已于 2026-09-07 归档至 ../_archive_20260907/utils_unused/descriptions_llm_variants.py。
"""

DESCRIPTIONS_ORI = {
    "normal": ["normal"],
    "abuse": ["abuse"],
    "arrest": ["arrest"],
    "arson": ["arson"],
    "assault": ["assault"],
    "burglary": ["burglary"],
    "explosion": ["explosion"],
    "fighting": ["fighting"],
    "roadaccidents": ["roadaccidents"],
    "robbery": ["robbery"],
    "shooting": ["shooting"],
    "shoplifting": ["shoplifting"],
    "stealing": ["stealing"],
    "vandalism": ["vandalism"],
}

DESCRIPTIONS_ORI_XD = {
    "A": ["normal"],
    "B1": ["fighting"],
    "B2": ["shooting"],
    "B4": ["riot"],
    "B5": ["abuse"],
    "B6": ["car accident"],
    "G": ["explosion"],
}
