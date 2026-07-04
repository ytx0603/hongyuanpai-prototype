
import json, os
kg = {}
kg['schema'] = '2.0'
kg['material_taxonomy'] = {
    'METAL_CU': {'name': 'copper alloy', 'subtypes': ['bronze', 'brass'], 'eras': ['Shang-Zhou', 'Ming-Qing']},
    'METAL_FE': {'name': 'iron/steel', 'subtypes': ['cast iron', 'wrought iron'], 'eras': ['Ming-Qing', 'modern']},
    'CERAMIC': {'name': 'ceramic', 'subtypes': ['celadon(W Jin)', 'blue-white', 'famille rose', 'enamel'], 'eras': ['W Jin', 'Ming-Qing', 'modern']},
    'WOOD': {'name': 'wood/bamboo', 'subtypes': ['hardwood', 'softwood', 'bamboo'], 'eras': ['Ming', 'Qing', 'modern']},
    'PAPER': {'name': 'paper', 'subtypes': ['handmade', 'machine-made'], 'eras': ['Ming-Qing', 'post-1949']},
    'LEATHER_TEXTILE': {'name': 'leather/textile', 'subtypes': ['cowhide', 'cotton', 'silk'], 'eras': ['any']}
}
print('Schema built')
with open(r'C:/Users/Lenovo/Projects/hongyuanpai-prototype/data/4dage_models/restoration_kg_v2.json', 'w', encoding='utf-8') as f:
    json.dump(kg, f, ensure_ascii=False, indent=2)
print('Saved')
