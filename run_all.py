"""
红源pi 全模块演示脚本 (使用统一数据接口)

运行方式: python run_all.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.data_interface import get_data_provider

def demo_rlvr(provider):
    print('='*60)
    print('模块一: RLVR 数字修复')
    print('='*60)
    sys.path.insert(0, '01_RLVR')
    from rlvr_train import RLVRPipeline
    pipeline = RLVRPipeline()
    pipeline.phase1_sft()
    pipeline.phase3_evaluate()
    print()

def demo_3dcnn(provider):
    print('='*60)
    print('模块二: 3D-CNN 智能监测')
    print('='*60)
    sys.path.insert(0, '02_3DCNN')
    from change_detection import demo
    demo()
    print()

def demo_narrative(provider):
    print('='*60)
    print('模块三: 多模态叙事引擎')
    print('='*60)
    sys.path.insert(0, '03_narrative_engine')
    from rag_generator import demo
    demo()
    print()

if __name__ == '__main__':
    t0 = time.time()
    provider = get_data_provider('synthetic', seed=42)
    print(f'Data Provider: {provider.provider_name}')
    print()

    # 验证数据接口
    scan = provider.get_artifact_scan('JGS-001')
    print(f'Sample: {scan.artifact_name} ({scan.metadata})')
    print()

    demo_rlvr(provider)
    demo_3dcnn(provider)
    demo_narrative(provider)

    elapsed = time.time() - t0
    print(f'Total: {elapsed:.1f}s')