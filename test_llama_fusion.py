"""
test_llama_fusion.py - Llama 模型融合集成验证脚本

测试 LlamaMLP 和 LlamaDecoderLayer 中的融合逻辑是否正确集成。
"""

import sys

def test_import_fusion_utils():
    """测试 fusion_utils 导入"""
    print("=" * 50)
    print("Test 1: Import fusion_utils")
    print("=" * 50)
    
    try:
        from infinilm.fusion_utils import (
            create_swiglu_pattern,
            create_add_rms_norm_pattern,
            LLMFusionContext,
            FusionManager
        )
        print("✅ All fusion_utils imports successful!")
        
        # 验证模式创建
        swiglu = create_swiglu_pattern()
        add_rms = create_add_rms_norm_pattern()
        print(f"  - SwiGLU pattern: {len(swiglu)} nodes")
        print(f"  - Add+RMSNorm pattern: {len(add_rms)} nodes")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_llama_config_fusion_toggle():
    """测试 LlamaConfig 的 enable_fusion 开关"""
    print("\n" + "=" * 50)
    print("Test 2: LlamaConfig enable_fusion toggle")
    print("=" * 50)
    
    try:
        from infinilm.models.llama import LlamaConfig
        
        # 测试默认开启
        config_on = LlamaConfig(torch_dtype='float16')
        print(f"  Default enable_fusion: {config_on.enable_fusion}")
        assert config_on.enable_fusion == True, "Default should be True"
        
        # 测试显式关闭
        config_off = LlamaConfig(enable_fusion=False, torch_dtype='float16')
        print(f"  Explicit enable_fusion=False: {config_off.enable_fusion}")
        assert config_off.enable_fusion == False, "Should be False when set"
        
        print("✅ LlamaConfig enable_fusion toggle works!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_llama_mlp_has_config():
    """测试 LlamaMLP 是否保存了 config"""
    print("\n" + "=" * 50)
    print("Test 3: LlamaMLP has self.config")
    print("=" * 50)
    
    try:
        from infinilm.models.llama import LlamaConfig
        from infinilm.models.llama.modeling_llama import LlamaMLP
        import infinicore
        
        config = LlamaConfig(
            hidden_size=256,
            intermediate_size=512,
            torch_dtype='float16'
        )
        
        # 创建 MLP (不需要 GPU，只检查结构)
        mlp = LlamaMLP(config)
        
        assert hasattr(mlp, 'config'), "LlamaMLP should have self.config"
        assert mlp.config.enable_fusion == True, "enable_fusion should be accessible"
        
        print(f"  mlp.config exists: {hasattr(mlp, 'config')}")
        print(f"  mlp.config.enable_fusion: {mlp.config.enable_fusion}")
        print("✅ LlamaMLP correctly stores config!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llama_decoder_layer_has_config():
    """测试 LlamaDecoderLayer 是否保存了 config"""
    print("\n" + "=" * 50)
    print("Test 4: LlamaDecoderLayer has self.config")
    print("=" * 50)
    
    try:
        from infinilm.models.llama import LlamaConfig
        from infinilm.models.llama.modeling_llama import LlamaDecoderLayer
        
        config = LlamaConfig(
            hidden_size=256,
            intermediate_size=512,
            num_attention_heads=4,
            num_key_value_heads=4,
            torch_dtype='float16'
        )
        
        layer = LlamaDecoderLayer(config, layer_idx=0)
        
        assert hasattr(layer, 'config'), "LlamaDecoderLayer should have self.config"
        assert layer.config.enable_fusion == True, "enable_fusion should be accessible"
        
        print(f"  layer.config exists: {hasattr(layer, 'config')}")
        print(f"  layer.config.enable_fusion: {layer.config.enable_fusion}")
        print("✅ LlamaDecoderLayer correctly stores config!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "#" * 60)
    print("  Llama Fusion Integration Test Suite")
    print("#" * 60 + "\n")
    
    results = []
    
    results.append(("Import fusion_utils", test_import_fusion_utils()))
    results.append(("LlamaConfig toggle", test_llama_config_fusion_toggle()))
    results.append(("LlamaMLP has config", test_llama_mlp_has_config()))
    results.append(("LlamaDecoderLayer has config", test_llama_decoder_layer_has_config()))
    
    # 汇总
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All Phase 5 integration tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
