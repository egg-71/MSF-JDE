"""
测试多尺度融合模块是否正常工作

运行此脚本验证：
1. 模块能否正确导入
2. 模型能否正确构建
3. 前向传播是否正常
"""

import torch
from ultralytics import YOLO

def test_module_import():
    """测试模块导入"""
    print("=" * 60)
    print("测试1: 模块导入")
    print("=" * 60)
    try:
        from ultralytics.nn.modules.head_msf import JDE_MSF, JDE_ASPP, JDE_ChannelAttention
        print("✅ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_model_build(model_yaml):
    """测试模型构建"""
    print(f"\n测试模型: {model_yaml}")
    print("-" * 60)
    try:
        model = YOLO(model_yaml, task='jde')
        print(f"✅ 模型构建成功")
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"   总参数量: {total_params:,}")
        
        return model
    except Exception as e:
        print(f"❌ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_forward_pass(model, model_name):
    """测试前向传播"""
    print(f"\n测试前向传播: {model_name}")
    print("-" * 60)
    try:
        # 创建虚拟输入
        batch_size = 2
        img_size = 640
        dummy_input = torch.randn(batch_size, 3, img_size, img_size)
        
        # 前向传播
        model.model.eval()
        with torch.no_grad():
            output = model.model(dummy_input)
        
        print(f"✅ 前向传播成功")
        print(f"   输入形状: {dummy_input.shape}")
        if isinstance(output, (list, tuple)):
            print(f"   输出数量: {len(output)}")
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    print(f"   输出{i}形状: {out.shape}")
        else:
            print(f"   输出形状: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 60)
    print("🚀 YOLO11-JDE 多尺度融合模块测试")
    print("=" * 60 + "\n")
    
    # 测试1: 模块导入
    if not test_module_import():
        print("\n❌ 测试失败：模块导入错误")
        return
    
    # 测试2: 模型构建
    print("\n" + "=" * 60)
    print("测试2: 模型构建")
    print("=" * 60)
    
    models_to_test = [
        ('yolo11s-jde.yaml', '基线模型'),
        ('yolo11s-jde-msf.yaml', '多尺度融合'),
        ('yolo11s-jde-aspp.yaml', 'ASPP版本'),
        ('yolo11s-jde-ca.yaml', '通道注意力版本'),
    ]
    
    results = {}
    for model_yaml, desc in models_to_test:
        model = test_model_build(model_yaml)
        if model is not None:
            # 测试3: 前向传播
            success = test_forward_pass(model, desc)
            results[desc] = success
        else:
            results[desc] = False
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    all_passed = True
    for model_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{model_name:20s}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！可以开始训练了")
        print("\n下一步：")
        print("  python train_msf.py")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
