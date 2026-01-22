import torch
import torch.nn as nn
import math
from typing import Optional

class RotaryPositionEmbeddingSimple(nn.Module):
    """
    修复bug后的直观RoPE实现
    """
    
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        
        assert dim % 2 == 0, f"维度必须是偶数，当前dim={dim}"
        
        self.dim = dim
        self.base = base
        
        print(f"🔧 RoPE初始化: dim={dim}, base={base}")
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        修复bug的RoPE前向传播
        """
        print(f"\n🔄 开始RoPE计算...")
        print(f"   输入形状: {x.shape}")
        
        # 保存原始形状
        original_shape = x.shape
        
        # 获取输入信息
        if x.dim() == 3:  # [batch, seq_len, dim]
            batch_size, seq_len, dim = x.shape
            num_heads = 1
        elif x.dim() == 4:  # [batch, heads, seq_len, dim]
            batch_size, num_heads, seq_len, dim = x.shape
        else:
            raise ValueError(f"不支持的输入维度: {x.dim()}")
        
        # 处理位置编码
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
            print(f"   自动生成位置: {positions.tolist()}")
        else:
            print(f"   使用自定义位置: {positions.tolist()}")
        
        # 1. 计算频率向量
        print(f"\n📊 步骤1: 计算频率向量")
        indices = torch.arange(0, dim, 2).float().to(x.device)  # [0, 2, 4, ..., dim-2]
        inv_freq = 1.0 / (self.base ** (indices / dim))
        print(f"   频率向量: {inv_freq.tolist()}")
        
        # 2. 计算角度矩阵
        print(f"\n📊 步骤2: 计算角度矩阵")
        # positions: [seq_len] -> [seq_len, 1]
        # inv_freq: [dim/2] -> [1, dim/2]
        positions_expanded = positions.unsqueeze(-1)  # [seq_len, 1]
        inv_freq_expanded = inv_freq.unsqueeze(0)    # [1, dim/2]
        
        angles = positions_expanded * inv_freq_expanded  # [seq_len, dim/2]
        print(f"   角度矩阵形状: {angles.shape}")
        
        # 3. 扩展角度到每个维度
        angles_expanded = angles.repeat_interleave(2, dim=-1)  # [seq_len, dim]
        print(f"   扩展后角度形状: {angles_expanded.shape}")
        
        # 4. 计算正弦余弦
        sin = torch.sin(angles_expanded)  # [seq_len, dim]
        cos = torch.cos(angles_expanded)  # [seq_len, dim]
        print(f"   正弦形状: {sin.shape}, 余弦形状: {cos.shape}")
        
        # 5. 关键修复：正确调整形状以匹配输入
        print(f"\n📊 步骤3: 调整形状匹配输入")
        if x.dim() == 3:  # [batch, seq_len, dim]
            # 扩展维度: [seq_len, dim] -> [batch, seq_len, dim]
            sin = sin.unsqueeze(0).expand(batch_size, -1, -1)  # 使用expand而不是repeat
            cos = cos.unsqueeze(0).expand(batch_size, -1, -1)
        elif x.dim() == 4:  # [batch, heads, seq_len, dim]
            # 扩展维度: [seq_len, dim] -> [batch, heads, seq_len, dim]
            sin = sin.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
            cos = cos.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
        
        print(f"   调整后正弦形状: {sin.shape}")
        print(f"   调整后余弦形状: {cos.shape}")
        print(f"   输入x形状: {x.shape}")
        
        # 6. 应用旋转
        result = self._apply_rotation_detailed(x, sin, cos)
        
        print(f"✅ RoPE计算完成")
        print(f"   输入: {original_shape} -> 输出: {result.shape}")
        
        return result
    
    def _apply_rotation_detailed(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        """修复后的旋转操作"""
        print(f"\n📊 步骤4: 应用旋转操作")
        
        # 检查形状是否匹配
        assert x.shape == sin.shape == cos.shape, f"形状不匹配: x{x.shape}, sin{sin.shape}, cos{cos.shape}"
        
        # 分割输入张量
        x1 = x[..., 0::2]  # 所有偶数索引维度
        x2 = x[..., 1::2]  # 所有奇数索引维度
        
        print(f"   x1形状 (偶数维度): {x1.shape}")
        print(f"   x2形状 (奇数维度): {x2.shape}")
        
        # 分割正弦余弦（确保形状匹配）
        sin1 = sin[..., 0::2]  # 对应x1的正弦
        cos1 = cos[..., 0::2]  # 对应x1的余弦
        sin2 = sin[..., 1::2]  # 对应x2的正弦  
        cos2 = cos[..., 1::2]  # 对应x2的余弦
        
        print(f"   sin1形状: {sin1.shape}, cos1形状: {cos1.shape}")
        print(f"   sin2形状: {sin2.shape}, cos2形状: {cos2.shape}")
        
        # 应用旋转公式（确保广播正确）
        rotated_x1 = x1 * cos1 - x2 * sin2
        rotated_x2 = x1 * sin1 + x2 * cos2
        
        print(f"   rotated_x1形状: {rotated_x1.shape}")
        print(f"   rotated_x2形状: {rotated_x2.shape}")
        
        # 重新组合
        result = torch.stack([rotated_x1, rotated_x2], dim=-1)
        result = result.flatten(start_dim=-2)
        
        print(f"   最终输出形状: {result.shape}")
        return result


def test_fixed_version():
    """测试修复后的版本"""
    print("=" * 60)
    print("🧪 测试修复后的版本")
    print("=" * 60)
    
    # 测试1: 3D输入
    print("测试1: 3D输入 [batch, seq_len, dim]")
    dim = 6
    rope = RotaryPositionEmbeddingSimple(dim)
    
    x_3d = torch.randn(2, 3, dim)  # [batch=2, seq_len=3, dim=6]
    positions = torch.tensor([0, 1, 2])
    
    try:
        result_3d = rope(x_3d, positions)
        print("✅ 3D输入测试通过")
    except Exception as e:
        print(f"❌ 3D输入测试失败: {e}")
    
    # 测试2: 4D输入
    print("\n测试2: 4D输入 [batch, heads, seq_len, dim]")
    x_4d = torch.randn(2, 4, 3, dim)  # [batch=2, heads=4, seq_len=3, dim=6]
    
    try:
        result_4d = rope(x_4d, positions)
        print("✅ 4D输入测试通过")
    except Exception as e:
        print(f"❌ 4D输入测试失败: {e}")


def debug_shape_issue():
    """调试原始的形状问题"""
    print("\n" + "=" * 60)
    print("🐛 调试原始的形状问题")
    print("=" * 60)
    
    dim = 4
    batch_size, seq_len = 2, 3
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, dim)
    positions = torch.arange(seq_len)
    
    print("原始问题分析:")
    print(f"输入x形状: {x.shape}")  # [2, 3, 4]
    
    # 计算正弦余弦（错误的方式）
    indices = torch.arange(0, dim, 2).float()
    inv_freq = 1.0 / (10000 ** (indices / dim))
    
    angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)  # [3, 2]
    angles_expanded = angles.repeat_interleave(2, dim=-1)     # [3, 4]
    
    sin = torch.sin(angles_expanded)  # [3, 4]
    cos = torch.cos(angles_expanded)  # [3, 4]
    
    print(f"计算出的sin形状: {sin.shape}")  # [3, 4]
    print(f"计算出的cos形状: {cos.shape}")  # [3, 4]
    
    # 错误：直接使用会导致形状不匹配
    print(f"❌ 问题: sin{sin.shape} 与 x{x.shape} 形状不匹配")
    print(f"❌ 需要将sin从[3,4]扩展到[2,3,4]")
    
    # 正确的方式
    sin_correct = sin.unsqueeze(0).expand(batch_size, -1, -1)  # [2, 3, 4]
    print(f"✅ 正确扩展后: {sin_correct.shape}")


def simple_demo():
    """简单的演示"""
    print("\n" + "=" * 60)
    print("🎯 简单演示")
    print("=" * 60)
    
    # 使用更小的维度便于观察
    dim = 4
    rope = RotaryPositionEmbeddingSimple(dim, base=100)
    
    # 创建简单的测试数据
    x = torch.tensor([
        [[1.0, 0.0, 0.5, 0.5],  # 第一个序列
         [0.0, 1.0, 0.3, 0.7]],
        
        [[0.5, 0.5, 1.0, 0.0],  # 第二个序列  
         [0.7, 0.3, 0.0, 1.0]]
    ])  # [batch=2, seq_len=2, dim=4]
    
    print("输入数据:")
    print(f"批次0, token0: {x[0,0].tolist()}")
    print(f"批次0, token1: {x[0,1].tolist()}")
    print(f"批次1, token0: {x[1,0].tolist()}")
    
    # 应用RoPE
    result = rope(x)
    
    print("\n旋转后数据:")
    print(f"批次0, token0: {result[0,0].tolist()}")
    print(f"批次0, token1: {result[0,1].tolist()}")
    print(f"批次1, token0: {result[1,0].tolist()}")


def verify_calculation():
    """验证计算正确性"""
    print("\n" + "=" * 60)
    print("✅ 验证计算正确性")
    print("=" * 60)
    
    # 使用2维向量手动验证
    dim = 2
    rope = RotaryPositionEmbeddingSimple(dim, base=10000)
    
    # 创建简单的测试向量
    x = torch.tensor([[[1.0, 0.0]]])  # [1, 1, 2]
    positions = torch.tensor([1])
    
    # 手动计算期望结果
    # 对于2维向量，只有一个频率θ
    theta = 1.0 / (10000 ** (0 / 2))  # i=0, θ=1.0
    angle = 1 * theta  # 位置1，角度=1弧度
    
    # 手动旋转计算
    x_manual = torch.tensor([
        [1.0 * math.cos(angle) - 0.0 * math.sin(angle),
         1.0 * math.sin(angle) + 0.0 * math.cos(angle)]
    ])
    
    # RoPE计算
    x_rope = rope(x, positions)
    
    print(f"手动计算: {x_manual.tolist()}")
    print(f"RoPE计算: {x_rope[0,0].tolist()}")
    
    # 检查是否一致
    diff = torch.abs(x_manual - x_rope[0,0]).max().item()
    if diff < 1e-6:
        print("✅ 计算正确性验证通过")
    else:
        print(f"❌ 计算有差异: {diff}")


def main():
    """运行所有测试"""
    print("🚀 开始修复后的RoPE测试")
    
    # 运行测试
    test_fixed_version()
    debug_shape_issue()
    simple_demo()
    verify_calculation()
    
    print("\n" + "=" * 60)
    print("🎉 所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()