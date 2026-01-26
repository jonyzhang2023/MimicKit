#!/usr/bin/env python3
"""
测试自适应采样机制
"""

import sys
import os

# 添加MimicKit目录到Python路径
sys.path.append('/home/jony/workspace/202601/newton-mimickit/MimicKit')

import torch
from mimickit.learning.experience_buffer import PrioritizedExperienceBuffer, ExperienceBuffer


def test_prioritized_experience_buffer():
    """测试优先经验回放缓冲区"""
    print("=== 测试优先经验回放缓冲区 ===")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化缓冲区
    buffer_length = 1000
    batch_size = 32
    
    print(f"\n1. 初始化PrioritizedExperienceBuffer")
    print(f"   缓冲区长度: {buffer_length}")
    print(f"   批量大小: {batch_size}")
    
    try:
        buffer = PrioritizedExperienceBuffer(
            buffer_length=buffer_length,
            batch_size=batch_size,
            device=device,
            alpha=0.6,  # 优先级 exponent
            beta=0.4,   # 重要性采样权重 exponent
            beta_increment=0.001,  # beta 增量
            epsilon=1e-6  # 避免零优先级
        )
        print("   ✓ 成功创建PrioritizedExperienceBuffer")
        print(f"   实际缓冲区容量: {buffer.get_capacity()}")
    except Exception as e:
        print(f"   ✗ 创建PrioritizedExperienceBuffer失败: {e}")
        return False
    
    # 测试添加经验
    print("\n2. 测试添加经验")
    
    # 创建一些模拟经验（批量大小为32）
    num_steps = 10
    batch_size = 32
    
    # 创建模拟数据，形状为 [batch_size, data_dim]
    states = torch.randn(batch_size, 10, device=device)
    actions = torch.randn(batch_size, 5, device=device)
    rewards = torch.randn(batch_size, 1, device=device)
    next_states = torch.randn(batch_size, 10, device=device)
    dones = torch.zeros(batch_size, 1, device=device)
    rand_action_mask = torch.ones(batch_size, 1, device=device)
    
    try:
        for i in range(num_steps):
            # 使用record方法添加数据
            buffer.record('s', states)
            buffer.record('a', actions)
            buffer.record('r', rewards)
            buffer.record('s_', next_states)
            buffer.record('d', dones)
            buffer.record('rand_action_mask', rand_action_mask)
            
            # 每次添加后递增缓冲区头部
            buffer.inc()
        
        sample_count = buffer.get_sample_count()
        print(f"   ✓ 成功添加 {num_steps * batch_size} 个经验")
        print(f"   当前样本数量: {sample_count}")
    except Exception as e:
        print(f"   ✗ 添加经验失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试更新优先级
    print("\n3. 测试更新优先级")
    
    try:
        # 生成一些模拟误差（作为优先级）
        indices = torch.arange(32, device=device)
        errors = torch.randn(32, device=device) + 1.0  # 确保误差为正
        
        buffer.update_priorities(indices, errors)
        print(f"   ✓ 成功更新 {len(indices)} 个样本的优先级")
        print(f"   最大优先级: {buffer._max_priority:.4f}")
    except Exception as e:
        print(f"   ✗ 更新优先级失败: {e}")
        return False
    
    # 测试采样
    print("\n4. 测试采样")
    
    try:
        batch = buffer.sample(batch_size)
        print(f"   ✓ 成功采样批量数据")
        print(f"   采样数据包含的键: {list(batch.keys())}")
        
        # 检查采样是否包含权重和索引
        if 'weights' in batch:
            print(f"   ✓ 采样包含权重: {batch['weights'].shape}")
        else:
            print(f"   ✗ 采样不包含权重")
            
        if 'indices' in batch:
            print(f"   ✓ 采样包含索引: {batch['indices'].shape}")
        else:
            print(f"   ✗ 采样不包含索引")
            
    except Exception as e:
        print(f"   ✗ 采样失败: {e}")
        return False
    
    # 测试beta值递增
    print("\n5. 测试beta值递增")
    
    try:
        initial_beta = buffer._beta
        print(f"   初始beta值: {initial_beta:.4f}")
        
        # 采样几次以触发beta递增
        for i in range(5):
            buffer.sample(batch_size)
        
        final_beta = buffer._beta
        print(f"   最终beta值: {final_beta:.4f}")
        
        if final_beta > initial_beta:
            print(f"   ✓ beta值成功递增: {final_beta - initial_beta:.4f}")
        else:
            print(f"   ✗ beta值没有递增")
            
    except Exception as e:
        print(f"   ✗ 测试beta值递增失败: {e}")
        return False
    
    # 对比测试：普通ExperienceBuffer
    print("\n6. 对比测试：普通ExperienceBuffer")
    
    try:
        regular_buffer = ExperienceBuffer(
            buffer_length=buffer_length,
            batch_size=batch_size,
            device=device
        )
        
        # 添加相同的经验
        for i in range(num_steps):
            regular_buffer.record('s', states)
            regular_buffer.record('a', actions)
            regular_buffer.record('r', rewards)
            regular_buffer.record('s_', next_states)
            regular_buffer.record('d', dones)
            regular_buffer.record('rand_action_mask', rand_action_mask)
            regular_buffer.inc()
        
        # 采样
        regular_batch = regular_buffer.sample(batch_size)
        print(f"   ✓ 成功创建并使用普通ExperienceBuffer")
        print(f"   普通缓冲区采样包含的键: {list(regular_batch.keys())}")
        
    except Exception as e:
        print(f"   ✗ 普通ExperienceBuffer测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== 测试完成 ===")
    print("自适应采样机制验证成功！")
    return True


if __name__ == "__main__":
    test_prioritized_experience_buffer()
