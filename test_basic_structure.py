#!/usr/bin/env python3
"""
Evo-1 项目基本结构测试脚本
这个脚本不需要安装依赖，仅验证代码结构的正确性
"""

import os
import sys

def test_file_structure():
    """测试文件结构是否完整"""
    print("=" * 80)
    print("1. 测试文件结构")
    print("=" * 80)

    required_files = [
        'Evo_1/scripts/Evo1.py',
        'Evo_1/scripts/train.py',
        'Evo_1/scripts/Evo1_server.py',
        'Evo_1/model/action_head/flow_matching.py',
        'Evo_1/model/internvl3/internvl3_embedder.py',
        'Evo_1/dataset/lerobot_dataset_pretrain_mp.py',
        'Evo_1/dataset/config.yaml',
        'Evo_1/requirements.txt',
        'README.md',
    ]

    missing_files = []
    existing_files = []

    for file_path in required_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✓ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"✗ {file_path} (缺失)")

    print(f"\n总计: {len(existing_files)}/{len(required_files)} 文件存在")

    if missing_files:
        print(f"\n缺失文件: {len(missing_files)}个")
        return False
    return True

def test_code_syntax():
    """测试代码语法是否正确（不执行）"""
    print("\n" + "=" * 80)
    print("2. 测试代码语法")
    print("=" * 80)

    python_files = [
        'Evo_1/scripts/Evo1.py',
        'Evo_1/scripts/train.py',
        'Evo_1/model/action_head/flow_matching.py',
    ]

    all_valid = True

    for file_path in python_files:
        if not os.path.exists(file_path):
            print(f"✗ {file_path} (文件不存在)")
            all_valid = False
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            # 尝试编译（不执行）
            compile(code, file_path, 'exec')
            print(f"✓ {file_path} (语法正确)")
        except SyntaxError as e:
            print(f"✗ {file_path} (语法错误: {e})")
            all_valid = False
        except Exception as e:
            print(f"? {file_path} (无法检查: {e})")

    return all_valid

def test_documentation():
    """测试文档是否生成"""
    print("\n" + "=" * 80)
    print("3. 测试文档完整性")
    print("=" * 80)

    doc_files = [
        'docs/原理说明.md',
        'docs/中文使用手册.md',
        'docs/Evo1_测试教程.ipynb',
    ]

    all_exist = True

    for doc_path in doc_files:
        if os.path.exists(doc_path):
            size = os.path.getsize(doc_path)
            print(f"✓ {doc_path} ({size / 1024:.1f} KB)")
        else:
            print(f"✗ {doc_path} (缺失)")
            all_exist = False

    return all_exist

def test_readme_structure():
    """测试README结构"""
    print("\n" + "=" * 80)
    print("4. 测试README结构")
    print("=" * 80)

    readme_path = 'README.md'

    if not os.path.exists(readme_path):
        print(f"✗ README.md 不存在")
        return False

    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    required_sections = [
        'Installation',
        'Meta-World',
        'LIBERO',
        'Training',
        'Citation',
    ]

    all_found = True
    for section in required_sections:
        if section in content:
            print(f"✓ 包含 '{section}' 章节")
        else:
            print(f"✗ 缺少 '{section}' 章节")
            all_found = False

    return all_found

def analyze_code_complexity():
    """分析代码复杂度"""
    print("\n" + "=" * 80)
    print("5. 代码统计信息")
    print("=" * 80)

    key_files = {
        'Evo_1/scripts/Evo1.py': 'EVO1主模型',
        'Evo_1/model/action_head/flow_matching.py': 'Flow Matching动作头',
        'Evo_1/scripts/train.py': '训练脚本',
    }

    for file_path, description in key_files.items():
        if not os.path.exists(file_path):
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total_lines = len(lines)
        code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))

        print(f"\n{description} ({file_path}):")
        print(f"  总行数: {total_lines}")
        print(f"  代码行: {code_lines}")
        print(f"  注释行: {comment_lines}")
        print(f"  代码比: {code_lines/total_lines*100:.1f}%")

def test_yaml_config():
    """测试YAML配置文件"""
    print("\n" + "=" * 80)
    print("6. 测试配置文件")
    print("=" * 80)

    config_path = 'Evo_1/dataset/config.yaml'

    if not os.path.exists(config_path):
        print(f"✗ {config_path} 不存在")
        return False

    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    required_keys = [
        'max_action_dim',
        'max_state_dim',
        'max_views',
        'data_groups',
    ]

    all_found = True
    for key in required_keys:
        if key in content:
            print(f"✓ 包含 '{key}' 配置")
        else:
            print(f"✗ 缺少 '{key}' 配置")
            all_found = False

    return all_found

def main():
    print("\n" + "=" * 80)
    print(" " * 25 + "Evo-1 项目结构测试")
    print("=" * 80 + "\n")

    results = {}

    results['file_structure'] = test_file_structure()
    results['code_syntax'] = test_code_syntax()
    results['documentation'] = test_documentation()
    results['readme'] = test_readme_structure()
    results['yaml_config'] = test_yaml_config()

    analyze_code_complexity()

    # 总结
    print("\n" + "=" * 80)
    print(" " * 30 + "测试总结")
    print("=" * 80 + "\n")

    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name:20s}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print(" " * 25 + "所有测试通过！✓")
    else:
        print(" " * 20 + "部分测试失败，请检查上述输出")
    print("=" * 80 + "\n")

    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
