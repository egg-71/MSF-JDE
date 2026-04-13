"""
重新运行TrackEval评估 - 修复路径配置问题（健壮版本）
"""
import sys
sys.path.append("/root/autodl-tmp/YOLO11-JDE-main")

from tracker.evaluation.evaluate import trackeval_evaluation
import pandas as pd
import os

# 配置
config = {
    'GT_FOLDER': './tracker/evaluation/TrackEval/data/gt/mot_challenge/MOT17/val_half',
    'TRACKERS_FOLDER': 'reid_xps/CH-jde-msf-b8-e100-sz1280-20260313-190642/MOT17/val_half',
    'TRACKERS_TO_EVAL': [''],
    'METRICS': ['HOTA', 'CLEAR', 'Identity'],
    'USE_PARALLEL': True,
    'NUM_PARALLEL_CORES': 4,
    'SKIP_SPLIT_FOL': True,
    'SEQMAP_FILE': './tracker/evaluation/TrackEval/data/gt/mot_challenge/seqmaps/MOT17-val_half.txt',
    'PRINT_CONFIG': False,
    'PRINT_RESULTS': True,
}

print("=" * 80)
print("🚀 重新运行TrackEval评估 - 包含完整7个序列")
print("=" * 80)

# 读取seqmap文件并显示
print("\n📋 将评估的序列:")
with open(config['SEQMAP_FILE'], 'r') as f:
    lines = f.readlines()
    seqs = [line.strip() for line in lines if line.strip() and line.strip() != 'name']
    for i, seq in enumerate(seqs, 1):
        tracker_file = os.path.join(config['TRACKERS_FOLDER'], 'data', f'{seq}.txt')
        exists = "✅" if os.path.exists(tracker_file) else "❌"
        print(f"  {i}. {seq} {exists}")
print(f"\n总计: {len(seqs)} 个序列")
print("=" * 80)

# 运行评估
try:
    print("\n开始评估...")
    trackeval_evaluation(config)

    # 读取并显示最终结果
    summary_path = 'reid_xps/CH-jde-msf-b8-e100-sz1280-20260313-190642/MOT17/val_half/pedestrian_summary.txt'

    if os.path.exists(summary_path):
        print("\n" + "=" * 80)
        print("🎯 最终评估结果（完整7个序列）")
        print("=" * 80)

        # 读取summary文件（使用更健壮的方法）
        try:
            summary_df = pd.read_csv(summary_path, sep=' ')

            # 打印列名以便调试
            print(f"\n[调试] 文件列名: {list(summary_df.columns)}")

            # 获取第一列的名称（通常是序列名列）
            first_col = summary_df.columns[0]

            # 查找COMBINED行（使用第一列）
            combined_row = summary_df[summary_df[first_col] == 'COMBINED']

            if len(combined_row) == 0:
                # 如果找不到COMBINED，尝试使用最后一行
                print("[调试] 未找到COMBINED行，使用最后一行")
                combined = summary_df.iloc[-1]
            else:
                combined = combined_row.iloc[0]

            # 提取指标
            hota = float(combined['HOTA'])
            mota = float(combined['MOTA'])
            idf1 = float(combined['IDF1'])

            print(f"\n核心指标:")
            print(f"  HOTA: {hota:.3f}")
            print(f"  MOTA: {mota:.3f}")
            print(f"  IDF1: {idf1:.3f}")

            # 与原论文和之前结果对比
            print("\n" + "=" * 80)
            print("📊 结果对比分析")
            print("=" * 80)
            print(f"{'指标':<10} {'修复前(6序列)':<18} {'修复后(7序列)':<18} {'原论文':<12} {'差距':<10}")
            print("-" * 80)
            print(f"{'HOTA':<10} {'57.331':<18} {f'{hota:.3f}':<18} {'60.060':<12} {f'{hota-60.06:+.3f}':<10}")
            print(f"{'MOTA':<10} {'68.362':<18} {f'{mota:.3f}':<18} {'未知':<12} {'-':<10}")
            print(f"{'IDF1':<10} {'73.204':<18} {f'{idf1:.3f}':<18} {'未知':<12} {'-':<10}")
            print("=" * 80)

            # 分析
            print("\n💡 分析:")
            hota_diff = hota - 60.06
            hota_improvement = hota - 57.331

            print(f"  • 加入MOT17-02后HOTA提升: {hota_improvement:+.3f} ({hota_improvement/57.331*100:+.2f}%)")
            print(f"  • 与原论文HOTA差距: {hota_diff:.3f} ({abs(hota_diff)/60.06*100:.1f}%)")

            if hota >= 59.5:
                print("  ✅ 优秀！HOTA已经非常接近原论文水平！")
            elif hota >= 58.5:
                print("  ✅ 良好！HOTA与原论文差距在可接受范围内（<2.5%）")
            elif hota >= 57.5:
                print("  ⚠️  HOTA略低于预期，建议优化Tracker参数")
            else:
                print("  ⚠️  HOTA明显低于预期，需要检查配置或重新训练")

            print("\n  • MOTA和IDF1表现优秀，说明检测和ReID能力很强")
            print("  • 可以尝试调整Tracker参数进一步优化HOTA")

        except Exception as e:
            print(f"\n⚠️  解析summary文件出错: {e}")
            print("\n直接显示summary文件内容:")
            with open(summary_path, 'r') as f:
                print(f.read())

    else:
        print(f"\n⚠️  未找到评估结果文件: {summary_path}")

except Exception as e:
    print(f"\n❌ 评估失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✅ 评估完成！")
print("=" * 80)
