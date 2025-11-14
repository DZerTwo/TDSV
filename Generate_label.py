import pandas as pd
import numpy as np
import os
import pysam

def generate_label(vcf_path, region_path, output_dir):
    
    # 读取VCF文件
    vcf = pysam.VariantFile(vcf_path)

    indexs = {}
    for rec in vcf.fetch():
        if rec.info.get('SVTYPE') == 'INS' and rec.chrom.isdigit():
            chrom = int(rec.chrom)
            start = rec.start  # pysam是0-based
            idx = start // 200
            # 记录染色体号和idx
            if chrom not in indexs:
                indexs[chrom] = []
            indexs[chrom].append(idx)
    
    # 确保输出目录存在
    
    os.makedirs(output_dir, exist_ok=True)
    # 读取区域列表
    file_list = [os.path.join(region_path, f) for f in os.listdir(region_path)]
    # 遍历每条染色体
    for file in file_list:
        if not os.path.isfile(file):
            continue
        with open(file, 'r') as f:
            area_list = [int(line.strip()) for line in f ]
            # 获取染色体号和对应的idx
            chrom = int(file.split("_")[-1].split(".")[0])
            idxs = indexs[chrom]
            labels = []
            for area in area_list:
                if area in idxs:
                    labels.append(1)
                else:
                    labels.append(0)
        label_path = os.path.join(output_dir, f"label_{chrom}.txt")
        # 保存标签列表到csv_path路径
        with open(label_path, "w") as f:
            for label in labels:
                f.write(f"{label}\n")
if __name__ == "__main__":
    vcf_path = "HG002_SVs_Tier1_v0.6.vcf"
    region_path = "area_list"
    output_dir = "labels2.0"
    generate_label(vcf_path, region_path, output_dir)
