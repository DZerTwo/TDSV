import math
import multiprocessing
import pandas as pd
import pysam
import numpy as np
from PIL import Image
import colorsys
from multiprocessing.dummy import Pool as ThreadPool
import os

def decode_alignment_flag(flag):
    """解析比对标志位，判断链信息"""
    signal_map = {1 << 2: 0, 1 >> 1: 1, 1 << 4: 2, 1 << 11: 3, 1 << 4 | 1 << 11: 4}
    return signal_map.get(flag, 0)

def calculate_cigar_positions(cigar_str, ref_start):
    """根据CIGAR字符串计算比对的参考和读段位置"""
    num_chars = [str(i) for i in range(10)]
    read_start = False
    read_end = False
    ref_end = False
    read_loc = 0
    ref_loc = ref_start
    num_buffer = ''
    for c in cigar_str:
        if c in num_chars:
            num_buffer += c
        else:
            num_val = int(num_buffer)
            if not read_start and c in ['M', 'I', '=', 'X']:
                read_start = read_loc
            if read_start and c in ['H', 'S']:
                read_end = read_loc
                ref_end = ref_loc
                break
            if c in ['M', 'I', 'S', '=', 'X']:
                read_loc += num_val
            if c in ['M', 'D', 'N', '=', 'X']:
                ref_loc += num_val
            num_buffer = ''
    if not read_end:
        read_end = read_loc
        ref_end = ref_loc
    return ref_start, ref_end, read_start, read_end 

def analyze_split_reads(chr_name, bam_handle):
    """分析分裂读段，检测潜在结构变异"""
    pos_list = []
    for read in bam_handle.fetch(chr_name):
        if read.has_tag('SA'):
            flag_code = decode_alignment_flag(read.flag)
            sa_list = read.get_tag('SA').split(';')[:-1]
            for sa in sa_list:
                sa_info = sa.split(',')
                contig, ref_start, strand, cigar = sa_info[0], int(sa_info[1]), sa_info[2], sa_info[3]
                if contig != chr_name:
                    continue
                if (strand == '-' and flag_code % 2 == 0) or (strand == '+' and flag_code % 2 == 1):
                    ref_start1, ref_end1, _, _ = read.reference_start, read.reference_end, read.query_alignment_start, read.query_alignment_end
                    ref_start2, ref_end2, _, _ = calculate_cigar_positions(cigar, ref_start)
                    a = read.query_alignment_end - read.query_alignment_start
                    b = ref_end1 - ref_start2
                    if abs(b - a) < 30:
                        continue
                    if abs(b) < 2000 and 50 < abs(b - a) < 200000:
                        pos_list.extend([ref_start2, ref_end1])
    pos_series = pd.Series(pos_list).value_counts()
    return pos_series

def extract_region_features(bam_handle, chrom, start, end):
    """提取区域内的测序特征（覆盖度、剪切、插入等）"""
    ref_coverage = np.array([])
    left_clip = np.array([])
    right_clip = np.array([])
    insert_count = np.array([])
    window_size = 1000000
    current_start = start
    current_end = start + window_size
    for _ in range(10):
        ref_pos = []
        del_count = []
        l_clip = []
        r_clip = []
        ins = []
        for read in bam_handle.fetch(chrom, current_start, current_end):
            aligned_len = read.reference_length or 0
            if read.mapping_quality >= 0 and aligned_len >= 0:
                cigar_tuples = np.array(read.cigartuples)
                ref_pos.extend(read.get_reference_positions())
                ref_pos_start = read.reference_start + 1
                for i in range(cigar_tuples.shape[0]):
                    op, length = cigar_tuples[i]
                    if op == 0:  
                        ref_pos_start += length
                    elif op == 7:  
                        ref_pos_start += length
                    elif op == 8:
                        ref_pos_start += length
                    elif op == 2:
                        ref_pos_start += length
                    elif op == 1 and length >= 20:
                        ins.append(ref_pos_start)
                if cigar_tuples[0, 0] == 4:
                    l_clip.append(read.reference_start + 1)
                if cigar_tuples[-1, 0] == 4:
                    r_clip.append(read.reference_end)
        ref_pos = np.array(ref_pos) + 1 if ref_pos else np.array([0])
        ref_cov = np.bincount(ref_pos.astype(int), minlength=current_end + 1)[current_start:current_end]
        ref_coverage = np.append(ref_coverage, ref_cov)
        
        l_clip = np.array(l_clip) if l_clip else np.array([0])
        l_clip_cov = np.bincount(l_clip.astype(int), minlength=current_end + 1)[current_start:current_end]
        left_clip = np.append(left_clip, l_clip_cov)
        
        r_clip = np.array(r_clip) if r_clip else np.array([0])
        r_clip_cov = np.bincount(r_clip.astype(int), minlength=current_end + 1)[current_start:current_end]
        right_clip = np.append(right_clip, r_clip_cov)
        
        ins = np.array(ins) if ins else np.array([0])
        ins_cov = np.bincount(ins.astype(int), minlength=current_end + 1)[current_start:current_end]
        insert_count = np.append(insert_count, ins_cov)
        
        current_start += window_size
        current_end += window_size
    return ref_coverage, left_clip, right_clip, insert_count

def combine_features(coverage, left_clip, right_clip, split_reads, insertions, start, end):
    """组合多类特征并进行标准化"""
    pos_range = np.arange(start, end)
    coverage = coverage.reshape(-1, 1)
    left_clip = left_clip.reshape(-1, 1)
    right_clip = right_clip.reshape(-1, 1)
    insertions = insertions.reshape(-1, 1)
    split_cov = split_reads.reindex(index=pos_range).fillna(0).values.reshape(-1, 1)
    feature_matrix = np.concatenate([coverage, left_clip, right_clip, insertions, split_cov], axis=1)
    return feature_matrix

def normalize_features(feat_array):
    """对特征矩阵进行标准化处理"""
    reshaped = feat_array.reshape(-1, 5).astype('float32')
    reshaped -= reshaped.mean(axis=0)
    reshaped /= (np.sqrt(reshaped.var(axis=0)) + 1e-10)
    return reshaped.reshape(feat_array.shape)

def generate_chrom_features(bam_path, out_dir, chrom, index_list):
    """生成染色体级别的特征张量"""
    bam = pysam.AlignmentFile(bam_path, 'rb', threads=20)
    chrom_length = bam.lengths[bam.references.index(chrom)]
    segment_count = math.ceil(chrom_length / 10000000)
    start = 0
    end = 10000000
    split_read_pos = analyze_split_reads(chrom, bam)
    for seg_idx in range(segment_count):
        print(f"Processing {chrom} segment {seg_idx+1}/{segment_count}, start: {start}, end: {end}")
        cov, l_clip, r_clip, ins = extract_region_features(bam, chrom, start, end)
        if len(cov) == 0:
            start += 10000000
            end += 10000000
            continue
        feat_mat = combine_features(cov, l_clip, r_clip, split_read_pos, ins, start, end)
        feat_mat = feat_mat.reshape(-1, 1000)
        
        selected_indices = []
        for idx in index_list:
            pos = idx * 200
            if start <= pos < end:
                selected_indices.append(idx)
            elif pos >= end:
                break
        if not selected_indices:
            start += 10000000
            end += 10000000
            continue
        
        segment_feats = []
        for idx in selected_indices:
            local_idx = idx - start // 200
            segment_feats.append(feat_mat[local_idx])
        segment_feats = np.array(segment_feats)
        segment_feats = normalize_features(segment_feats)
        
        out_file = os.path.join(out_dir, f"{chrom}_{start}_{end}.npy")
        np.save(out_file, segment_feats)
        start += 10000000
        end += 10000000

def process_region(args):
    """多进程任务函数：处理单个区域文件"""
    bam_file, region_path, feat_dir = args
    chrom = region_path.split("/")[-1].split(".")[0].split("_")[-1]
    with open(region_path, 'r') as f:
        indices = [int(line.strip()) for line in f if line.strip()]
    generate_chrom_features(bam_file, feat_dir, chrom, indices)

def parallel_process_files(bam_file, region_dir, feat_dir, processes=8):
    """多进程处理所有区域文件"""
    region_files = [os.path.join(region_dir, f) for f in os.listdir(region_dir)]
    task_args = [(bam_file, f, feat_dir) for f in region_files]
    os.makedirs(feat_dir, exist_ok=True)
    with multiprocessing.Pool(processes=processes) as pool:
        pool.map(process_region, task_args)

if __name__ == "__main__":
    bam_input = "HG002_PB_70x_RG_HP10XtrioRTG.bam"
    region_dir = "area_list"
    feature_dir = "tensors"
    parallel_process_files(bam_input, region_dir, feature_dir, processes=8)
