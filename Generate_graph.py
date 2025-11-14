import multiprocessing
import pysam
import numpy as np
from PIL import Image
import colorsys
import os

def filter_cigar_operations(cigar_list, win_start, win_end, ref_pos):
    """
    筛选并处理指定窗口内的CIGAR操作，返回优化后的CIGAR列表及有效长度
    ref_pos: 读段在参考序列上的起始位置
    """
    valid_length = 0
    if not cigar_list:
        return [], 0
    
    current_ref = ref_pos
    filtered_cigar = []
    for op, length in cigar_list:
        # 仅处理M(0)、I(1)、D(2)、S(4)操作
        if op in (0, 2):  # 消耗参考序列的操作
            op_start = current_ref
            op_end = current_ref + length
            # 完全在窗口外则跳过
            if op_end <= win_start:
                current_ref += length
                continue
            if op_start >= win_end:
                break
            # 短操作合并为匹配
            if length < 5:
                op = 0
            filtered_cigar.append((op, length))
            current_ref += length
            # 计算窗口内有效长度
            overlap_start = max(op_start, win_start)
            overlap_end = min(op_end, win_end)
            valid_length += overlap_end - overlap_start
        elif op == 1:  # 插入操作（不消耗参考）
            if win_start <= current_ref < win_end:  # 仅保留窗口内的插入
                if length < 5:
                    op = 0
                filtered_cigar.append((op, length))
                valid_length += length
    
    # 合并相邻相同操作
    optimized_cigar = []
    for op, length in filtered_cigar:
        if optimized_cigar and op == optimized_cigar[-1][0]:
            optimized_cigar[-1] = (op, optimized_cigar[-1][1] + length)
        else:
            optimized_cigar.append((op, length))
    return optimized_cigar, valid_length

def modify_color_strength(base_rgb, seg_len, total_len=200):
    """根据片段长度调整颜色明暗（基于HSV色彩空间）"""
    r, g, b = base_rgb
    # 转换为HSV并调整明度
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    v *= (1 - min(seg_len / total_len, 1.0))  # 长度越长，明度越低
    # 转回RGB
    adj_r, adj_g, adj_b = colorsys.hsv_to_rgb(h, s, v)
    return (int(adj_r*255), int(adj_g*255), int(adj_b*255))

def create_region_image(bam_path, chrom_region, img_output_dir):
    """
    为指定基因组区域生成可视化图像
    chrom_region: (染色体名, 起始位置, 终止位置)
    以200bp窗口为单位，将测序读段的CIGAR特征绘制成RGB图像
    """
    bam_handle = pysam.AlignmentFile(bam_path, "rb")
    chrom, start_pos, end_pos = chrom_region
    
    # 操作类型与颜色映射
    op_color = {
        0: (0, 255, 0),    
        1: (0, 0, 255),   
        2: (255, 0, 0),   
        4: (0, 0, 0)      
    }
    
    read_rows = []
    # 遍历区域内的所有读段
    for read in bam_handle.fetch(chrom, start_pos, end_pos):
        if read.is_unmapped:
            continue
        align_len = read.reference_length or 0
        # 过滤低质量比对
        if read.mapping_quality < 20 or align_len <= 0:
            continue
        
        # 处理当前读段的CIGAR操作
        read_start = read.reference_start
        filtered_cigar, total_valid = filter_cigar_operations(
            read.cigartuples, start_pos, end_pos, read_start
        )
        # 计算读段在窗口内的实际范围
        win_read_start = max(read_start, start_pos)
        win_read_end = min(read.reference_end, end_pos)
        
        # 构建当前读段的像素行
        pixel_row = []
        # 填充窗口起始到读段起始的空白（黑色）
        if win_read_start > start_pos:
            pixel_row.extend([(0, 0, 0)] * (win_read_start - start_pos))
        
        current_pos = read_start
        for op, length in filtered_cigar:
            orig_len = length
            # 处理参考消耗型操作（M/D）
            if op in (0, 2):
                if current_pos + length <= win_read_start:
                    current_pos += length
                    continue
                # 截断到窗口范围内
                if current_pos < win_read_start:
                    length -= win_read_start - current_pos
                if current_pos + length > win_read_end:
                    length = win_read_end - current_pos
                current_pos += orig_len  # 保持原始位置计算
            # 处理插入操作（I）
            elif op == 1:
                if not (win_read_start <= current_pos < win_read_end):
                    continue  # 跳过窗口外的插入
            
            # 添加对应颜色的像素
            if op in (0, 1, 2):
                base_rgb = op_color[op]
                if op == 1:
                    # 插入操作根据长度调整颜色
                    adj_rgb = modify_color_strength(base_rgb, length, total_valid)
                    pixel_row.extend([adj_rgb] * length)
                else:
                    pixel_row.extend([base_rgb] * length)
        
        # 填充读段结束到窗口结束的空白（黑色）
        if win_read_end < end_pos:
            pixel_row.extend([(0, 0, 0)] * (end_pos - win_read_end))
        
        if pixel_row:  # 仅保留有效行
            read_rows.append(pixel_row)
    
    # 生成图像（200x200）
    canvas_size = (200, 200)
    if not read_rows:
        # 无有效读段时生成黑色图像
        black_img = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    else:
        # 统一所有行长度并调整尺寸
        max_row_len = max(len(row) for row in read_rows)
        for i in range(len(read_rows)):
            row = read_rows[i]
            if len(row) < max_row_len:
                row.extend([(255, 255, 255)] * (max_row_len - len(row)))  # 空白用白色填充
        # 转换为图像并缩放
        img_array = np.array(read_rows, dtype=np.uint8)
        pil_img = Image.fromarray(img_array).resize(canvas_size, Image.NEAREST)
        black_img = np.array(pil_img)
    
    # 保存图像
    output_dir = os.path.join(img_output_dir, f"chr{chrom}_pics")
    os.makedirs(output_dir, exist_ok=True)
    img_name = f"pic_{start_pos//200}.png"
    Image.fromarray(black_img).save(os.path.join(output_dir, img_name))
    bam_handle.close()

def region_image_task(args):
    """多进程任务：处理单个区域文件生成图像"""
    bam_path, region_file, img_dir = args
    # 解析染色体名
    chrom = region_file.split("/")[-1].split(".")[0].split("_")[-1]
    # 处理每个窗口索引
    with open(region_file, 'r') as f:
        for idx_line in f:
            idx = idx_line.strip()
            if idx:
                win_start = 200 * int(idx)
                win_end = win_start + 200
                create_region_image(bam_path, (chrom, win_start, win_end), img_dir)

def parallel_image_generation(bam_path, region_dir, img_dir, processes=8):
    """多进程批量处理区域文件生成图像"""
    # 获取所有区域文件
    region_files = [
        os.path.join(region_dir, f) 
        for f in os.listdir(region_dir) 
        if os.path.isfile(os.path.join(region_dir, f))
    ]
    # 构建任务参数
    task_args = [(bam_path, f, img_dir) for f in region_files]
    # 多进程执行
    with multiprocessing.Pool(processes=processes) as pool:
        pool.map(region_image_task, task_args)

if __name__ == "__main__":
    # 输入输出路径配置
    bam_input = "HG002_PB_70x_RG_HP10XtrioRTG.bam"
    region_input_dir = "area_list"
    image_output_dir = "pics_new"
    # 启动多进程处理
    parallel_image_generation(bam_input, region_input_dir, image_output_dir, processes=8)
