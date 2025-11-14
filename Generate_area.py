import itertools
import os
import pysam
from collections import defaultdict
import multiprocessing
from collections import Counter

def decode_flag(Flag):
    signal = {1 << 2: 0, 1 >> 1: 1, 1 << 4: 2, 1 << 11: 3, 1 << 4 | 1 << 11: 4}
    return signal[Flag] if(Flag in signal) else 0

def c_pos(cigar, refstart):
    number = ''
    numlist = [str(i) for i in range(10)]
    readstart = False
    readend = False
    refend = False
    readloc = 0
    refloc = refstart
    for c in cigar:
        if(c in numlist):
            number += c
        else:
            number = int(number)
            if(readstart == False and c in ['M', 'I', '=', 'X']):
                readstart = readloc
            if(readstart != False and c in ['H', 'S']):
                readend = readloc
                refend = refloc
                break

            if(c in ['M', 'I', 'S', '=', 'X']):
                readloc += number

            if(c in ['M', 'D', 'N', '=', 'X']):
                refloc += number
            number = ''
    if(readend == False):
        readend = readloc
        refend = refloc

    return refstart, refend, readstart, readend 

def parse_cigar(cigar, ref_start):
    """
    解析CIGAR字符串，提取长度超过30bp的插入（I）和缺失（D）作为候选SV。
    """
    operations = []
    num_temp = ""
    for char in cigar:
        if char.isdigit():
            num_temp += char
        else:
            if num_temp:
                operations.append((int(num_temp), char))
                num_temp = ""
    if num_temp:
        operations.append((int(num_temp), char))
    
    sv_candidates = []
    position = ref_start
    for op in operations:
        if op[1] == 'M':
            position += op[0]
        elif op[1] == 'I' and op[0] > 30:
            sv_candidates.append(('INS', position, position + op[0], op[0]))
        elif op[1] == 'D' and op[0] > 30:
            sv_candidates.append(('DEL', position, position + op[0], op[0]))
            position += op[0]
    return sv_candidates

def detect_split_reads(read):
    sv_candidates = []
    if(read.has_tag('SA') == True):
        code = decode_flag(read.flag)
        rawsalist = read.get_tag('SA').split(';')
        for sa in rawsalist[:-1]:
            sainfo = sa.split(',')
            tmpcontig, tmprefstart, strand, cigar = sainfo[0], int(sainfo[1]), sainfo[2], sainfo[3]
            if(tmpcontig != read.reference_name):
                continue 
            if((strand == '-' and (code %2) ==0) or (strand == '+' and (code %2) ==1)):
                refstart_1, refend_1, readstart_1, readend_1 =  read.reference_start, read.reference_end,read.query_alignment_start,read.query_alignment_end
                refstart_2, refend_2, readstart_2, readend_2 = c_pos(cigar, tmprefstart)
                read_distance = readend_1 - readstart_2
                ref_distance = refend_1 - refstart_2
                sv_len = abs(ref_distance - read_distance)
                if(abs(ref_distance-read_distance)<30):
                    continue
                if sv_len > 50 :
                        sv_type = 'DEL' if ref_distance > read_distance else 'INS'
                        sv_candidates.append((sv_type, refstart_2-100,refstart_2+100, 200))
                        sv_candidates.append((sv_type, refend_1-100,refend_1+100, 200))
    return sv_candidates

def create_area2(sv_list,path):
    """
    根据sv_list中类型为'INS'的SV，生成区域索引。
    """
    index_counter = Counter()
    for sv in sv_list:
        if sv[3] == 'INS':
            start = int(sv[1] // 200)
            end = int(sv[2] // 200)
            for idx in range(start, end + 1):
                index_counter[idx] += 1
    
    os.makedirs(path, exist_ok=True)
    area_list = [idx for idx, count in index_counter.items() if count > 3]
    # area_list = [idx for idx, count in index_counter.items() if count > 1]
    
    # 对每个区域索引，添加其左右各一个索引，并去重
    extended_area_set = set(area_list)
    for idx in area_list:
        extended_area_set.add(idx - 1)
        extended_area_set.add(idx + 1)
    area_list = sorted(extended_area_set)
    
    with open(os.path.join(path,f"area_list_{sv[0]}.txt"), "w") as f:
        for area in area_list:
            f.write(f"{area}\n")
    return area_list

def create_area(sv_list,path):
    """
    根据sv_list中类型为'INS'的SV，生成区域索引。
    """
    index_counter = Counter()
    for sv in sv_list:
        if sv[3] == 'INS':
            start = int(sv[1] // 200)
            end = int(sv[2] // 200)
            for idx in range(start, end + 1):
                index_counter[idx] += 1
    
    os.makedirs(path, exist_ok=True)
    # area_list = [idx for idx, count in index_counter.items() if count > 3]
    # area_list = [idx for idx, count in index_counter.items() if count > 2]
    area_list = [idx for idx, count in index_counter.items() if count > 1]


    with open(os.path.join(path,f"area_list_{sv[0]}.txt"), "w") as f:
        for area in area_list:
            f.write(f"{area}\n")
    return area_list


def analyze_chromosome(args):
    """
    分析BAM文件中的单条染色体以检测候选SV区域。
    """
    bam_file, chromosome, output_dir,area_path = args
    bamfile = pysam.AlignmentFile(bam_file, "rb")
    sv_candidates = []

    for read in bamfile.fetch(chromosome):
        if read.mapping_quality > 20:
            # 从CIGAR字符串检测SV
            cigar_svs = parse_cigar(read.cigarstring, read.reference_start)
            for sv in cigar_svs:
                sv_candidates.append((chromosome, sv[1], sv[2], sv[0])) # 筛选出长度超过30bp且质量大于20的SV
            

            # 从分裂读段检测SV
            split_svs = detect_split_reads(read)
            for sv in split_svs:
                sv_candidates.append((chromosome, sv[1], sv[2], sv[0]))

    sv_candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    # 为当前染色体创建单独的输出文件
    output_file = os.path.join(output_dir, f"candidate_sv_regions_{chromosome}.txt")
    with open(output_file, "w") as f:
        f.write("染色体\t起始位置\t结束位置\t类型\n")
        for region in sv_candidates:
            f.write(f"{region[0]}\t{region[1]}\t{region[2]}\t{region[3]}\n")
            
    create_area(sv_candidates, area_path)
    
    # 合并候选SV为区域
    # sv_regions = merge_svs(sv_candidates)
    
    # 为当前染色体创建单独的输出文件
    # output_file = os.path.join(output_dir, f"candidate_sv_regions_{chromosome}.txt")
    # with open(output_file, "w") as f:
    #     f.write("染色体\t起始位置\t结束位置\t类型\t支持读段数\n")
    #     for region in sv_regions:
    #         f.write(f"{region[0]}\t{region[1]}\t{region[2]}\t{region[3]}\t{region[4]}\n")
    
    # return sv_regions

def analyze_bam_file_parallel(bam_file, output_dir, area_path, num_processes=1):
    """
    并行分析BAM文件以检测候选SV区域。
    """
    chromosomes = [f"{i}" for i in range(1, 23)]
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 使用多进程处理每个染色体
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(analyze_chromosome, 
                           [(bam_file, chr_name, output_dir ,area_path) for chr_name in chromosomes])
        
    # 调试
    # analyze_chromosome((bam_file, "1", output_dir))


if __name__ == "__main__":
    bam_file = "/ifs/laicx/00.data/HG002_PB_70x_RG_HP10XtrioRTG.bam"  
    num_processes = 15
    sv_output_dir = "/home/zhangjj/01.code/06.Mul-Graph/sv_results2.0"  # 指定输出目录
    area_path = "/home/zhangjj/01.code/06.Mul-Graph/area_list2.0"
    
    # 分析BAM文件并保存每个染色体的结果
    analyze_bam_file_parallel(bam_file, sv_output_dir, area_path, num_processes)
    
    # # 可选：将所有结果合并到一个文件中
    # combined_output = os.path.join(output_dir, "combined_candidate_sv_regions.txt")
    # with open(combined_output, "w") as f:
    #     f.write("染色体\t起始位置\t结束位置\t类型\t支持读段数\n")
    #     for region in candidate_sv_regions:
    #         f.write(f"{region[0]}\t{region[1]}\t{region[2]}\t{region[3]}\t{region[4]}\n")
    
    # #demo
    # output_dir = "/home/zhangjj/01.code/06.Mul-Graph/demo_chromosome_sv_results"  # 指定输出目录
    # analyze_chromosome((bam_file, "15", output_dir))
