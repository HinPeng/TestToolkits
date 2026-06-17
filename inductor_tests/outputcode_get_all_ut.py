import re
import sys

def process_python_file(input_file, output_dir, kernel):
    """
    处理Python文件，注释满足条件的行
    
    条件：
    1. 以triton_*.= async_compile.triton(开始的行到以''', device_str='npu')之间
    2. 该行中存在triton_*.run
    """
    
    # 编译正则表达式模式
    pattern_start = re.compile(r'^(\s*)triton_.*?=\s*async_compile\.triton\(')
    pattern_end = re.compile(r".*''', device_str='npu'\)")
    pattern_run = re.compile(r'triton_.*?\.run')

    # pattern_kernel_start = re.compile(r'^(\s*)triton_' + kernel + r'=\s*async_compile\.triton\(')
    pattern_kernel_name_run = re.compile(r'triton_' + kernel + r'\.run')
    
    in_comment_block = False
    processed_lines = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        original_line = line
        
        # 检查是否进入注释块开始
        if pattern_start.search(line) and kernel not in line:
            in_comment_block = True
        
        # 检查是否需要注释
        should_comment = False
        
        # 条件1：在注释块内
        if in_comment_block:
            should_comment = True
        
        # 条件2：包含triton_*.run
        if pattern_run.search(line) and not pattern_kernel_name_run.search(line):
            should_comment = True
        
        # 检查是否到达注释块结束
        if pattern_end.search(line) and in_comment_block:
            in_comment_block = False
            should_comment = True  # 结束行也需要注释
        
        # 如果需要注释且该行不是空行
        if should_comment and line.strip():
            # 保留缩进，在行首添加注释
            leading_spaces = len(line) - len(line.lstrip())
            indent = line[:leading_spaces]
            content = line[leading_spaces:]
            
            # 如果行已经有注释，保留原有注释
            if '#' in content and not content.strip().startswith('#'):
                # 分割代码和注释
                code_part, comment_part = content.split('#', 1)
                commented_line = f"{indent}# {code_part.rstrip()}  # {comment_part}"
            else:
                # 直接注释整行
                commented_line = f"{indent}# {content}"
            
            processed_lines.append(commented_line)
        else:
            processed_lines.append(original_line)
    
    output_file = f"{output_dir}/ut_{kernel}.py"
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)
    
    print(f"处理完成！已将结果保存到 {output_file}")

def get_kernel_names(input_file):
    """
    提取所有triton_*.run的kernel名称
    """
    pattern_run = re.compile(r'triton_(.*?)\.run')
    kernel_names = set()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            matches = pattern_run.findall(line)
            for match in matches:
                kernel_names.add(match.rstrip())
    
    return list(kernel_names)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用方法: python script.py <输入文件> <输出目录>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    try:
        kernel_names = get_kernel_names(input_file)
        print("提取的kernel名称:", kernel_names, "count: ", len(kernel_names))
        for kernel in kernel_names:
            process_python_file(input_file, output_dir, kernel)
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
