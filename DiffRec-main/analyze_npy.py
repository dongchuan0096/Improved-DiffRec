import numpy as np
import os

def analyze_npy_file(file_path):
    """分析.npy文件的内容"""
    print(f"\n分析文件: {os.path.basename(file_path)}")
    print("-" * 50)
    
    try:
        # 读取.npy文件
        data = np.load(file_path, allow_pickle=True)
        
        # 显示数组维度
        print(f"数组维度: {data.ndim}")
        print(f"数组形状: {data.shape}")
        
        # 显示第一行内容
        print("\n第一行内容:")
        if data.ndim == 1:
            print(data[0])
        elif data.ndim == 2:
            print(data[0])
        else:
            print(f"这是一个{data.ndim}维数组，显示第一个切片:")
            print(data[0])
        
    except Exception as e:
        print(f"无法读取文件: {str(e)}")

def main():
    # 指定要分析的目录
    data_dir = "."  # 当前目录，根据实际情况修改
    
    # 获取所有.npy文件
    npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    if not npy_files:
        print("当前目录下没有找到.npy文件")
        return
    
    print(f"找到 {len(npy_files)} 个.npy文件")
    
    # 分析每个文件
    for npy_file in npy_files:
        file_path = os.path.join(data_dir, npy_file)
        analyze_npy_file(file_path)

if __name__ == "__main__":
    main() 