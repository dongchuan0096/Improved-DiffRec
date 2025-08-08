import numpy as np
import os

def analyze_npy_file(file_path):
    """分析.npy文件的内容"""
    print(f"\n分析文件: {os.path.basename(file_path)}")
    print("-" * 50)
    
    try:
        # 读取.npy文件
        data = np.load(file_path, allow_pickle=True)
        
        # 显示基本信息
        print(f"数组形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"数组维度: {data.ndim}")
        print(f"元素总数: {data.size:,}")
        print(f"内存占用: {data.nbytes / (1024*1024):.2f} MB")
        
        # 显示数据统计信息
        if np.issubdtype(data.dtype, np.number):  # 只对数值类型计算统计信息
            print(f"\n数值统计:")
            print(f"最小值: {np.min(data)}")
            print(f"最大值: {np.max(data)}")
            print(f"平均值: {np.mean(data)}")
            print(f"标准差: {np.std(data)}")
        
        # 显示数据预览
        print("\n数据预览:")
        if data.ndim == 1:
            print("前5个元素:")
            print(data[:5])
        elif data.ndim == 2:
            print("前3行:")
            print(data[:10])
            print(f"\n矩阵形状: {data.shape[0]} 行 x {data.shape[1]} 列")
        else:
            print(f"这是一个{data.ndim}维数组，显示第一个切片:")
            print(data[0])
        
        # 如果是结构化数组或对象数组，显示更多信息
        if data.dtype.names is not None:
            print("\n结构化数组字段:")
            for name in data.dtype.names:
                print(f"- {name}: {data[name].dtype}")
        
        # 检查是否包含缺失值
        if np.issubdtype(data.dtype, np.number):
            n_nan = np.isnan(data).sum()
            if n_nan > 0:
                print(f"\n包含 {n_nan:,} 个缺失值 (NaN)")
        
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

# 使用示例:
"""
# 读取单个.npy文件
data = np.load('文件名.npy', allow_pickle=True)

# 如果是压缩的.npz文件
with np.load('文件名.npz') as data:
    # .npz文件中可能包含多个数组
    for key in data.files:
        array = data[key]
        print(f"数组 {key}:")
        print(array)

# 保存数组到.npy文件
array = np.array([1, 2, 3, 4, 5])
np.save('新文件.npy', array)

# 保存多个数组到.npz文件
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
np.savez('多个数组.npz', a=array1, b=array2)
"""