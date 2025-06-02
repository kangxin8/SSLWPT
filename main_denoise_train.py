'''
2025.3.3 by kx
'''
import logging
import argparse
from datetime import datetime  #
import os
# from train_object.train_spec_sparse_LWPT import Train
from train_object.train_modiified_spec_sparse_LWPT import Train


# import matplotlib
# matplotlib.use('TkAgg')


def setlogger(path):
    logger = logging.getLogger()  # 创建记录器，提供接口
    logger.setLevel(logging.INFO)  # 决定日志记录的级别
    logFormatter = logging.Formatter("%(message)s", "%S")  # 设置日志内容的格式,以及时间的格式
    # 设置日志处理器
    fileHandler = logging.FileHandler(path)  # 文件记录类型处理器
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()  # 屏幕输出类型记录器
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)


# 训练超参数
def parse_args():
    parser = argparse.ArgumentParser()  # 实例化参数解析器
    # 训练的数据集
    parser.add_argument('--task_Descrip', type=str, default='denoising', help='the task description')  # 跨机械？跨工况？降噪？
    parser.add_argument('--model_name', type=str, default='LWPT', help='the name of trained model')  # 训练好的模型和训练过程中的日志的存储路径
    parser.add_argument('--source_dataset', type=str, default=['HNU'], help='the source domain dataset')  # 源域数据集名称'Simulate'
    parser.add_argument('--target_dataset', type=str, default=['PHM2024'], help='the target domain dataset')  # 目标域数据集名称
    # parser.add_argument('--source_sub_dataset', type=str, default=['sine_sample_list'], help='the source domain dataset')  # 源域子集名称，对应相应数据集对象的方法名称
    parser.add_argument('--source_sub_dataset', type=str, default=['HNU_single_fault_condition_samplelist_obtain'], help='the source domain dataset')  # 源域子集名称，对应相应数据集对象的方法名称
    parser.add_argument('--target_sub_dataset', type=str, default=['PHM2024_for_test_normal', 'PHM2024_for_test_BO',
                                                                   'PHM2024_for_test_BI', 'PHM2024_for_test_GC',
                                                                   'PHM2024_for_test_GW',
                                                                   'PHM2024_for_test_GM', 'PHM2024_for_test_GB'], help='the source domain dataset')  # 目标域子集名称，对应相应数据集对象的方法名称
    # parser.add_argument('--file_name', type=str, default=['cllw40-zczc-0-1500rpm-0-4N-20480Hz-1'], help='the target file')  # 目标文件
    parser.add_argument('--sample_len', type=int, default=4096, help='the len of single simple')  # 样本的长度
    parser.add_argument('--sample_rate', type=int, default=20480, help='the len of single simple')  # 样本的长度
    parser.add_argument('--random_seed', type=int, default=96, help='the len of single simple')  #  训练集、验证集划分的随机数种子
    parser.add_argument('--cut_off', type=int, default=None, help='the number of samples in single file')  # 单个文件取样本数，4096样本长度，单个文件最多取60个样本
    parser.add_argument('--preprocess', type=bool, default=False, help='whether preprocess')  # 是否执行降噪预处理？
    parser.add_argument('--tsne_preprocess', type=bool, default=False, help='whether preprocess')  # 是否需要tsne所需的数据进行预处理操作

    # 模拟信号数据集
    parser.add_argument('--number_samples', type=int, default=100, help='the number of samples')  #  训练集、验证集划分的随机数种子
    parser.add_argument('--SNR', type=int, default=0, help='the SNR (dB)')  #  噪声等级


    # 模型结构
    # 可学习离散小波降噪、可学习的离散小波包变换、谱稀疏驱动的离散小波包变换


    # 训练过程超参数
    parser.add_argument('--batch_size', type=int, default=10, help='the batch_size of dataloader')
    parser.add_argument('--max_epoch', type=int, default=1001)
    parser.add_argument('--check_epoch_for_test', type=int, default=50, help='the test position')  # 每隔50代获取测试集信息
    parser.add_argument('--check_epoch_for_vis', type=int, default=200, help='the test position')  # 每隔200代获取测试集可视化信息，检查降噪效果
    parser.add_argument('--check_epoch_for_save', type=int, default=200, help='the test position')  # 每隔50代获取测试集信息
    parser.add_argument('--epoch_for_stop', type=int, default=700, help='the test position')  # 从700代开始

    # 湖南大学齿轮数据集参数
    parser.add_argument('--HNU_fs', type=int, default=20480, help='fs')  # 训练集、验证集划分的随机数种子
    parser.add_argument('--HNU_index', type=int, default=2)  # 选择不同方向的振动信号
    parser.add_argument('--root_dir', type=str, default=r'D:\F\Data_hub\HNU齿轮\弧齿锥齿轮箱数据（608项目）\齿轮裂纹数据', help='root')  # 训练集、验证集划分的随机数种子
    parser.add_argument('--HNU_fault_root_dir', type=str, default=r'D:\F\Data_hub\HNU齿轮\弧齿锥齿轮箱数据（608项目）\齿轮裂纹数据\clzc-zczc', help='root')  # 训练集、验证集划分的随机数种子
    parser.add_argument('--HNU_file_name', type=str, default='clqk-zczc-1200rpm-0N-20480Hz', help='the target file')  # 指定要读取的故障类型和工况
    parser.add_argument('--HNU_add_noise', type=bool, default=False, help='whether add noise')  # True表示注入噪声

    # PU轴承数据集参数
    parser.add_argument('--PU_dir', type=str, default='K005', help='data_dir')  # 指定数据集的故障类别‘K005’,'KA01','KI01'
    parser.add_argument('--PU_condition', type=str, default='N09_M07_F10', help='condition')  # 指定数据集的工况'N09_M07_F10','N15_M01_F10','N15_M07_F04','N15_M07_F10'
    parser.add_argument('--PU_add_noise', type=bool, default=False, help='whether add noise')  # True表示注入噪声
    parser.add_argument('--PU_denoise', type=bool, default=False, help='whether use digital label')  # True表示训练降噪模型，使用干净信号作为标签
    parser.add_argument('--PU_label', type=int, default=0, help='the label of current data')  # 当前数据集对应的标签
    parser.add_argument('--PU_signal_process_method', type=str, default='original_1d', help='the method to construct sample')  # 获取样本的方式


    # 损失函数参数
    parser.add_argument('--sparse_weight', type=int, default=1.05)  # 稀疏损失的权重0.01，  1

    # 模型存储路径
    parser.add_argument('--information_save_dir_path', type=str, default=r'D:\OneDrive - hnu.edu.cn\项目\project_code\paper5_spec_sparse_wavelet_denoise\train_log_information\数值实验_泛化性验证\HNU\LWPT_HNU', help='the directory to save the model and training log')  # 训练好的模型和训练过程中的日志的存储路径

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # 循环执行
    args = parse_args()  # 训练全过程的控制参数字典
    # dir_list = ['D:\F\Data_hub\HNU齿轮\弧齿锥齿轮箱数据（608项目）\齿轮裂纹数据\cllw20-zczc', 'D:\F\Data_hub\HNU齿轮\弧齿锥齿轮箱数据（608项目）\齿轮裂纹数据\cllw30-zczc',
    #             'D:\F\Data_hub\HNU齿轮\弧齿锥齿轮箱数据（608项目）\齿轮裂纹数据\cllw40-zczc','D:\F\Data_hub\HNU齿轮\弧齿锥齿轮箱数据（608项目）\齿轮裂纹数据\cllw50-zczc',
    #             ]  # 数据集文件夹列表
    dir_list = ['D:\F\Data_hub\HNU齿轮\弧齿锥齿轮箱数据（608项目）\齿轮裂纹数据\clzc-zczc' ]  # 数据集文件夹列表
    # condition_list = ['cllw50-zczc-600rpm-0N-20480Hz','cllw50-zczc-600rpm-4N-20480Hz',
    #                   'cllw50-zczc-900rpm-0N-20480Hz','cllw50-zczc-900rpm-4N-20480Hz',
    #                   'cllw50-zczc-1200rpm-0N-20480Hz', 'cllw50-zczc-1200rpm-4N-20480Hz',
    #                   'cllw50-zczc-1500rpm-0N-20480Hz', 'cllw50-zczc-1500rpm-4N-20480Hz']  # 工况列表
    condition_list = ['clzc-zczc-1200rpm-0N-20480Hz', 'clzc-zczc-1200rpm-4N-20480Hz',
                      'clzc-zczc-1500rpm-0N-20480Hz', 'clzc-zczc-1500rpm-4N-20480Hz']  # 工况列表
    for dir in dir_list:
        args.HNU_fault_root_dir = dir
        for condition in condition_list:
            args.HNU_file_name = condition
            # 存储位置的设置
            sub_dir = args.HNU_file_name + '_' + datetime.strftime(datetime.now(),'%m%d-%H%M%S')
            save_dir = os.path.join(args.information_save_dir_path, sub_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 日志记录器对象
            setlogger(os.path.join(save_dir, 'training.log'))
            for k, v in args.__dict__.items():
                logging.info("the train args")
                logging.info("{}: {}".format(k, v))
            trainer = Train(args, save_dir)
            trainer.train_prepare()
            trainer.train()
    print('1')