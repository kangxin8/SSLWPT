'''
2025.3.21 by kx
'''
import os
import torch
from torch.optim.lr_scheduler import StepLR
import logging
import warnings
from Dataloader_prepare.train_dataloader_obtain_new import Dataloader_obtain
# from model_backbone.spec_sparse_LWPT import S_LWPT
from model_backbone.modified_spec_sparse_LWPT import MS_LWPT
from signal_process_obej.signal_1d_transform_and_analysis import Signal_1d_Processer
from signal_process_obej.signal_2d_transform_and_analysis import wavelet_scaterring_analysis, STFT_TF_analysis
import matplotlib
matplotlib.use("module://backend_interagg")  # 强制使用PyCharm工具窗口后端
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np


def inverse_l2_regularizer(params, epsilon=1e-6, reg_lambda=0.000001):
    """
    反向L2正则化项：惩罚参数过小。
    - params: 需要约束的参数列表
    - epsilon: 防止除以零的小常数
    - reg_lambda: 正则化强度
    """
    reg = 0.0
    for param in params:
        reg += reg_lambda * torch.sum(1.0 / (param.pow(2) + epsilon))
    return reg

class PlotAnalyzer(object):
    def __init__(self, args=None, am=None, x=None, y=None):
        self.am = am
        self.x = x
        self.y = y
        self.args = args
    # 使用imshow或者matshow可视化二维数组
    def visualize_2darray(self, title=None, method='i'):
        # 创建画布和坐标轴对象
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2), dpi=100)  # 双栏（3.2， 2.4）dpi=300
        # 绘制图形
        pcm = ax.imshow(np.abs(self.am), cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None)
        # pcm = ax.matshow(np.abs(self.am), cfignum=None, cmap=None, norm=None, aspect=None)
        # 设置图形
        ax.set_title(title)
        # fig.colorbar(pcm, ax=ax)
        plt.tight_layout()
        # 建立游标
        def format_coord(x, y):
            if method == 'p':
                col = int(x)
                row = int(y)
            else:  # for 'i' and 'm'
                col = int(x + 0.5)
                row = int(y + 0.5)
            if 0 <= row < self.am.shape[0] and 0 <= col < self.am.shape[1]:
                z = np.abs(self.am)[row, col]
                return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz, Intensity: {z:.2f}'
            else:
                return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz'
        ax.format_coord = format_coord
        plt.show()
        return fig

    # 使用pcolor可视化stft获取的时频图（使用插值）
    def visualize_stft_tf(self, amp, t, f, title=None, method='p', save=False):
        # 创建画布和坐标轴对象
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.4), dpi=300)  # 双栏（3.2， 2.4）dpi=300
        # 绘制图形
        pcm = ax.pcolormesh(t, f, np.abs(amp), cmap='viridis', norm=None, vmin=None, vmax=None,
                            shading=None, alpha=None)  # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        # ax.axis('off')  # 取消刻度
        # 设置图形
        # ax.set_xlabel('Time [sec]')
        # ax.set_ylabel('Frequency [Hz]')
        # ax.set_title(title)
        # # fig.colorbar(pcm, ax=ax)
        # plt.tight_layout()

        # # 建立游标
        # def format_coord(x, y):
        #     if method == 'p':
        #         col = int(x)
        #         row = int(y)
        #     else:  # for 'i' and 'm'
        #         col = int(x + 0.5)
        #         row = int(y + 0.5)
        #     if 0 <= row < self.am.shape[0] and 0 <= col < self.am.shape[1]:
        #         z = np.abs(self.am)[row, col]
        #         return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz, Intensity: {z:.2f}'
        #     else:
        #         return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz'
        #
        # ax.format_coord = format_coord
        plt.show()
        if save:
            file_name = 'from-visualize_stft_tf-{}-{}.png'.format(self.args.data_file_name, title)
            save_path = os.path.join(self.args.save_dir, file_name)
            fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)

        return fig
    # 获取scwt获取的时频图
    def visualize_scwt_tf(self, amp, t, f, title=None, method='p', save=False):
        # 创建画布和坐标轴对象
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.4), dpi=300)  # 双栏（3.2， 2.4）dpi=300
        # 绘制图形
        pcm = ax.pcolormesh(t, f, np.abs(amp), cmap='viridis', norm=None, vmin=None, vmax=None,
                            shading=None, alpha=None)  # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        # ax.axis('off')  # 取消刻度
        # 设置图形
        # ax.set_xlabel('Time [sec]')
        # ax.set_ylabel('Frequency [Hz]')
        # ax.set_title(title)
        # # fig.colorbar(pcm, ax=ax)
        # plt.tight_layout()

        # 建立游标
        # def format_coord(x, y):
        #     if method == 'p':
        #         col = int(x)
        #         row = int(y)
        #     else:  # for 'i' and 'm'
        #         col = int(x + 0.5)
        #         row = int(y + 0.5)
        #     if 0 <= row < self.am.shape[0] and 0 <= col < self.am.shape[1]:
        #         z = np.abs(self.am)[row, col]
        #         return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz, Intensity: {z:.2f}'
        #     else:
        #         return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz'
        #
        # ax.format_coord = format_coord
        plt.show()
        if save:
            file_name = 'from-visualize_scwt_tf-{}-{}.png'.format(self.args.data_file_name, title)
            save_path = os.path.join(self.args.save_dir, file_name)
            fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)

        return fig

    # 绘制频谱
    def plot_1d_signal_fft(self,amp=None, fre=None, save=False, title=None):
        # 消除0频分量
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 1.5), dpi=300)  # 画布设置,信号频谱和信号时域：(3.2,1.5),时频图（3.2，2.4）
        # self.am = self.am - np.mean(self.am)
        # ax.plot(fre, amp, color='black', linewidth=1.25)
        ax.plot(fre, amp, color='black', linewidth=0.5)
        # ax.axis('off')  # 取消刻度
        # 只显示纵轴的刻度和标签
        # ax.yaxis.set_visible(True)
        # ax.xaxis.set_visible(False)
        # 可选：设置纵轴刻度字体和大小
        # plt.setp(ax.get_xticklabels(), fontname='Times New Roman', fontsize=10)  # 设置字体的格式
        # plt.setp(ax.get_yticklabels(), fontname='Times New Roman', fontsize=10)  # 设置字体的格式
        # ax.set_xlim(min(self.x), max(self.x))  # 设置显示的范围
        # ax.set_ylim(min(self.am), max(self.am) + 0.01)
        # ax.xticks(fontsize=10, fontweight='bold')  # 设置坐标刻度，刻度字体大小为10，粗体，还可设置颜色等参数
        # ax.yticks(fontsize=10, fontweight='bold')  # 设置坐标刻度，刻度字体大小为10，粗体，还可设置颜色等参数
        # ax.spines['top'].set_color('none')  # 消除边框
        # ax.spines['right'].set_color('none')
        # ax.spines['left'].set_color('none')
        # ax.spines['bottom'].set_color('none')
        # plt.show(block=True)
        plt.show()
        plt.pause(0.01)
        if save:
            file_name = 'from-plot_1d_signal_fft-{}-{}.png'.format(self.args.data_file_name, title)
            save_path = os.path.join(self.args.save_dir, file_name)
            fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    # 绘制原始振动信号
    def plot_1d_original_signal(self,signal=None, t=None, save=False, title=None):
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 1.5), dpi=300)  # 画布设置
        ax.plot(t, signal, color='black', linewidth=0.7)
        # ax.axis('off')  # 取消刻度
        # 只显示纵轴的刻度和标签
        # ax.yaxis.set_visible(True)
        # ax.xaxis.set_visible(False)
        # 可选：设置纵轴刻度字体和大小
        # plt.setp(ax.get_xticklabels(), fontname='Times New Roman', fontsize=10)  # 设置字体的格式
        # plt.setp(ax.get_yticklabels(), fontname='Times New Roman', fontsize=10)  # 设置字体的格式
        # ax.set_xlim(min(self.x), max(self.x))  # 设置显示的范围
        # ax.set_ylim(min(self.y), max(self.y) + 0.0001)
        # ax.xticks(fontsize=10, fontweight='bold')  # 设置坐标刻度，刻度字体大小为10，粗体，还可设置颜色等参数
        # ax.yticks(fontsize=10, fontweight='bold')  # 设置坐标刻度，刻度字体大小为10，粗体，还可设置颜色等参数
        # ax.spines['top'].set_color('none')  # 消除边框
        # ax.spines['right'].set_color('none')
        # ax.spines['left'].set_color('none')
        # ax.spines['bottom'].set_color('none')
        plt.show()
        if save:
            file_name = 'from-plot_1d_original_signal-{}-{}.png'.format(self.args.data_file_name, title)
            save_path = os.path.join(self.args.save_dir, file_name)
            fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)

    # 绘制不同通道的权重
    def plot_1d_attention(self, amp=None, fre=None, save=False, title=None):
        # 消除0频分量
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 1.5), dpi=300)  # 画布设置,信号频谱和信号时域：(3.2,1.5),时频图（3.2，2.4）
        # self.am = self.am - np.mean(self.am)
        # ax.plot(fre, amp, color='black', linewidth=1.25)
        ax.bar(range(1, len(amp) + 1), amp,  color="skyblue", edgecolor="blue")
        ax.xticks(fontsize=5)
        ax.yticks(fontsize=5)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        fig.tight_layout()
        # ax.axis('off')  # 取消刻度
        # 只显示纵轴的刻度和标签
        # ax.yaxis.set_visible(True)
        # ax.xaxis.set_visible(False)
        # 可选：设置纵轴刻度字体和大小
        # plt.setp(ax.get_xticklabels(), fontname='Times New Roman', fontsize=10)  # 设置字体的格式
        # plt.setp(ax.get_yticklabels(), fontname='Times New Roman', fontsize=10)  # 设置字体的格式
        # ax.set_xlim(min(self.x), max(self.x))  # 设置显示的范围
        # ax.set_ylim(min(self.am), max(self.am) + 0.01)
        # ax.xticks(fontsize=10, fontweight='bold')  # 设置坐标刻度，刻度字体大小为10，粗体，还可设置颜色等参数
        # ax.yticks(fontsize=10, fontweight='bold')  # 设置坐标刻度，刻度字体大小为10，粗体，还可设置颜色等参数
        # ax.spines['top'].set_color('none')  # 消除边框
        # ax.spines['right'].set_color('none')
        # ax.spines['left'].set_color('none')
        # ax.spines['bottom'].set_color('none')
        plt.show()
        if save:
            file_name = 'from-plot_1d_signal_fft-{}-{}.png'.format(self.args.data_file_name, title)
            save_path = os.path.join(self.args.save_dir, file_name)
            fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)

    def img_save(self, img=None, file_path=None):

        pass
    def quick_visualize_2darray(self, array=None, title=None, method='i', save=False, figsize=None):
        # 创建画布和坐标轴对象
        fig, ax = plt.subplots(1, 1, figsize=figsize)  # 双栏（3.2， 2.4）dpi=300
        # 绘制图形
        pcm = ax.imshow(np.abs(array), cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None)
        # pcm = ax.matshow(np.abs(self.am), cfignum=None, cmap=None, norm=None, aspect=None)
        # 设置图形
        # ax.set_title(title)
        # fig.colorbar(pcm, ax=ax)
        plt.tight_layout()
        # 建立游标
        def format_coord(x, y):
            if method == 'p':
                col = int(x)
                row = int(y)
            else:  # for 'i' and 'm'
                col = int(x + 0.5)
                row = int(y + 0.5)
            if 0 <= row < self.am.shape[0] and 0 <= col < self.am.shape[1]:
                z = np.abs(self.am)[row, col]
                return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz, Intensity: {z:.2f}'
            else:
                return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz'
        ax.format_coord = format_coord
        plt.show()
        if save:
            file_name = 'from-quick_visualize_2darray-{}-{}.png'.format(self.args.data_file_name, title)
            save_path = os.path.join(self.args.save_dir, file_name)
            fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        return fig

    # 将原始矩形数组resize后可视化
    def visualize_resize_2darray(self, array=None, title=None, save=False, method='i', figsize=None):
        resize_array = resize(array, output_shape=(224, 224), anti_aliasing=True)  # 尺寸校准# 数组插值实现统一的尺寸
        # 创建画布和坐标轴对象
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.4), dpi=300)  # 双栏（3.2， 2.4）dpi=300
        # 绘制图形
        # pcm = ax.imshow(np.abs(resize_array), cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None,
        #                 vmax=None, origin=None, extent=None)
        pcm = ax.matshow(np.abs(resize_array))
        # 设置图形
        # ax.set_title(title)
        # fig.colorbar(pcm, ax=ax)
        plt.tight_layout()

        # 建立游标
        def format_coord(x, y):
            if method == 'p':
                col = int(x)
                row = int(y)
            else:  # for 'i' and 'm'
                col = int(x + 0.5)
                row = int(y + 0.5)
            if 0 <= row < self.am.shape[0] and 0 <= col < self.am.shape[1]:
                z = np.abs(self.am)[row, col]
                return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz, Intensity: {z:.2f}'
            else:
                return f'Time: {x:.2f}s, Frequency: {y:.2f}Hz'

        ax.format_coord = format_coord
        plt.show()
        if save:
            file_name = 'from-visualize_resize_2darray-{}-{}.png'.format(self.args.data_file_name, title)
            save_path = os.path.join(self.args.save_dir, file_name)
            fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        return fig


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        :param patience: 多少个 epoch 连续不下降才停止训练
        :param min_delta: 最小的损失下降幅度，如果小于这个值则认为没有改善
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # 重新计数
        else:
            self.counter += 1  # 连续不下降次数 +1

        return self.counter >= self.patience  # 如果达到 patience，返回 True 停止训练

class Train():
    def __init__(self, config, save_dir_path):
        self.config = config  # 控制训练过程的参数字典
        self.save_dir_path = save_dir_path  # 训练过程日志的存储路径




    def train_prepare(self):
        args = self.config
        # =======训练的硬件设备=========
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # ========训练集准备（单一类别数据集）======
        Dataloader_obtainer = Dataloader_obtain(args)  # 实例化dataloader获取器
        source_train_test_iterator_list = Dataloader_obtainer.dataloader_obtain()  # 获取多源域训练和测试的迭代器列表，每个元素表示来自同一个域的训练和测试对象元组
        self.train_loaders_src_list = [train for train, test in source_train_test_iterator_list]  # 训练集dataloader
        self.test_loaders_src_list = [test for train, test in source_train_test_iterator_list]  # 测试集dataloader

        # ========模型准备==========
        self.model = MS_LWPT(Input_Size=args.sample_len, Input_Level=7, Filt_Style='Filter_Free', Act_Train=True).to(self.device)

        # =====优化器与优化参数=======
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-7)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.7)
        # ========损失函数======
        self.loss_L1 = torch.nn.L1Loss()  # 重构损失
        # 谱稀疏损失
        # 稀疏损失



    def denoise_vis_and_val(self, model, loader_list, epoch):
        re_loss_log = [] # 每一批次的损失
        sparse_loss_log = [] # 每一批次的损失
        for i in range(len(loader_list)):
            # 迭代获取测试集损失
            loader = loader_list[i]
            for j, batched_data in enumerate(loader):
                x, _ = batched_data
                x = x.to(self.device).to(dtype=torch.float64)
                re_x, emb = model(x)  # 执行分解操作
                re_loss = self.loss_L1(re_x, x)
                sparse_loss = self.model.L1_sum(emb)
                re_loss_log.append(re_loss.detach().cpu().item())
                sparse_loss_log.append(sparse_loss.detach().cpu().item())



            # 判定是否要进行可视化
            if epoch % self.config.check_epoch_for_vis == 0:
                first_batch = next(iter(loader))
                orig_sanmple = first_batch[0][1].unsqueeze(0)
                re_x, _ = model(orig_sanmple.to(self.device).to(dtype=torch.float64))
                vis_ori_sample = orig_sanmple.squeeze().detach().cpu().numpy()  # 原始样本获取
                vis_denoise_sample = re_x.squeeze().detach().cpu().numpy()  # 降噪重构信号的获取
                Signal_Processer = Signal_1d_Processer(self.config)  # 信号1d处理器
                Signal_2d_Processer1 = wavelet_scaterring_analysis()  # scwt处理器
                Signal_2d_Processer2 = STFT_TF_analysis(fs=self.config.sample_rate, nperseg=256, noverlap=255)  # stft处理器
                Ploter = PlotAnalyzer(self.config)
                # 原始信号可视化数据（只绘制一次）
                if epoch == 200:
                    ori_fft_fre, ori_fft_amp = Signal_Processer.plain_fft_transform(vis_ori_sample, sample_rate=self.config.sample_rate)  # 原始信号频谱
                    ori_stft_y, ori_stft_x, ori_stft_amp = Signal_2d_Processer2.stft_results_obtain(vis_ori_sample)  # 原始信号时频图stft
                    ori_scwt_amp, ori_scwt_x, ori_scwt_y = Signal_2d_Processer1.scattering_result(vis_ori_sample, fs=self.config.sample_rate)  # 原始信号时频图scwt
                    Ploter.plot_1d_signal_fft(amp=ori_fft_amp, fre=ori_fft_fre, save=False, title='ori_fre')  # 可视化原始信号频谱
                    Ploter.visualize_stft_tf(ori_stft_amp, ori_stft_x, ori_stft_y, save=False, title='ori_stft')  # 可视化原始信号stft时频图
                    Ploter.visualize_scwt_tf(ori_scwt_amp, ori_scwt_x, ori_scwt_y, save=False, title='de_stft')  # 可视化原始信号scwt时频图
                # 降噪信号可视化数据
                denoise_fft_fre, denoise_fft_amp = Signal_Processer.plain_fft_transform(vis_denoise_sample, sample_rate=self.config.sample_rate)  # 降噪信号的频谱
                de_scwt_amp, de_scwt_x, de_scwt_y = Signal_2d_Processer1.scattering_result(vis_denoise_sample, fs=self.config.sample_rate)  # 降噪信号时频图scwt
                de_stft_y, de_stft_x, de_stft_amp = Signal_2d_Processer2.stft_results_obtain(vis_denoise_sample)  # 降噪信号时频图stft
                # 可视化
                Ploter.plot_1d_signal_fft(amp=denoise_fft_amp, fre=denoise_fft_fre, save=False, title='denoise_fre')  # 可视化降噪信号频谱
                Ploter.visualize_stft_tf(de_stft_amp, de_stft_x, de_stft_y, save=False, title='de_scwt')  # 可视化降噪信号stft时频图
                Ploter.visualize_scwt_tf(de_scwt_amp, de_scwt_x, de_scwt_y, save=False, title='ori_scwt')  # 可视化降噪信号scwt时频图

        return np.mean(re_loss_log), np.mean(sparse_loss_log)



    def train(self):
        args = self.config

        # =======迭代训练=======
        for epoch in range(args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)  # 输出当前训练进度

            # ======生成批次数据=======
            batch_sample = [dataloader_obj.get_next_batch() for dataloader_obj in self.train_loaders_src_list]
            inputs, _ = zip(*batch_sample)
            inputs = torch.cat(inputs).to(self.device).to(dtype=torch.float64)
            # labels = torch.tensor(torch.cat(labels)).to(self.device)

            # =====模型前向传播=====
            self.model.train()
            re_x, emb = self.model(inputs)  # 执行分解操作
            re_loss = self.loss_L1(re_x, inputs)  # 重构损失
            sparse_loss = self.model.L1_sum(emb)  # 稀疏损失
            para_loss = inverse_l2_regularizer(self.model.Filt.kernel)  # 参数惩罚项
            loss = re_loss + args.sparse_weight*sparse_loss + para_loss  # 损失计算

            # =====误差反向传播和参数更新========
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # =======输出当前的训练状态（损失大小)、滤波核参数信息========
            logging.info('re_loss:{},sparse_loss:{}'.format(re_loss.item(), sparse_loss.item()))
            total_abs_mean = 0.0  # 滤波核参数绝对值之和
            for idx, param in enumerate(self.model.Filt.kernel):
                abs_values = param.data.abs()
                mean_abs = abs_values.mean().item()
                max_abs = abs_values.max().item()
                min_abs = abs_values.min().item()
                # print(f"  Kernel {idx}: Mean={mean_abs:.4f}, Max={max_abs:.4f}, Min={min_abs:.4f}")
                total_abs_mean += mean_abs
            avg_abs_mean = total_abs_mean / len(self.model.Filt.kernel)
            logging.info('Average Mean Absolute Value:{}'.format(avg_abs_mean))

            # =======指定的节点展示降噪的可视化结果以及测试集损失记录=======
            if epoch % args.check_epoch_for_test == 0 and epoch != 0:
                logging.info('epoch:{},check_for_test'.format(epoch))
                with torch.no_grad():
                    self.model.eval()
                    re_loss_log, sparse_loss_log = self.denoise_vis_and_val(self.model, self.test_loaders_src_list, epoch)
                    logging.info('re_loss_mean:{},sparse_loss_mean: {}'.format(re_loss_log, sparse_loss_log))

            # =======早停========
            if epoch  > args.epoch_for_stop:
                early_stopping1 = EarlyStopping(patience=5, min_delta=1e-4)
                early_stopping2 = EarlyStopping(patience=5, min_delta=1e-4)
                if early_stopping1.step(re_loss_log) and early_stopping2.step(re_loss_log):
                    logging.info(f"Early stopping at epoch {epoch}")
                    model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    torch.save(model_state_dic,os.path.join(self.save_dir_path,'{}-best_model.pth'.format(epoch)))
                    logging.info("save best model epoch {}".format(epoch))
                    break  # 停止训练


            # 模型参数保存
            if epoch % 200 == 0 and epoch != 0:
                model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                # acc_base = acc_target
                logging.info("save best model epoch {}".format(epoch))
                torch.save(model_state_dic,
                           os.path.join(self.save_dir_path,
                                        '{}-best_model.pth'.format(epoch)))


            # 参数更新策略
            self.scheduler.step()
            print(f"Epoch {epoch}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
