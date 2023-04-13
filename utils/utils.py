import os
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['KaiTi']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

_COLOR_ = ['g', 'b', 'c', 'm', 'y', 'k', 'r']
_MARK_ = ['_', '.', '+', '_', '_']

def softmax(x):
    max = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x


def cas2att(cas_seq):
    cas_seq = cas_seq.squeeze()
    seq_len, cls_cnt = cas_seq.shape
    att = softmax(cas_seq.T).T
    return np.max(att[:, :-1], axis=1)


def cal_gt_ratio(gts_pdframe, seq_length):
    y_seq = np.zeros(seq_length)
    t_starts = gts_pdframe['t-start'].to_list()
    t_ends = gts_pdframe['t-end'].to_list()
    for s, e in zip(t_starts, t_ends):
        if s == e:
            y_seq[s] = 1
        else:
            y_seq[s:e] = 1
    gt_units = len(y_seq[y_seq > 0])
    return gt_units, gt_units/seq_length


def vis_gt_units(gt_units, gt_ratios, save_dir):
    x = range(len(gt_units))
    plt.figure()
    ax = plt.subplot(211)
    # ax.plot(x, np.sort(gt_units), 'o', 'r')
    ax.scatter(x, np.sort(gt_units))
    ax2 = plt.subplot(212)
    # ax2.plot(x, np.sort(gt_ratios), 'o','r')
    ax2.scatter(x, np.sort(gt_ratios))
    plt.savefig(os.path.join(save_dir, 'all_gt.png'))
    plt.close()


def vis_att(gts_pdframe, idx_lst, attn_seq, attn_seq_cas, proposals, save_dir, map_value=-1):
    y_positive_value = 0.8
    y_positive_shift_value = 0.05
    p_positive_value = -1
    p_positive_shift_value = 1
    seq_length = attn_seq.shape[0]
    x = list(range(seq_length))
    plt.figure()
    ax = plt.subplot(211)
    ax.plot(x, attn_seq, 'r' + '-')
    ax.plot(x, attn_seq_cas, 'purple')
    ax.plot(x, attn_seq * attn_seq_cas, 'k')
    # plt.text(x=2.2,#文本x轴坐标 
    #         y=8, #文本y轴坐标
    #         s='basic unility of text')
    # plt.set_color('r')#修改文字颜色    
    # ax.text(40, 30, "北京", fontsize=12, color = "r")
    if isinstance(idx_lst, list):
        ax.plot(idx_lst[0], attn_seq_cas[idx_lst[0]], _COLOR_[0]+'^')
        ax.plot(idx_lst[1], attn_seq_cas[idx_lst[1]], _COLOR_[1]+'*')
    else:
        ax.plot(idx_lst, attn_seq_cas[idx_lst], _COLOR_[0] + '^')
    
    label = list(set(gts_pdframe['label']))
    for idx, l in enumerate(label):
        # draw gts
        y_seq = np.zeros(seq_length)
        t_starts = gts_pdframe[gts_pdframe['label'] == l]['t-start'].to_list()
        t_ends = gts_pdframe[gts_pdframe['label'] == l]['t-end'].to_list()
        for s, e in zip(t_starts, t_ends):
            if s == e:
                y_seq[s] = y_positive_value
            else:
                y_seq[s:e] = y_positive_value
        ax.plot(x, y_seq, _COLOR_[idx]+_MARK_[idx])
        y_positive_value += y_positive_shift_value
        # draw proposals
        p_seq = np.zeros(seq_length)
        ax2 = plt.subplot(212)
        p_starts = proposals[proposals['label'] == l]['t-start'].to_list()[:]
        p_ends = proposals[proposals['label'] == l]['t-end'].to_list()[:]
        score = proposals[proposals['label'] == l]['score'].to_list()
        use_p_color_inpredict = False
        if len(p_starts) == 0:
            # there are no predictions of the ground truth label
            # use the predicted label instead
            p_label = list(set(proposals['label'].to_list()))[0]
            use_p_color_inpredict = True
            p_starts = proposals[proposals['label'] == p_label]['t-start'].to_list()[:]
            p_ends = proposals[proposals['label'] == p_label]['t-end'].to_list()[:]
            score = proposals[proposals['label'] == p_label]['score'].to_list()

        for s, e, score_ in zip(p_starts, p_ends, score):
            s, e = int(s), int(e)
            if s == e:
                p_seq[s] = p_positive_value
            else:
                p_seq[s:e] = p_positive_value

            if use_p_color_inpredict:
                ax2.plot(x, p_seq, _COLOR_[-1]+_MARK_[-1])
            else:
                ax2.plot(x, p_seq, _COLOR_[idx]+_MARK_[idx])
            # ax2.text(2,1.3,'x',fontsize=10, color='green')
            p_positive_value -= p_positive_shift_value

    v_name = gts_pdframe['video-id'].to_list()[0]
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'all'), exist_ok=True)
    if map_value !=-1:
        plt.savefig(os.path.join(save_dir, 'all', str(map_value) + '=map@0.5_' + v_name + '.png'))
    else:
        plt.savefig(os.path.join(save_dir, 'all', v_name + '.png'))
    # save another copy
    dir_name = str(len(label))
    os.makedirs(os.path.join(save_dir, dir_name), exist_ok=True)
    if map_value !=-1:
        plt.savefig(os.path.join(save_dir, dir_name, str(map_value) + '=map@0.5_' + v_name + '.png'))
    else:
        plt.savefig(os.path.join(save_dir, dir_name, v_name + '.png'))
    # plt.show()
    plt.close()