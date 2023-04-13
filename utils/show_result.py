import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import options
import numpy as np
from collections import Counter
from eval.eval_detection import ANETdetection
from utils.utils import *
args = options.parser.parse_args()

npy_file = 'C:\\Users\\Administrator\\Desktop\\default_testout.npy'
save_dir = 'C:\\Users\\Administrator\\Desktop\\atten_vis\\bad'
threshold_map = 20

# load results
results_dict = np.load(npy_file, allow_pickle=True).item()
video_names = list(results_dict.keys())
print('video numbers: {}'.format(len(video_names)))
# proposals format:--(31, 5) video-id  t-start  t-end  label  score
all_proposals = [results_dict[i]['proposals'] for i in video_names]

# evaluate result
iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dmap_detect = ANETdetection(
    'C:\\Users\\Administrator\\Desktop\\dev\\ECCV2022-DELU\\Thumos14reduced-Annotations', iou, args=args)
all_proposals = pd.concat(all_proposals).reset_index(drop=True)
dmap_detect.prediction = all_proposals
do_eval = False
if do_eval:
    dmap = dmap_detect.evaluate()
    print('||'.join(['map @ {} = {:.3f} '.format(
        iou[i], dmap[i] * 100) for i in range(len(iou))]))

# show results
dataset_gt = dmap_detect.ground_truth
# gt format:--video-id  t-start  t-end  label
dataset_gt = dataset_gt.groupby("video-id")
gt_units, gt_ratios = [], []
map_one_halfs = []
wrong_predicted_label_vids = []
for i in tqdm(range(len(video_names))):
    vid_name = video_names[i]
    one_video_result = results_dict[vid_name]
    cas = one_video_result['cas']  # (1, 223, 21)
    attn = one_video_result['attn']  # (1, 223, 1)
    supp_cas = cas * attn
    rat = 7
    topk_val, topk_ind = torch.topk(
        torch.Tensor(supp_cas),
        k=max(1, int(supp_cas.shape[-2] // rat)),
        dim=-2)
    # print('topk_ind.shape:{}'.format(topk_ind.shape))
    # topk_ind.shape:torch.Size([1, 23, 21])
    softmax_cas = one_video_result['softmax_cas']  # (21,)
    idx_lst = topk_ind[0, :, np.argmax(softmax_cas)].numpy()
    idx_lst2 = topk_ind[0, :, np.argsort(softmax_cas)[-2]].numpy()
    labels = one_video_result['labels']  # (20,)
    proposals = one_video_result['proposals']
    # (31, 5) video-id  t-start  t-end  label  score
    str_vid_name = vid_name.decode('utf-8')
    gts = dataset_gt.get_group(str_vid_name)
    label_g = gts['label'].to_list()[:]
    label_p = proposals['label'].to_list()[:]    
    inter_label = [i for i in label_p if i in label_g]
    if len(list(set(label_g))) == 1:
        draw_topk = idx_lst
    else:
        draw_topk = [idx_lst, idx_lst2]
    if len(inter_label) == 0:
        wrong_predicted_label_vids.append([str_vid_name, label_g, label_p])
    # if str_vid_name in ['video_test_0000242', 'video_test_0001118']:
    #     print('gts:{}'.format(gts))
    #     print('proposals:{}'.format(proposals))
    #     continue
    # else:
    #     continue
    gt_unit, gt_ratio = cal_gt_ratio(gts, attn.shape[1])
    gt_units.append(gt_unit)
    gt_ratios.append(gt_ratio)
    
    video_dmap = dmap_detect.evaluate_single_video(str_vid_name)
    map_one_half = video_dmap[4] * 100
    map_one_halfs.append(map_one_half)
    if map_one_half < threshold_map:
        print('map_one_half:{}'.format(map_one_half))
        vis_att(gts, draw_topk, attn.squeeze(), cas2att(cas), \
            proposals, save_dir, map_one_half)

print('mean map_one_halfs:{}'.format(np.mean(map_one_halfs)))
print('min map_one_halfs:{}'.format(np.min(map_one_halfs)))
print('max map_one_halfs:{}'.format(np.max(map_one_halfs)))
print('median map_one_halfs:{}'.format(np.median(map_one_halfs)))

vis_gt_units(gt_units, gt_ratios, save_dir)
print('mean ratio:{}'.format(np.mean(gt_ratios)))
print('min ratio:{}'.format(np.min(gt_ratios)))
print('max ratio:{}'.format(np.max(gt_ratios)))
print('median ratio:{}'.format(np.median(gt_ratios)))
gt_ratios = np.array(gt_ratios)
less_half = len(gt_ratios[gt_ratios < 0.5]) / len(gt_ratios)
print('less_half:{}'.format(less_half))
less_quarter = len(gt_ratios[gt_ratios < 0.25]) / len(gt_ratios)
print('less_quarter:{}'.format(less_quarter))
less_17 = len(gt_ratios[gt_ratios < 1/7]) / len(gt_ratios)
print('less_17:{}'.format(less_17))

print('mean gt units:{}'.format(np.mean(gt_units)))
print('min gt units:{}'.format(np.min(gt_units)))
print('max gt units:{}'.format(np.max(gt_units)))
print('median gt units:{}'.format(np.median(gt_units)))


do_change_predict_label_eval = True
if do_change_predict_label_eval:
    dmap = dmap_detect.evaluate()
    print('||'.join(['map @ {} = {:.3f} '.format(
        iou[i], dmap[i] * 100) for i in range(len(iou))]))    
    #change the predicr label and evaluate again
    for v_ in wrong_predicted_label_vids:
        str_vid_name, label_g, label_p = v_
        old_label = all_proposals[all_proposals['video-id'] == str_vid_name]['label'].copy()
        new_label = old_label.replace(Counter(old_label.to_list()).most_common(1)[0][0], \
            Counter(label_g).most_common(1)[0][0])
        all_proposals.loc[all_proposals['video-id'] == str_vid_name, 'label'] = new_label
    dmap_detect.prediction = all_proposals
    dmap = dmap_detect.evaluate()
    print('||'.join(['map @ {} = {:.3f} '.format(
        iou[i], dmap[i] * 100) for i in range(len(iou))]))