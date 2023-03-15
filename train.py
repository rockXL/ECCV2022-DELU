import torch
import numpy as np
import wandb

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def model_forward(model, features, device, itr, args, requires_grad):
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, :np.max(seq_len), :]
    features = torch.from_numpy(features).float().to(device)
    if not requires_grad:
        features = features.detach()
    return seq_len, model(features, seq_len=seq_len, is_training=True, itr=itr, opt=args)

def train(itr, dataset, args, model, optimizer, device):
    model.train()
    # features, labels, pairs_id = dataset.load_data(n_similar=args.num_similar)
    if args.use_multi_speed_feature:
        features, labels, pairs_id  = dataset.load_aug_data(args, speed=[0.5, 1, 2])
        features_slow, features, features_fast = features 
        seq_len, outputs = model_forward(model, features, device, itr, args, requires_grad=True)
        slow_seq_len, slow_outputs = model_forward(model, features_slow, device, itr, args, requires_grad=True)
        fast_seq_len, fast_outputs = model_forward(model, features_fast, device, itr, args, requires_grad=True)
        # print("seq_len:{}".format(seq_len))
        seq_len = max(max(seq_len), max(slow_seq_len), max(fast_seq_len))
        outputs = [outputs, slow_outputs, fast_outputs]
    else:
        features, labels, pairs_id = dataset.load_aug_data(args, speed=1.0)
        seq_len, outputs = model_forward(model, features, device, itr, args, requires_grad=True)
    labels = torch.from_numpy(labels).float().to(device)
    
    total_loss, loss_dict = model.criterion(outputs, labels, seq_len=seq_len, device=device, opt=args,
                                            itr=itr, pairs_id=pairs_id, inputs=features)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if not args.without_wandb:
        if itr % 20 == 0 and itr != 0:
            wandb.log(loss_dict)

    return total_loss.data.cpu().numpy()
