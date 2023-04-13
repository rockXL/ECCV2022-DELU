from joblib import Parallel, delayed
import subprocess
import os
batch_sizes = list(range(10, 200, 20))
num_similar_ratios = [0.2, 0.3, 0.4, 0.5]
train_cmds = []
test_cmds = []
with open('train_cmd.txt', 'w') as fout:
    with open('test_cmd.txt', 'w') as f1out:
        i = 0
        for bs in batch_sizes:
            for nsr in num_similar_ratios:
                i += 1
                num_similar = int(bs * nsr)
                gpu_id = i % 8
                # generate train cmd
                cmd = 'CUDA_VISIBLE_DEVICES={} sh scripts/train_robot.sh {} {}'\
                    .format(gpu_id, num_similar, bs)
                fout.write('{}\n'.format(cmd))
                train_cmds.append(cmd)
                # generate test cmd
                cmd = 'CUDA_VISIBLE_DEVICES={} sh scripts/test_model.sh {}'\
                    .format(gpu_id, "default_"+str(num_similar)+"_"+str(bs))
                test_cmds.append(cmd)
                f1out.write('{}\n'.format(cmd))


def run_train_cmd(cmd):
    # return subprocess.check_call([cmd], shell=True)
    return os.system(cmd)


def run_test_cmd(cmd):
    # return subprocess.check_call([cmd], shell=True, stdout=subprocess.PIPE)
    run_status_, output_ = subprocess.getstatusoutput([cmd])
    if run_status_ == 0:
        # success
        parse_ = output_.split('\n')
        models = [i for i in parse_ if 'testing model' in i]
        model1, model2 = models
        maps = [i for i in parse_ if "map @ 0.1 = " in i]
        maps_avg = [i for i in parse_ if "mAP Avg 0.1" in i]
        v1, v2 = maps
        v1_avg, v2_avg = maps_avg
        v1_half = v1[v1.index('map @ 0.5 = ')+12:].split(' ')[0]
        v2_half = v2[v2.index('map @ 0.5 = ')+12:].split(' ')[0]
        return [model1, float(v1_half), v1, v1_avg] if float(v1_half) > float(v2_half) else [model2, float(v2_half), v2, v2_avg]
    else:
        return -1
    # return os.system(cmd)


if __name__ == '__main__':
    cores = 8
    # train
    results = Parallel(n_jobs=cores)(delayed(run_train_cmd)(cmd) for cmd in train_cmds)
    print(results)

    # test
    results = Parallel(n_jobs=cores)(delayed(run_test_cmd)(cmd) for cmd in test_cmds)
    total_task = len(results)
    results = [i for i in results if i != -1]
    success_task = len(results)
    s_results = sorted(results, key=lambda x: x[1], reverse=True)
    for i in s_results[:3]:
        print(i)
    with open('default_out.txt', 'w') as fout:
        for i in s_results:
            fout.write('{}\n'.format(i))
    print('total_task:{}'.format(total_task))
    print('success_task:{}'.format(success_task))            
