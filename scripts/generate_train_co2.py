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
                cmd = 'CUDA_VISIBLE_DEVICES={} sh scripts/train_robot_co2.sh {} {}'\
                    .format(gpu_id, num_similar, bs)
                fout.write('{}\n'.format(cmd))
                train_cmds.append(cmd)
                # generate test cmd
                cmd = 'CUDA_VISIBLE_DEVICES={} sh scripts/test_model.sh {}'\
                    .format(gpu_id, "co2_"+str(num_similar)+"_"+str(bs))
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
        model1 = parse_[0]
        map1 = parse_[12].split('||')[4].split('=')[-1].strip()
        model2 = parse_[15]
        map2 = parse_[-3].split('||')[4].split('=')[-1].strip()
        return [model1, float(map1), parse_[12:15]] if float(map1) > float(map2) else [model2, float(map2), parse_[-3:]]
    else:
        return -1
    # return os.system(cmd)


if __name__ == '__main__':
    cores = 16
    # train
    results = Parallel(n_jobs=cores)(delayed(run_train_cmd)(cmd) for cmd in train_cmds)
    print(results)

    # test
    results = Parallel(n_jobs=cores)(delayed(run_test_cmd)(cmd) for cmd in test_cmds)
    results = [i for i in results if i != -1]
    s_results = sorted(results, key=lambda x: x[1], reverse=True)
    for i in s_results[:3]:
        print(i)
    with open('co2_out.txt', 'w') as fout:
        for i in s_results:
            fout.write('{}\n'.format(i))
