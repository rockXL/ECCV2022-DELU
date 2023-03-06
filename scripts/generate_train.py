from joblib import Parallel, delayed
import subprocess
import os
batch_sizes = list(range(10, 200, 20))
num_similar_ratios = [0.2, 0.3, 0.4, 0.5]
cmds = []
with open('train_cmd.txt', 'w') as fout:
    i = 0
    for bs in batch_sizes:
        for nsr in num_similar_ratios:
            i += 1
            num_similar = int(bs * nsr)
            gpu_id = i % 8
            cmd = 'CUDA_VISIBLE_DEVICES={} sh scripts/train_robot.sh {} {}'\
                .format(gpu_id, num_similar, bs)
            fout.write('{}\n'.format(cmd))
            cmds.append(cmd)


def run_cmd(cmd):
    # return subprocess.check_call([cmd])
    return os.system(cmd)


if __name__ == '__main__':
    cores = 64
    results = Parallel(n_jobs=cores)(delayed(run_cmd)(cmd) for cmd in cmds)
    print(results)
