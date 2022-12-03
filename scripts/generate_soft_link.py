

import os, sys

os.makedirs('download_log', exist_ok=True)

current_path = sys.path[0]
working_dirs = [p for p in os.listdir('./') if os.path.isdir(p)]
working_dirs.remove('download_log')

for working_dir in working_dirs:
    print('\n\n\nworking_dir: ', current_path, working_dir)
    experiments = [p for p in os.listdir(working_dir) if os.path.isdir(os.path.join(working_dir, p))]
    # print('experiments: ', experiments)

    for experiment in experiments:
        log_path = os.path.join(working_dir, experiment, 'log')
        if os.path.exists(log_path) and os.path.isdir(log_path):
            # print(experiment, os.path.isdir(log_path))
        
            src = os.path.join(current_path, log_path)
            dst = os.path.join(current_path, 'download_log', working_dir, experiment)

            print('\n    src: ', src)
            print('    dst: ', dst)

            os.makedirs(dst, exist_ok=True)
            try:
                os.symlink(src, os.path.join(dst, 'log'))
            except FileExistsError:
                pass

