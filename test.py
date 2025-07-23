from SMCIR import SMCIR_run

if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')  # 可选，Windows 默认是 'spawn'
    # run LMF on MOSI with default hyper parameters
    SMCIR_run('smcir', 'mosei', seeds=[1111], gpu_ids=[0])