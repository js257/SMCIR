import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import logging
from easydict import EasyDict as edict
import gc
from config import get_config_regression, get_config_tune
from data_loader import MMDataLoader
from models import AMIO
from trains import ATIO
from utils import assign_gpu, count_parameters, setup_seed

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2" # This is crucial for reproducibility




logger = logging.getLogger('SMCIR_master')


def _set_logger(log_dir, model_name, dataset_name, verbose_level):

    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('SMCIR_master')
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def SMCIR_run(
    model_name: str, dataset_name: str, config_file: str = None,
    config: dict = None, seeds: list = [], is_tune: bool = False,
    tune_times: int = 50, custom_feature: str = None, feature_T: str = None,
    feature_A: str = None, feature_V: str = None, gpu_ids: list = [0],
    num_workers: int = 4, verbose_level: int = 1,
    model_save_dir: str = Path().home() / "SMCIR_master" / "saved_models",
    res_save_dir: str = Path().home() / "SMCIR_master" / "results",
    log_dir: str = Path().home() / "SMCIR_master" / "logs",
):
    """Train and Test MSA models.

    Given a set of hyper-parameters(via config), will train models on training
    and validation set, then test on test set and report the results. If 
    `is_tune` is set, will accept lists as hyper-parameters and conduct a grid
    search to find the optimal values.

    Args:
        model_name: Name of MSA model.
        dataset_name: Name of MSA dataset.
        config_file: Path to config file. If not specified, default config
            files will be used.
        config: Config dict. Used to override arguments in config_file. 
        seeds: List of seeds. Default: [1111, 1112, 1113, 1114, 1115]
        is_tune: Tuning mode switch. Default: False
        tune_times: Sets of hyper parameters to tune. Default: 50
        custom_feature: Path to custom feature file. The custom feature should
            contain features of all three modalities. If only one modality has
            customized features, use `feature_*` below. 
        feature_T: Path to text feature file. Provide an empty string to use
            default BERT features. Default: ""
        feature_A: Path to audio feature file. Provide an empty string to use
            default features provided by dataset creators. Default: ""
        feature_V: Path to video feature file. Provide an empty string to use
            default features provided by dataset creators. Default: ""
        gpu_ids: GPUs to use. Will assign the most memory-free gpu if an empty
            list is provided. Default: [0]. Currently only supports single gpu.
        num_workers: Number of workers used to load data. Default: 4
        verbose_level: Verbose level of stdout. 0 for error, 1 for info, 2 for
            debug. Default: 1
        model_save_dir: Path to save trained model weights. Default: 
            "~/SMCIR_master/saved_models"
        res_save_dir: Path to save csv results. Default: "~/SMCIR_master/results"
        log_dir: Path to save log files. Default: "~/SMCIR_master/logs"
    """
    # Initialization
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    
    if config_file is not None:
        config_file = Path(config_file)
    else: # use default config files
        if is_tune:
            config_file = Path(__file__).parent / "config" / "config_tune.json"
        else:
            config_file = Path(__file__).parent / "config" / "config_regression.json"
    if not config_file.is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)
    if model_save_dir is None: # use default model save dir
        model_save_dir = Path.cwd() / "SMCIR_master" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir is None: # use default result save dir
        res_save_dir = Path.cwd() / "SMCIR_master" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir is None: # use default log save dir
        log_dir = Path.cwd() / "SMCIR_master" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)

    logger.info("======================================== Program Start ========================================")
    
    if is_tune: # run tune
        logger.info(f"Tuning with seed {seeds[0]}")
        initial_args = get_config_tune(model_name, dataset_name, config_file)
        initial_args['model_save_path'] = Path(model_save_dir) / f"{initial_args['model_name']}-{initial_args['dataset_name']}.pth"
        initial_args['device'] = assign_gpu(gpu_ids)
        initial_args['train_mode'] = 'regression' # backward compatibility. TODO: remove all train_mode in code
        initial_args['custom_feature'] = custom_feature
        initial_args['feature_T'] = feature_T
        initial_args['feature_A'] = feature_A
        initial_args['feature_V'] = feature_V

        # torch.cuda.set_device() encouraged by pytorch developer, although dicouraged in the doc.
        # https://github.com/pytorch/pytorch/issues/70404#issuecomment-1001113109
        # It solves the bug of RNN always running on gpu 0.
        torch.cuda.set_device(initial_args['device'])

        res_save_dir = Path(res_save_dir) / "tune"
        res_save_dir.mkdir(parents=True, exist_ok=True)
        has_debuged = [] # save used params
        csv_file = res_save_dir / f"{dataset_name}-{model_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
            for i in range(len(df)):
                has_debuged.append([df.loc[i,k] for k in initial_args['d_paras']])

        for i in range(tune_times):
            args = edict(**initial_args)
            random.seed(time.time())
            new_args = get_config_tune(model_name, dataset_name, config_file)
            args.update(new_args)
            if config:
                if config.get('model_name'):
                    assert(config['model_name'] == args['model_name'])
                args.update(config)
            args['cur_seed'] = i + 1
            logger.info(f"{'-'*30} Tuning [{i + 1}/{tune_times}] {'-'*30}")
            logger.info(f"Args: {args}")
            # check if this param has been run
            cur_param = [args[k] for k in args['d_paras']]
            if cur_param in has_debuged:
                logger.info(f"This set of parameters has been run. Skip.")
                time.sleep(1)
                continue
            # actual running
            setup_seed(seeds[0])
            result = _run(args, num_workers, is_tune)
            has_debuged.append(cur_param)
            # save result to csv file
            if Path(csv_file).is_file():
                df2 = pd.read_csv(csv_file)
            else:
                df2 = pd.DataFrame(columns = [k for k in args.d_paras] + [k for k in result.keys()])
            res = [args[c] for c in args.d_paras]
            for col in result.keys():
                value = result[col]
                res.append(value)
            df2.loc[len(df2)] = res
            df2.to_csv(csv_file, index=None)
            logger.info(f"Results saved to {csv_file}.")
    else: # run normal
        args = get_config_regression(model_name, dataset_name, config_file)
        args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
        args['device'] = assign_gpu(gpu_ids)
        args['train_mode'] = 'regression' # regression，backward compatibility. TODO: remove all train_mode in code
        args['num_labels'] = 1  #
        args['custom_feature'] = custom_feature
        args['feature_T'] = feature_T
        args['feature_A'] = feature_A
        args['feature_V'] = feature_V
        if config: # override some arguments
            if config.get('model_name'):
                assert(config['model_name'] == args['model_name'])
            args.update(config)

        # torch.cuda.set_device() encouraged by pytorch developer, although dicouraged in the doc.
        # https://github.com/pytorch/pytorch/issues/70404#issuecomment-1001113109
        # It solves the bug of RNN always running on gpu 0.
        torch.cuda.set_device(args['device'])

        logger.info("Running with args:")
        logger.info(args)
        logger.info(f"Seeds: {seeds}")
        res_save_dir = Path(res_save_dir) / "normal"
        print(res_save_dir)
        res_save_dir.mkdir(parents=True, exist_ok=True)
        model_results = []
        for i, seed in enumerate(seeds):
            setup_seed(seed)
            args['cur_seed'] = i + 1
            logger.info(f"{'-'*30} Running with seed {seed} [{i + 1}/{len(seeds)}] {'-'*30}")
            # actual running
            result = _run(args, num_workers, is_tune)
            logger.info(f"Result for seed {seed}: {result}")
            model_results.append(result)
        criterions = list(model_results[0].keys())
        # save result to csv
        csv_file = res_save_dir / f"{dataset_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Model"] + criterions)
        # save results
        res = [model_name]
        for c in criterions:
            values = [r[c] for r in model_results]
            mean = round(np.mean(values)*100, 2)
            std = round(np.std(values)*100, 2)
            res.append((mean, std))
        df.loc[len(df)] = res
        df.to_csv(csv_file, index=None)
        logger.info(f"Results saved to {csv_file}.")


def _run(args, num_workers=4, is_tune=False, from_sena=False):
    # load data and models
    dataloader = MMDataLoader(args, num_workers)
    model = AMIO(args).to(args['device'])

    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    # TODO: use multiple gpus
    # if using_cuda and len(args.gpu_ids) > 1:
    #     model = torch.nn.DataParallel(model,
    #                                   device_ids=args.gpu_ids,
    #                                   output_device=args.gpu_ids[0])
    trainer = ATIO().getTrain(args)
    epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(args['device'])
    if from_sena:
        final_results = {}
        final_results['train'] = trainer.do_test(model, dataloader['train'], mode="TRAIN", return_sample_results=True)
        final_results['valid'] = trainer.do_test(model, dataloader['valid'], mode="VALID", return_sample_results=True)
        final_results['test'] = trainer.do_test(model, dataloader['test'], mode="TEST", return_sample_results=True)
    elif is_tune:
        # use valid set to tune hyper parameters
        # results = trainer.do_test(model, dataloader['valid'], mode="VALID")
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        # delete saved model
        Path(args['model_save_path']).unlink(missing_ok=True)
    else:
        results = trainer.do_test(model, dataloader['test'], mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)

    return {"epoch_results": epoch_results, 'final_results': final_results} if from_sena else results
    # return results
