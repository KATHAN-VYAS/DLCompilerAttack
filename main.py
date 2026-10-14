import os

import torch
import logging
import argparse

from utils import load_DLCL, load_attack_model_cls
from utils import load_model, load_dataloader, set_random_seed

from src.attack.dlcl_attack import DLCompilerAttack
from src import TargetDevice
from utils import SPLIT_SYM, WORK_DIR, LOG_DIR, GENERAL_DIR
from utils import init_bd_trigger


def get_loger(task_name):
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(LOG_DIR, f"{task_name}.log"),
        filemode='w',
        format='%(asctime)s - %(lineno)d - %(module)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    return logger


def main(
    task_id,
    cl_id,
    hardware_id,
    attack_variant="paper",
    batch_size=100,
    max_train_samples=None,
    max_test_samples=None,
    seed=42,
):
    # Configure for Attack
    set_random_seed(seed, deterministic=True)
    if hardware_id == -1:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        raise RuntimeError("GPU mode was requested, but CUDA is unavailable. Use --hardware_id -1.")
    trigger_size, trigger_pos = 8, "left_up"
    bd_trigger = init_bd_trigger(trigger_size, trigger_pos, device)

    cl_func = load_DLCL(cl_id)
    hardware_target = TargetDevice(hardware_id)

    model = load_model(task_id, load_pretrained=True)
    # The original '::::' delimiter is invalid in Windows directory names.
    # Keep each baseline/CDG/seed result isolated for fair comparisons.
    safe_model_name = model.model_data_name.replace(SPLIT_SYM, "__")
    experiment_id = f"{safe_model_name}__cl_{cl_id}{hardware_target}__{attack_variant}__seed_{seed}"
    task_name = experiment_id
    work_dir = os.path.join(WORK_DIR, task_name)
    if not os.path.isdir(work_dir):
        os.mkdir(work_dir)

    FeatureModel, TunedModel, embed_shape = load_attack_model_cls(task_id)
    train_loader, valid_loader, test_loader = (
        load_dataloader(
            task_id, False, batch_size, batch_size,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
        ))

    D = FeatureModel(model)
    tuned_model = TunedModel(model, embed_shape)

    # logger = get_loger(task_name)

    print(task_name)

    attack_config = {
        "trigger_opt_epoch": 10,
        "trigger_opt_lr": 1e-2,
        "finetune_cl_epoch": 10,
        "finetune_cl_lr": 1e-2,
        "finetune_epoch": 50,
        "finetune_lr": 1e-4,
        "save_freq": 10,
        "task_name": task_name,
        "model_data_name": experiment_id,
        "work_dir": work_dir,
        "batch_size": batch_size,
        "general_dir": GENERAL_DIR,
        "bd_rate": 1.0,
        # Eq. 7 in the paper includes compiled-clean supervision.
        "compiled_clean_weight": 1.0,
        "attack_variant": attack_variant,
        # CDG-DcL-BD only uses these when --attack_variant cdg is selected.
        "cdg_tau": 0.95,
        "cdg_support_channels": 1,
        "cdg_calibration_batches": 1,
        "cdg_spsa_radius": 0.02,
        "cdg_spsa_trials": 1,
        "cdg_stealth_weight": 0.25,
    }
    # train_loader = test_loader   #TODO for debug
    attacker = DLCompilerAttack(
        D, tuned_model,
        train_loader, test_loader, bd_trigger,
        cl_func, hardware_target, device, attack_config
    )
    attacker.run_attack(attack_stage_list={0, 1, 2})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=1)
    parser.add_argument('--cl_id', type=int, default=0)
    parser.add_argument('--hardware_id', type=int, default=-1)
    parser.add_argument('--attack_variant', choices=['paper', 'cdg'], default='paper')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_train_samples', type=int)
    parser.add_argument('--max_test_samples', type=int)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    assert args.task_id in [0, 1, 2, 3, 4, 5]
    assert args.cl_id in [0, 1, 2]
    assert args.hardware_id in [-1, 0]
    main(
        task_id=args.task_id,
        cl_id=args.cl_id,
        hardware_id=args.hardware_id,
        attack_variant=args.attack_variant,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )
