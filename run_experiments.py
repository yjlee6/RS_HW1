"""RS HW1: run all experiments on ml-100k with RecBole and save results to results.json.

Usage:
    python run_experiments.py            # run everything
    python run_experiments.py knn bpr    # run selected parts (knn, bpr, item2item)
"""
import json
import os
import sys

from recbole.quick_start import run_recbole

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(HERE, "results.json")

BASE_CONFIG = {
    "data_path": os.path.join(HERE, "dataset"),
    "seed": 2020,
    "reproducibility": True,
    "metrics": ["Recall", "MRR", "NDCG", "Hit", "Precision"],
    "topk": [10],
    "valid_metric": "NDCG@10",
    "show_progress": False,
    "save_dataset": True,
    "save_dataloaders": False,
    "checkpoint_dir": os.path.join(HERE, "saved"),
}

K_LIST = [1, 5, 10, 100, 500]
EMB_LIST = [32, 64, 128, 256]
REG_LIST = [0.01, 0.1, 0.5]


def load_results():
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH) as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)


def run(name, model, extra, results, model_cls=None):
    if name in results:
        print(f"[skip] {name}")
        return
    print(f"[run] {name}")
    config = {**BASE_CONFIG, **extra}
    if model_cls is None:
        out = run_recbole(model=model, dataset="ml-100k", config_dict=config)
    else:
        out = run_custom(model_cls, config)
    results[name] = {
        "valid": {k: float(v) for k, v in out["best_valid_result"].items()},
        "test": {k: float(v) for k, v in out["test_result"].items()},
    }
    save_results(results)


def run_custom(model_cls, config_dict):
    """Run a model class that is not registered in RecBole (BPR with L2 reg)."""
    from logging import getLogger

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_trainer, init_logger, init_seed

    config = Config(model=model_cls, dataset="ml-100k", config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = model_cls(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], "BPR")(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=False)
    logger.info(f"best valid : {best_valid_result}")
    logger.info(f"test result: {test_result}")
    return {"best_valid_result": best_valid_result, "test_result": test_result}


def part_knn(results):
    for method in ["item", "user"]:
        tag = "ItemKNN" if method == "item" else "UserKNN"
        for k in K_LIST:
            run(f"{tag}_k{k}", "ItemKNN", {"k": k, "knn_method": method}, results)


def part_bpr(results):
    from bpr_reg import BPRReg

    for emb in EMB_LIST:
        run(f"BPR_emb{emb}_reg0.1", None, {"embedding_size": emb, "reg_lambda": 0.1},
            results, model_cls=BPRReg)
    for reg in REG_LIST:
        run(f"BPR_emb64_reg{reg}", None, {"embedding_size": 64, "reg_lambda": reg},
            results, model_cls=BPRReg)


def part_item2item(results):
    run("SLIMElastic", "SLIMElastic", {}, results)
    run("EASE", "EASE", {}, results)
    # ItemKNN with RecBole's default k for a fair reference point
    run("ItemKNN_default", "ItemKNN", {}, results)


PARTS = {"knn": part_knn, "bpr": part_bpr, "item2item": part_item2item}

if __name__ == "__main__":
    sys.path.insert(0, HERE)
    selected = sys.argv[1:] or list(PARTS)
    results = load_results()
    for p in selected:
        PARTS[p](results)
    print(json.dumps(results, indent=2))
