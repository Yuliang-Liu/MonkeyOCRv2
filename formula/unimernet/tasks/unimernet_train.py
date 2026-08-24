import torch

from unimernet.common.registry import registry
from unimernet.tasks.base_task import BaseTask
from unimernet.common.dist_utils import main_process
from unimernet.common.metrics import compute_corpus_bleu
import os.path as osp
import json
import numpy as np
from rapidfuzz.distance import Levenshtein


@registry.register_task("unimernet_train")
class UniMERNet_Train(BaseTask):

    def __init__(self, temperature, do_sample, top_p, evaluate, report_metric=True, agg_metric="edit_distance"):
        super(UniMERNet_Train, self).__init__()
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.evaluate = evaluate
        self.agg_metric = agg_metric

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        generate_cfg = run_cfg.generate_cfg

        temperature = generate_cfg.get('temperature', .2)
        do_sample = generate_cfg.get("do_sample", False)
        top_p = generate_cfg.get("top_p", 0.95)

        evaluate = run_cfg.evaluate
        report_metric = run_cfg.get("report_metric", True)
        agg_metric = run_cfg.get("agg_metric", "edit_distance")

        return cls(
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            evaluate=evaluate,
            report_metric=report_metric,
            agg_metric=agg_metric,
        )

    def valid_step(self, model, samples):
        results = []
        image, text = samples["image"], samples["text_input"]
        preds = model.generate(
            samples,
            temperature=self.temperature,
            do_sample=self.do_sample,
            top_p=self.top_p
        )
        pred_tokens = preds["pred_tokens"]
        pred_strs = preds["pred_str"]
        pred_ids = preds["pred_ids"]  # [b, n-1]

        truth_inputs = model.tokenizer.tokenize(text)
        truth_ids = truth_inputs["input_ids"][:, 1:]
        truth_tokens = model.tokenizer.detokenize(truth_inputs["input_ids"])
        truth_strs = model.tokenizer.token2str(truth_inputs["input_ids"])

        ids = samples["id"]

        for pred_token, pred_str, pred_id, truth_token, truth_str, truth_id, id_ in zip(pred_tokens, pred_strs,
                                                                                        pred_ids, truth_tokens,
                                                                                        truth_strs, truth_ids, ids):
            pred_id = pred_id.tolist()
            truth_id = truth_id.tolist()
            shape_diff = len(pred_id) - len(truth_id)
            if shape_diff < 0:
                pred_id = pred_id + [model.tokenizer.pad_token_id] * (-shape_diff)
            else:
                truth_id = truth_id + [model.tokenizer.pad_token_id] * shape_diff
            pred_id, truth_id = torch.LongTensor(pred_id), torch.LongTensor(truth_id)
            mask = torch.logical_or(pred_id != model.tokenizer.pad_token_id, truth_id != model.tokenizer.pad_token_id)
            tok_acc = (pred_id == truth_id)[mask].float().mean().item()

            this_item = {
                "pred_token": pred_token,
                "pred_str": pred_str,
                "truth_str": truth_str,
                "truth_token": truth_token,
                "token_acc": tok_acc,
                "id": id_
            }
            results.append(this_item)
        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        with open(eval_result_file) as f:
            results = json.load(f)

        edit_dists = []
        all_pred_tokens = []
        all_truth_tokens = []
        all_pred_strs = []
        all_truth_strs = []
        token_accs = []
        for result in results:
            pred_token, pred_str, truth_token, truth_str, tok_acc = result["pred_token"], result["pred_str"], result[
                "truth_token"], result["truth_str"], result["token_acc"]

            if len(truth_str) > 0:
                norm_edit_dist = Levenshtein.normalized_distance(pred_str, truth_str)
                edit_dists.append(norm_edit_dist)

            all_pred_tokens.append(pred_token)
            all_truth_tokens.append([truth_token])
            all_pred_strs.append(pred_str)
            all_truth_strs.append(truth_str)
            token_accs.append(tok_acc)

        bleu_score = compute_corpus_bleu(all_pred_strs, all_truth_strs)
        
        edit_distance = np.mean(edit_dists)
        token_accuracy = np.mean(token_accs)
        eval_ret = {"bleu": bleu_score, "edit_distance": edit_distance, "token_accuracy": token_accuracy}

        log_stats = {split_name: {k: v for k, v in eval_ret.items()}}

        with open(
                osp.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in eval_ret.items()}
        # agg_metrics = sum([v for v in eval_ret.values()])
        if "edit" in self.agg_metric.lower():  # edit_distance
            agg_metrics = (1 - edit_distance) * 100
        elif "bleu" in self.agg_metric.lower():  # bleu_score
            agg_metrics = bleu_score * 100
        elif "token" in self.agg_metric.lower():  # token_accuracy
            agg_metrics = token_accuracy * 100
        else:
            raise ValueError(f"Invalid metrics: '{self.agg_metric}'")

        coco_res["agg_metrics"] = agg_metrics

        return coco_res
