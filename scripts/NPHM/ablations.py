from collections import defaultdict
from typing import Dict

from elias.util import load_json
from matplotlib import pyplot as plt

from visualizations.env_nphm import NPHM_DATA_PATH

NPHM_ABLATIONS_FOLDER = f"{NPHM_DATA_PATH}/ablations"

SELECTED_METHODS = ["NPM", "NPHM"]
SELECTED_METRICS = {
    "f_score_05": "F-Score @ 1.5mm" #, "chamfer_l1", "normals consistency"
}


def parse_ablations_json(path: str) -> Dict[str, Dict[int, float]]:
    ablations_json = load_json(path)

    ablation_results = defaultdict(lambda: dict())

    for ablation_run_name, ablation_run_results in ablations_json.items():
        # Examples for ablation_run_name:
        #  - NPM_dir100
        #  - NPHM_dir1000_r1

        method_name, ablation_config = ablation_run_name.split('_dir')
        if '_' in ablation_config:
            # Handling of _r1, _r2 regularization types
            ablation_config, regularization_type = ablation_config.split('_')
            method_name = f"{method_name}_{regularization_type}"

        if not ablation_config:
            print(f"Skipping ablation results for {ablation_run_name} as the number of points could not be parsed")
            continue
        ablation_config = float(ablation_config)

        ablation_results[method_name][ablation_config] = ablation_run_results

    return ablation_results


def plot_ablation_results(ablation_results,
                          name: str,
                          skip_data_points: int = 0,
                          scale_x: float = 1):
    metrics = {metric
               for single_method_ablation_results in ablation_results.values()
               for ablation_result in single_method_ablation_results.values()
               for metric in ablation_result.keys()}

    for metric in metrics:
        if metric not in SELECTED_METRICS.keys():
            continue
        metric_label = SELECTED_METRICS[metric]
        plt.figure(figsize=(4, 2), dpi=500)
        plt.grid()
        #plt.title(f"{name} - {metric}")
        plt.xlabel(f"{name}")
        plt.ylabel(metric_label)

        for method_name, method_ablation_results in ablation_results.items():
            if method_name not in SELECTED_METHODS:
                continue

            method_ablation_results_items = sorted(method_ablation_results.items(), key=lambda x: x[0])
            method_ablation_results_keys, method_ablation_results_values = zip(*method_ablation_results_items)

            xs = method_ablation_results_keys[skip_data_points:]
            xs = [x * scale_x for x in xs]
            values = [result[metric] for result in method_ablation_results_values][skip_data_points:]

            plt.plot(xs, values, 's--', label=method_name)

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{NPHM_ABLATIONS_FOLDER}/plot_ablation_{name.lower().replace(' ', '_')}_{metric}.pdf")


if __name__ == '__main__':
    ablation_results_fitting_n_points = parse_ablations_json(f"{NPHM_ABLATIONS_FOLDER}/ablations.json")
    ablation_results_fitting_noise = parse_ablations_json(f"{NPHM_ABLATIONS_FOLDER}/ablations_noise.json")

    plot_ablation_results(ablation_results_fitting_n_points, "Number of points", skip_data_points=1)
    plot_ablation_results(ablation_results_fitting_noise,
                          "Noise standard deviation [mm]",
                          scale_x=1000 / (5 / 1.5),
                          skip_data_points=1)
