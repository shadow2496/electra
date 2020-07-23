import argparse
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Location of data files (model weights, etc).")
    parser.add_argument("--model-name", required=True,
                        help="The name of the model being fine-tuned.")

    args = parser.parse_args()
    tasks = ["cola", "sst", "mrpc", "sts", "qqp", "mnli", "qnli", "rte"]

    for task in tasks:
        hparams = str('{"task_names": ["') + str(task) + str('"]}')
        cmd = ['python3', 'run_finetuning.py', '--data-dir', args.data_dir, '--model-name', args.model_name, '--hparams', str(hparams)]
        subprocess.call(cmd)
