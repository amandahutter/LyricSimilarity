import argparse
import yaml
from typing import List
from matplotlib import pyplot
import csv

def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Train <something... fill this out later>")
    parser.add_argument('--config', help="name of the config file in the config folder, without the extention. e.g. '--config default'")
    args = parser.parse_args()

    with open(f'./config/{args.config}.yaml') as config_file:
        try:
            config = yaml.safe_load(config_file)
        except yaml.YAMLError as yamlex:
            print("Error parsing config file:")
            print(yamlex)
    config['config_name'] = args.config
    return config

def plot_loss(loss_history: List[float], filename: str):
    pyplot.plot(loss_history)
    pyplot.xlabel('1000 batches')
    pyplot.ylabel('Loss')
    pyplot.savefig(filename)

def write_results_to_csv(TP, TN, FP, FN, filename):
    with open(filename, 'w+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['TP', 'TN', 'FP', 'FN'])
        writer.writerow([TP, TN, FP, FN])