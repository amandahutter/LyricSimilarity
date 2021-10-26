import argparse
import yaml

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
    return config