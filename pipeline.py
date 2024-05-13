import argparse
import datetime
import logging.config
from pathlib import Path

import yaml

import src.acquire_data as ad
import src.analysis as eda
import src.aws_utils as aws
import src.create_dataset as cd
# import cloud_classifier.evaluate_performance as ep
import src.generate_features as gf
import src.score_model as sm
import src.train_model as tm

logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger("clouds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acquire, clean, and create features from clouds data"
    )
    parser.add_argument(
        "--config", default="config/default-config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration file for parameters and run config
    with open(args.config, "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.error.YAMLError as e:
            logger.error(
                "Error while loading configuration from %s", args.config)
        else:
            logger.info("Configuration file loaded from %s", args.config)

    run_config = config.get("run_config", {})

    # Set up output directory for saving artifacts
    now = int(datetime.datetime.now().timestamp())
    artifacts = Path(run_config.get("output", "runs")) / str(now)
    artifacts.mkdir(parents=True)

    # Save config file to artifacts directory for traceability
    with (artifacts / "config.yaml").open("w") as f:
        yaml.dump(config, f)

    # Acquire data from online repository and save to disk
    ad.acquire_data(run_config["data_source"], artifacts / "clouds.data")

    # Create structured dataset from raw data; save to disk
    data = cd.create_dataset(artifacts / "clouds.data", config)
    cd.save_dataset(data, artifacts / "clouds.csv")

    # Enrich dataset with features for model training; save to disk
    target, features = gf.generate_additional_features(data)

    # Generate statistics and visualizations for summarizing the data; save to disk
    figures = artifacts / "figures"
    figures.mkdir()
    eda.save_figures(target, features, figures)

    # Split data into train/test set and train model based on config; save each to disk
    size = config["test_size"]
    n_estimators = config["model_training"]["hyperparameters"]["n_estimators"]
    max_depth = config["model_training"]["hyperparameters"]["max_depth"]
    tmo, train, test = tm.train_model(
        target, features, test_size=size, n_estimators=n_estimators, max_depth=max_depth)
    tm.save_data(train, test, artifacts)
    tm.save_model(tmo, artifacts / "trained_model_object.pkl")

    # Score model on test set; save scores to disk
    scores = sm.score_model(test, tmo)
    sm.save_scores(scores, artifacts / "scores.csv")

    # Upload all artifacts to S3
    aws_config = config.get("aws")
    if aws_config.get("upload", False):
        aws.upload_artifacts(artifacts, aws_config["bucket_name"], aws_config["prefix"])
