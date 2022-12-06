import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Input,Flatten
from keras.models import Model
from glob import glob
import os
import argparse
from get_data import get_data
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
import tensorflow
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from mlflow.tracking import MlflowClient
from pprint import pprint
import joblib


def log_production_model(config_path):
    config=get_data(config_path)
    mlflow_config=config["mlflow_config"]
    model_name=mlflow_config["registered_model_name"]
    #experiment_id = mlflow.create_experiment(model_name)
    print(model_name)
    remote_server_uri=mlflow_config["remote_server_uri"]
    print(remote_server_uri)
    mlflow.set_tracking_uri=(remote_server_uri)
    runs= mlflow.search_runs([model_name]) 
    # runs[0]
    # print(runs[["metrics.m", "tags.s.release", "run_id"]])
    # runs=MlflowClient.search_runs(experiment_ids=[0])
    # lowest=runs["metrics.train_accuracy"].sort_values(ascending=True)[0]
    # lowest_run_id=runs[runs["metrics.train_accuracy"]==lowest]["run_id"][0]
    #client=MlflowClient()
    #filter_string = "name='{}'".format(model_name)
    #print(filter_string)
    # results = client.search_model_versions(filter_string)

    # for mv in client.search_model_versions(f"name='{model_name}' "):
    #     mv=dict(mv)
    #     if mv["run_id"]==lowest_run_id:
    #         current_version=mv["version"]
    #         logged_model=mv["source"]
    #         pprint(mv, indent=4)
    #         client.transition_model_version_stage(name=model_name, version=current_version, stage="Production")
    #     else:
    #         current_version=mv["version"]
    #         client.transition_model_version_stage(name=model_name, version=current_version, stage="Staging")
    # loaded_model=mlflow.pyfunc.load_model(logged_model)
    # model_path= config["web_model_directory"]
    # joblib.dump(loaded_model, model_path)



if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config',default='params.yaml')
    passed_args = args_parser.parse_args()
    data=log_production_model(config_path=passed_args.config)