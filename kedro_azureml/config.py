from collections import defaultdict
from typing import Dict, Optional, Type

import yaml
from pydantic import BaseModel, validator


class DefaultConfigDict(defaultdict):
    def __getitem__(self, key):
        defaults: BaseModel = super().__getitem__("__default__")
        this: BaseModel = super().__getitem__(key)
        return defaults.copy(update=this.dict(exclude_none=True)) if defaults else this


class DockerConfig(BaseModel):
    image: str


class AzureTempStorageConfig(BaseModel):
    account_name: str
    container: str


class ResourceConfig(BaseModel):
    cluster_name: str


class AzureMLConfig(BaseModel):
    @staticmethod
    def _create_default_dict_with(
        value: dict, default, dict_cls: Type = DefaultConfigDict
    ):
        default_value = (value := value or {}).get("__default__", default)
        return dict_cls(lambda: default_value, value)

    @validator("resources", always=True)
    def _validate_resources(cls, value):
        return AzureMLConfig._create_default_dict_with(
            value, ResourceConfig(cluster_name="{cluster_name}")
        )

    experiment_name: str
    workspace_name: str
    resource_group: str
    cluster_name: str
    temporary_storage: AzureTempStorageConfig
    resources: Optional[Dict[str, ResourceConfig]]


class KedroAzureMLConfig(BaseModel):
    azure: AzureMLConfig
    docker: DockerConfig


class KedroAzureRunnerConfig(BaseModel):
    # Class for use only in the runner
    temporary_storage: AzureTempStorageConfig
    run_id: str
    storage_account_key: str


CONFIG_TEMPLATE_YAML = """
azure:
  # Name of the Azure ML Compute Cluster
  cluster_name: "{cluster_name}"
  # Azure ML Experiment Name
  experiment_name: "{experiment_name}"
  # Azure resource group to use
  resource_group: "{resource_group}"
  # Azure ML Workspace name
  workspace_name: "{workspace_name}"

  # Temporary storage settings - this is used to pass some data between steps
  # if the data is not specified in the catalog directly
  temporary_storage:
    # Azure Storage account name, where the temp data should be stored
    # It's recommended to set Lifecycle management rule for storage container, to avoid costs of long-term storage
    # of the temporary data. Temporary data will be stored under abfs://<containter>/kedro-azureml-temp path
    # See https://docs.microsoft.com/en-us/azure/storage/blobs/lifecycle-management-policy-configure?tabs=azure-portal
    account_name: "{storage_account_name}"
    # Name of the storage container
    container: "{storage_container}"
  resources:
    __default__:
      cluster_name: "{cluster_name}"
    chunky:
      cluster_name: "chunky-cpu-cluster"
docker:
  # Docker image to use during pipeline execution
  image: "{docker_image}"
""".strip()

# This auto-validates the template above during import
_CONFIG_TEMPLATE = KedroAzureMLConfig.parse_obj(yaml.safe_load(CONFIG_TEMPLATE_YAML))
