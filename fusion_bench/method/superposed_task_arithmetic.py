import logging
from copy import deepcopy
from typing import List, Mapping, TypeVar, Union

import torch
from torch import Tensor, nn
import numpy as np

from fusion_bench.method.base_algorithm import ModelFusionAlgorithm
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import ModelPool, to_modelpool
from fusion_bench.utils.type import StateDictType
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)

Module = TypeVar("Module")

log = logging.getLogger(__name__)

class SuperposedTaskArithmeticAlgorithm(
    ModelFusionAlgorithm,
    SimpleProfilerMixin,
):    
    @torch.no_grad()
    def run(self, modelpool: ModelPool):
        modelpool = to_modelpool(modelpool)
        log.info("Compressing models using superposed task arithmetic.")
        task_vector = None
        with self.profile("load model"):
            pretrained_model = modelpool.load_model("_pretrained_")

        # Calculate the task vector superposition
        retrieval_context = {}
        task_vectors = {}
        models = {}
        for model_name in modelpool.model_names:
            with self.profile("load model"):
                model = modelpool.load_model(model_name)
            task_vector = state_dict_sub(
                model.state_dict(keep_vars=True),
                pretrained_model.state_dict(keep_vars=True),
            )

            state_dict = state_dict_add(pretrained_model.state_dict(keep_vars=True), task_vector)
            task_vectors[model_name] = task_vector
        
        with self.profile("superpose weights"):
            # retrieved_task_vectors = self._compress_and_retrieve(task_vectors)
            retrieved_task_vectors = task_vectors
        with self.profile("retrieve models"):
            for model_name in modelpool.model_names:
                retrieved_task_vector = state_dict_mul(retrieved_task_vectors[model_name], self.config.scaling_factor)
                state_dict = state_dict_add(pretrained_model.state_dict(keep_vars=True), retrieved_task_vector)
                retrieved_model = deepcopy(pretrained_model)
                retrieved_model.load_state_dict(state_dict)
                models[model_name] = retrieved_model
        
        self.print_profile_summary()
        return models
  

    def _compress_and_retrieve(self, state_dicts: dict):
        """Assume the state_dicts have the same layers."""
        layers = state_dicts[list(state_dicts.keys())[0]].keys()
        models = list(state_dicts.keys())
        compressed_layers = {}
        retrieval_context = {model: {} for model in models}
        retrieval_models = deepcopy(state_dicts)

        # compress
        if self.config.mode == "random_binary_diagonal_matrix":
            for layer in layers:
                shape = state_dicts[models[0]][layer].shape
                compressed_layer = None
                for model in models:
                    if 'mlp' in layer and 'bias' not in layer:
                        context = np.random.binomial(p=.5, n=1, size=(1, shape[-1])).astype(np.float32) * 2 - 1
                        context = torch.from_numpy(context)
                        retrieval_context[model][layer] = context
                        if compressed_layer is None:
                            compressed_layer = state_dicts[model][layer] * context
                        else:
                            compressed_layer += state_dicts[model][layer] * context
                    else:
                        if compressed_layer is None:
                            compressed_layer = state_dicts[model][layer]
                        else:
                            compressed_layer += state_dicts[model][layer]
                compressed_layers[layer] = compressed_layer
        # retrieve
        for model in models:
            for layer in layers:
                if layer in retrieval_context[model]:
                    retrieval_models[model][layer] = compressed_layers[layer] * retrieval_context[model][layer]
                else:
                    retrieval_models[model][layer] = compressed_layers[layer]
        return retrieval_models