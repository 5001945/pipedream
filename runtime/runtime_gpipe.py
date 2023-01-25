# Edited by Mapae.

import collections
import itertools
import time
import torch
import torch.distributed as dist

import communication
import runtime_utilities

IMAGE_CLASSIFICATION = "image_classification"
TRANSLATION = "translation"
SPEECH_TO_TEXT = "speech_to_text"


## START_MAPAEAN

import math
from typing import Iterable, Union, Literal, Any, Optional
from dataclasses import dataclass

import torch.nn as nn
import torch.utils.data

Shape = tuple[int]
ModelType = tuple[nn.Module, list[str], list[str]]

def override(method):
    return method

def mapaean_original(method):
    return method


def get_cuda_devices() -> list[torch.device]:
    total_device_num = torch.cuda.device_count()
    return [torch.device(i) for i in range(total_device_num)]

def _forward_worker(rank: int, modules: nn.Sequential, microbatch: torch.Tensor):
    pass

@dataclass
class ConfigMaps:
    module_to_stage_map: list[int]
    stage_to_rank_map: dict[int, list[int]]
    stage_to_depth_map: Optional[dict[int, int]]
    
    def __post_init__(self):
        self.stage_to_depth_map = {
            int(k): v for (k, v) in self.stage_to_depth_map.items()
        }


class ModulesWithDependencies:
    """자기 stage의 module들만 모아 놓음.
    """
    def __init__(self, modules_with_dependencies: list[ModelType]):
        self._modules: list[nn.Module] = []
        self._all_input_names: list[list[str]] = []
        self._all_output_names: list[list[str]] = []
        for (module, input_names, output_names) in modules_with_dependencies:
            self._modules.append(module)
            self._all_input_names.append(input_names)
            self._all_output_names.append(output_names)

    def modules(self):
        return self._modules

    def all_input_names(self):
        return self._all_input_names

    def all_output_names(self):
        return self._all_output_names

    def is_input_tensor(self, tensor_name):
        for module_input_names in self._all_input_names:
            if tensor_name in module_input_names:
                return True
        return False

# class MapaeanGPipe(nn.Module):
class MapaeanGPipe():
    """GPipe module by Mapae."""
    @override
    def __init__(self,
        model: list[ModelType],
        # ex) [
        #       (Stage0(), ["input0"], ["out0"]),
        #       (Stage1(), ["out0"], ["out1"]),
        #       (criterion, ["out1"], ["loss"])
        #     ]
        *,
        distributed_backend: Literal['gloo', 'nccl'],
        fp16: bool,  # 이건 안 쓸 것.
        loss_scale: float = 1,
        training_tensor_shapes: dict[str, Shape],
        eval_tensor_shapes: dict[str, Shape],
        training_tensor_dtypes: dict[str, torch.dtype],
        inputs_module_destinations: dict[str, int],  # ex) {"input": 0}
        target_tensor_names: set[str],  # ex) {"targets"}
        configuration_maps: dict[Literal['module_to_stage_map', 'stage_to_rank_map', 'stage_to_depth_map'], Any],
        master_addr: str,  # ex) '127.0.0.1'
        rank: int,
        local_rank: int,
        num_ranks_in_server: int,
        verbose_freq: int = 0,
        model_type: str,  # ex) "IMAGE_CLASSIFICATION"
        enable_recompute: bool = False
    ):
        # 이 stage의 forward와 backward에서 필요한 메타데이터들을 정의한다.
        self.tensors = []
        self.gradients = {}
        self.distributed_backend = distributed_backend
        self.fp16 = False  # 여기서는 사용하지 않음
        self.loss_scale = loss_scale
        self.training_tensor_shapes = training_tensor_shapes
        self.eval_tensor_shapes = eval_tensor_shapes
        self.training_tensor_dtypes = training_tensor_dtypes
        self.model_type = model_type
        self.target_tensor_names = target_tensor_names

        # GPipe용
        self.tensor_stack: list[dict[str, torch.Tensor]] = []

        self.initialize(model, inputs_module_destinations, ConfigMaps(**configuration_maps),
                        master_addr, rank, local_rank, num_ranks_in_server)

        self.verbose_freq = verbose_freq
        self.forward_only = False

        self.forward_stats = runtime_utilities.RuntimeStats(forward=True)
        self.backward_stats = runtime_utilities.RuntimeStats(forward=False)

        # Enable recomputation to prevent the need to save activations
        # computed from the forward pass for the backward pass.
        self.enable_recompute = enable_recompute

        # Disable recomputation for the last stage.
        if rank == num_ranks_in_server - 1:
            self.enable_recompute = False

    def initialize(self,
        model: list[ModelType],
        inputs_module_destinations: dict[str, int],
        cfg: ConfigMaps,
        master_addr: str,
        rank: int,
        local_rank: int,
        num_ranks_in_server: int
    ):
        """서로 다른 rank 사이의 모듈들

        Args:
            model (ModelType): 돌리고자 하는 전체 모델.
            inputs_module_destinations (dict[str, int]): ???
            configuration_maps (ConfigMaps): 각 모듈, stage, rank, depth 간 관계를 정의한다.
                - module_to_stage_map (list[int]): 각 모듈(Stage0, Stage1, criterion 등)이 어떤 stage로 배치될지를 결정.
                - stage_to_rank_map (dict[int, int]): 각 stage가 어떤 machine에 배치될지를 결정.
                - stage_to_depth_map (dict[int, int]): 각 stage가 
            master_addr (str): 컴퓨터 address
            rank (int): 현재 GPU의 rank.
            local_rank (int): 현재 GPU의 local rank.
            num_ranks_in_server (int): 한 server에 몇 개의 rank가 있는지?
        """
        self.send_ranks: dict[str, list[int]] = {}
        self.receive_ranks: dict[str, list[int]] = {}
        self.rank = rank
        self.local_rank = local_rank
        self.stage = None
        self.tensor_tags: dict[str, int] = {}
        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0
        self.criterion_input_name = str(model[-1][1][0])

        # model 간 전달하는 각 tensor에 대해, 고유한 tensor_tag 번호를 붙여준다.
        tensor_tag = 1
        for (_, input_tensors, output_tensors) in model:
            for input_tensor in input_tensors:
                if input_tensor not in self.tensor_tags:
                    self.tensor_tags[input_tensor] = tensor_tag
                    tensor_tag += 1
            for output_tensor in output_tensors:
                if output_tensor not in self.tensor_tags:
                    self.tensor_tags[output_tensor] = tensor_tag
                    tensor_tag += 1
        for target_tensor_name in sorted(self.target_tensor_names):  # {"targets"}
            self.tensor_tags[target_tensor_name] = tensor_tag
            tensor_tag += 1
        self.tensor_tags["ack"] = tensor_tag
        tensor_tag += 1

        if cfg.module_to_stage_map is None:
            # 각 stage의 IP 주소가 특정되지 않으면, 모든 stage를 single machine에 배치한다.
            # 물론 이는 우리가 원하는 건 아니다.
            assert self.rank is None
            self.modules_with_dependencies = ModulesWithDependencies(model)
            self.is_criterion = True
            self.rank_in_stage = 0
            self.num_ranks = 1
            self.num_ranks_in_first_stage = 1
            self.num_ranks_in_previous_stage = 0
            self.num_ranks_in_next_stage = 0
            self.num_stages = 1
            self.num_ranks_in_stage = 1
            self.num_warmup_minibatches = 0
            self.comm_handler = None
        else:
            # GPipe에서 중요한 건 사실 module_to_stage_map 하나뿐이다.
            assert len(cfg.module_to_stage_map) == len(model)
            assert self.rank is not None

            # 각 stage에 어떤 module을 배치할지 정한다.
            stage_to_module_map: dict[int, list[int]] = collections.defaultdict(list)  # ex) {0: [0], 1: [1, 2]}
            for module in range(len(cfg.module_to_stage_map)):
                stage_to_module_map[cfg.module_to_stage_map[module]].append(module)

            rank_to_stage_map: dict[int, int] = {}  # ex) {0: 0, 1: 0, 2: 0, 3: 1}
            for stage in cfg.stage_to_rank_map:
                for rank in cfg.stage_to_rank_map[stage]:
                    rank_to_stage_map[rank] = stage

            # 여기서 중요한 건 stage, ranks_in_prev_stage, ranks_in_next_stage, modules_with_dependencies, is_criterion 이다.
            # rank_in_stage는 쓰지 않는다. GPipe이므로 data parallelism은 중요치 않음.
            assert 0 <= self.rank < len(rank_to_stage_map)
            self.num_ranks = len(rank_to_stage_map)
            self.num_stages = len(stage_to_module_map)
            self.stage = rank_to_stage_map[self.rank]
            self.rank_in_stage = cfg.stage_to_rank_map[self.stage].index(self.rank)
            self.num_ranks_in_stage = len(cfg.stage_to_rank_map[self.stage])
            self.num_ranks_in_first_stage = len(cfg.stage_to_rank_map[0])
            self.num_ranks_in_previous_stage = 0
            self.ranks_in_previous_stage = []
            if self.stage > 0:
                self.num_ranks_in_previous_stage = len(cfg.stage_to_rank_map[self.stage - 1])
                self.ranks_in_previous_stage = cfg.stage_to_rank_map[self.stage - 1]
            self.num_ranks_in_next_stage = 0
            self.ranks_in_next_stage = []
            if self.stage < self.num_stages - 1:
                self.num_ranks_in_next_stage = len(cfg.stage_to_rank_map[self.stage + 1])
                self.ranks_in_next_stage = cfg.stage_to_rank_map[self.stage + 1]
            modules = stage_to_module_map[self.stage]
            self.modules_with_dependencies = ModulesWithDependencies([model[module] for module in modules])  # 자기 stage의 module들을 모아놓았다.
            self.is_criterion = self.stage == (self.num_stages - 1)
            if cfg.stage_to_depth_map is not None:
                self.num_warmup_minibatches = cfg.stage_to_depth_map[str(self.stage)]
            else:
                self.num_warmup_minibatches = self.num_ranks - 1
                for i in range(self.stage):
                    self.num_warmup_minibatches -= len(cfg.stage_to_rank_map[i])
                self.num_warmup_minibatches = self.num_warmup_minibatches // self.num_ranks_in_stage

            # tensor를 어디로 주고받을지를 정하기 위해, 먼저 각 텐서를 만들거나 사용하는 모듈의 ID를 정한다.
            # 그리고 해당 rank를 이용해 실제로 주고받는다.
            self.comm_handler = communication.CommunicationHandler(
                master_addr=master_addr,
                master_port=12345,  # magic number?
                rank=self.rank,
                local_rank=self.local_rank,
                num_ranks_in_server=num_ranks_in_server,
                world_size=self.num_ranks,
                fp16=self.fp16,
                backend=self.distributed_backend
            )

            # receive_ranks와 send_ranks를 결정한다.
            for i in range(len(model)):
                for j in range(i+1, len(model)):
                    for tensor_name in model[i][2]:
                        if tensor_name in model[j][1]:
                            if cfg.module_to_stage_map[i] == cfg.module_to_stage_map[j]:
                                continue
                            # For now, assume that each stage is served by only a single machine.
                            if cfg.module_to_stage_map[j] == self.stage:
                                self.receive_ranks[tensor_name] = cfg.stage_to_rank_map[cfg.module_to_stage_map[i]]
                            if cfg.module_to_stage_map[i] == self.stage:
                                self.send_ranks[tensor_name] = cfg.stage_to_rank_map[cfg.module_to_stage_map[j]]

            for model_inputs in inputs_module_destinations.keys():  # {"input": 0}
                destination_stage = cfg.module_to_stage_map[inputs_module_destinations[model_inputs]]
                # destination_stage는 목적지라는 뜻인 듯.
                if self.stage < destination_stage:
                    self.send_ranks[model_inputs] = self.ranks_in_next_stage

                if 0 < self.stage <= destination_stage:
                    self.receive_ranks[model_inputs] = self.ranks_in_previous_stage

                if destination_stage > 0:
                    if model_inputs not in self.tensor_tags:
                        self.tensor_tags[model_inputs] = tensor_tag
                        tensor_tag += 1

        # 모든 모듈들을 cuda로 보낸다.
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i] = modules[i].cuda()

        # Initialize all groups in the same order on every worker.
        # 아마 GPipe에서는 항상 len(ranks) == 1 이므로 group=None으로 고정될 것.
        if cfg.stage_to_rank_map is not None:
            groups = []
            for stage in range(self.num_stages):
                ranks = cfg.stage_to_rank_map[stage]
                if len(ranks) > 1:
                    groups.append(dist.new_group(ranks=ranks))
                else:
                    groups.append(None)
            group = groups[self.stage]
        else:
            group = None

        # self.modules_with_dependencies contains a list of PyTorch
        # modules, along with a list of user-defined input and output
        # tensor names. We use our module_executor.ModuleExecutor
        # class to wrap these dependencies, and use run_forward and
        # run_backward methods downstream.
        num_parameters = 0
        if group is not None:
            for i in range(len(modules)):
                if ((i < (len(modules)-1) and self.is_criterion) or not self.is_criterion):
                    num_parameters += \
                        sum(x.size()[0] * x.size()[1]
                            if len(x.size()) > 1 else x.size()[0]
                            for x in modules[i].parameters() if x.size())
                    modules[i] = torch.nn.parallel.DistributedDataParallel(
                        modules[i],
                        process_group=group,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank
                    )
        if self.num_ranks_in_stage > 1:
            module_size = 4. * num_parameters
            print("Replicating stage: ranks=%d, module_size=%.3f" % (
                self.num_ranks_in_stage, module_size))

        self.master_parameters = list(self.parameters())
        self.model_parameters = None

        if self.comm_handler is not None:
            self.comm_handler.initialize(
                self.receive_ranks,
                self.send_ranks,
                self.tensor_tags,
                self.target_tensor_names,
                self.training_tensor_dtypes,
                self.rank_in_stage,
                self.num_ranks_in_stage,
                self.ranks_in_previous_stage,
                self.ranks_in_next_stage
            )

    @property
    def target(self):
        return self.tensors[-1]["target"]

    def modules(self):
        return self.modules_with_dependencies.modules()

    def parameters(self):
        parameter_iterators = []
        for module in self.modules_with_dependencies.modules():
            parameter_iterators.append(module.parameters())
        return itertools.chain(*parameter_iterators)

    def state_dict(self):
        state_dict = collections.OrderedDict()
        for i, module in enumerate(self.modules_with_dependencies.modules()):
            state_dict["module%d" % i] = module.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for i, module in enumerate(self.modules_with_dependencies.modules()):
            module.load_state_dict(state_dict["module%d" % i])

    def cuda(self):
        """모든 모듈들을 cuda로 보낸다.
        """
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i] = modules[i].cuda()

    def zero_grad(self):
        """모든 모듈에 대해 zero_grad를 적용한다.
        """
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].zero_grad()

    def train(self, num_iterations):
        self.tensors = []
        self.gradients = {}
        self.tensor_shapes = self.training_tensor_shapes
        self.forward_only = False

        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)
            self.comm_handler.start_helper_threads(
                num_iterations, forward_only=False)

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].train()

    def eval(self, num_iterations):
        self.tensors = []
        self.gradients = {}
        self.tensor_shapes = self.eval_tensor_shapes
        self.tensor_shapes["ack"] = (1,)
        self.forward_only = True

        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)
            self.comm_handler.start_helper_threads(
                num_iterations, forward_only=True)

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].eval()

    def set_loader(self, loader):
        if loader is not None:
            self.loader_iter = iter(loader)
        else:
            self.loader_iter = None

    def receive_tensors_forward(self):
        if self.forward_only and len(self.tensors) > 0:
            self.tensors.pop(0)
        self.tensors.append({})
        if self.loader_iter is not None:
            input = next(self.loader_iter)
            if self.model_type == TRANSLATION:
                (input, target) = input
                src, src_length = input
                tgt, tgt_length = target

                self.tensors[-1]["input0"] = src.cuda(non_blocking=True)
                self.tensors[-1]["input1"] = torch.LongTensor(src_length).cuda(
                    non_blocking=True)
                self.tensors[-1]["input2"] = tgt[:-1].cuda(non_blocking=True)
                self.tensors[-1]["target"] = tgt[1:].cuda().contiguous().view(-1)
                self.tensors[-1]["target_length"] = \
                    torch.tensor([int(sum(torch.LongTensor(tgt_length) - 1))],
                                 dtype=torch.int).cuda()
            elif self.model_type == IMAGE_CLASSIFICATION:
                (input, target) = input
                self.tensors[-1]["input0"] = input.cuda(non_blocking=True)
                self.tensors[-1]["target"] = target.cuda(non_blocking=True)
            elif self.model_type == SPEECH_TO_TEXT:
                input, target, input_percentages, target_sizes = input
                input_sizes = input_percentages.mul_(int(input.size(3))).int()
                self.tensors[-1]["input0"] = input.cuda(non_blocking=True)
                self.tensors[-1]["input1"] = input_sizes.cuda(non_blocking=True)
                self.tensors[-1]["target"] = target.cuda(non_blocking=True)
                self.tensors[-1]["target_length"] = target_sizes.cuda(
                    non_blocking=True)
        else:
            # Receive all required tensors from upstream machines.
            for input_name in self.receive_ranks:
                if input_name == "ack":
                    continue

                self.tensors[-1][input_name] = \
                    self.comm_handler.recv(
                        input_name,
                        forward_minibatch_id=self.forward_minibatch_id,
                        backward_minibatch_id=self.backward_minibatch_id,
                        backward=False)

                self.forward_stats.stats['receive_tensors_size'] += \
                    (self.tensors[-1][input_name].element_size() *
                     self.tensors[-1][input_name].nelement())

            # Used to track where to receive forward from.
            self.comm_handler.increment_messaging_index(
                sending=False)

    def send_tensors_forward(self):
        # Send all required tensors downstream.
        for output_name in self.send_ranks:
            if output_name == "ack":
                continue

            self.comm_handler.send(
                output_name,
                self.tensors[-1][output_name],
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=False)

            self.forward_stats.stats['send_tensors_size'] += \
                (self.tensors[-1][output_name].element_size() *
                 self.tensors[-1][output_name].nelement())

    def receive_tensors_backward(self):
        # Receive all required gradients from downstream
        # machines.
        for output_name in self.send_ranks:
             if output_name in self.target_tensor_names:
                continue

             self.gradients[output_name] = \
                self.comm_handler.recv(
                    output_name,
                    forward_minibatch_id=self.forward_minibatch_id,
                    backward_minibatch_id=self.backward_minibatch_id,
                    backward=True)

             self.backward_stats.stats['receive_tensors_size'] += \
                 (self.gradients[output_name].element_size() *
                  self.gradients[output_name].nelement())

    def send_tensors_backward(self):
        # Send all required gradients upstream.
        for input_name in self.receive_ranks:
            if input_name in self.target_tensor_names:
                continue

            self.comm_handler.send(
                input_name,
                self.gradients[input_name],
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)

            self.backward_stats.stats['send_tensors_size'] += \
                (self.gradients[input_name].element_size() *
                 self.gradients[input_name].nelement())

        if self.num_ranks_in_previous_stage > 0:
            # Used to track where to send tensors in the
            # backward pass.
            self.comm_handler.increment_messaging_index(
                sending=True)

    @mapaean_original
    def run_forward_4times(self):
        # 만약 마지막 stage라면 stack을 이용해 뒤부터 pop해야 하므로, tensors를 백업해 놓는다.
        if self.is_criterion:
            self.tensor_stack.clear()  # 실제로는 불필요
            self.run_forward()
            self.tensor_stack.append(self.tensors)
            self.run_forward()
            self.tensor_stack.append(self.tensors)
            self.run_forward()
            self.tensor_stack.append(self.tensors)
            self.run_forward()
        else:
            self.run_forward()
            self.run_forward()
            self.run_forward()
            self.run_forward()

    def run_forward(self, recompute_step=False):
        """Run forward pass.
        """
        # Receive tensors from previous worker.
        self.receive_tensors_forward()
        tensors = self.tensors[-1]

        # Run forward pass.
        self._run_forward(tensors)

        # Send tensors forward.
        self.send_tensors_forward()
        if self.verbose_freq > 0 and self.forward_minibatch_id % self.verbose_freq == 0:
            self.forward_stats.print_stats()
        self.forward_stats.reset_stats()
        self.forward_minibatch_id += 1

    def _run_forward(self, tensors):
        # Perform forward pass through model (self.modules_with_dependencies already
        # has modules in topological order).
        modules = self.modules_with_dependencies.modules()
        all_input_names = self.modules_with_dependencies.all_input_names()
        all_output_names = self.modules_with_dependencies.all_output_names()
        for i, (module, input_names, output_names) in \
                enumerate(zip(modules, all_input_names, all_output_names)):
            if i == (len(modules) - 1) and self.is_criterion:
                # If layer is criterion (loss).
                if self.model_type == SPEECH_TO_TEXT:
                    output = tensors["output"].transpose(0, 1).float()
                    output_sizes = tensors["output_sizes"].cpu()
                    target = tensors["target"].cpu()
                    target_sizes = tensors["target_length"].cpu()
                    input0_size = tensors["input0_size"].cpu()
                    module_outputs = [module(output, target, output_sizes, target_sizes) / input0_size[0]]
                else:
                    module_outputs = [module(tensors[input_name],
                                             tensors["target"])
                                      for input_name in input_names]
                    module_outputs = [sum(module_outputs)]
            else:
                # If layer is non-criterion.
                module_outputs = module(*[tensors[input_name]
                                          for input_name in input_names])
                if not isinstance(module_outputs, tuple):
                    module_outputs = (module_outputs,)
                module_outputs = list(module_outputs)

            for (output_name, module_output) in zip(output_names, module_outputs):
                tensors[output_name] = module_output

        # 마지막 stage에서만 중요한 부분
        self.output = tensors[input_names[0]]
        if self.is_criterion and self.model_type == TRANSLATION:
            loss_per_batch = tensors[output_names[0]] * tensors[self.criterion_input_name].size(1)
            loss_per_token = loss_per_batch / tensors["target_length"][0].item()
            self.loss = loss_per_token
        elif self.is_criterion:
            self.loss = tensors[output_names[0]]
        else:
            self.loss = 1

    @mapaean_original
    def run_backward_4times(self):
        # 모든 gradients를 합산한 뒤 optimizer에서 한 번에 step해야 제대로 sync가 이루어진다.
        if self.is_criterion:
            self.run_backward(0)
            self.tensors = self.tensor_stack.pop()
            self.run_backward(1)
            self.tensors = self.tensor_stack.pop()
            self.run_backward(2)
            self.tensors = self.tensor_stack.pop()
            self.run_backward(3)
        else:
            self.run_backward(0)
            self.run_backward(1)
            self.run_backward(2)
            self.run_backward(3)

    def run_backward(self, microbatch_count: int):
        # Receive input gradients needed for backward pass.
        self.receive_tensors_backward()
        # Backward pass through modules in reverse order.
        inputs = {}
        outputs = {}
        input_gradients = {}
        output_gradients = {}

        # Get input and output names spanning all modules in this stage.
        all_input_names_set = set()
        all_output_names_set = set()

        modules = self.modules_with_dependencies.modules()
        all_input_names = self.modules_with_dependencies.all_input_names()
        all_output_names = self.modules_with_dependencies.all_output_names()

        for (input_names, output_names) in zip(all_input_names, all_output_names):
            for input_name in input_names:
                all_input_names_set.add(input_name)
            for output_name in output_names:
                all_output_names_set.add(output_name)

        tensors = self.tensors.pop(0)
        # Set inputs, outputs, and output_gradients.
        # Only set outputs/output_gradients for tensors that are not inputs of
        # other modules in this stage.
        # Similarly, only set inputs for tensors that are not outputs of other
        # modules in this stage.
        for (module, input_names, output_names) in \
            zip(reversed(modules), reversed(all_input_names), reversed(all_output_names)):
            for output_name in output_names:
                if output_name not in all_input_names_set:
                    if output_name not in self.gradients:
                        output_gradients[output_name] = None
                    else:
                        output_gradients[output_name] = self.gradients[output_name]
                    if tensors[output_name].requires_grad:
                        outputs[output_name] = tensors[output_name]
            for input_name in input_names:
                if input_name not in all_output_names_set:
                    inputs[input_name] = tensors[input_name]

        # Hook to record input gradients.
        def hook_wrapper(input_name):
            def hook(input_gradient):
                input_gradients[input_name] = input_gradient
            return hook

        for input_name in inputs:
            if input_name != "input0" and input_name != "input1" and input_name != "input2" \
                    and inputs[input_name].requires_grad:
                inputs[input_name].register_hook(hook_wrapper(input_name))

        if "loss" in outputs:
            outputs["loss"] *= self.loss_scale

        # Perform backward pass.
        torch.autograd.backward(tuple([outputs[output_name] for output_name in outputs]),
          grad_tensors=tuple([output_gradients[output_name] for output_name in outputs]),
          retain_graph=(microbatch_count != 3))  # 마지막 microbatch일 경우 다음 backward를 준비한다.

        # Input tensors don't need gradients.
        for input_name in inputs:
            if not inputs[input_name].requires_grad:
                self.gradients[input_name] = inputs[input_name]
                continue

            if input_name != "input0" and input_name != "input1" and input_name != "input2" and input_name != "input":
                self.gradients[input_name] = input_gradients[input_name]

        # Send output gradients.
        self.send_tensors_backward()
        if self.verbose_freq > 0 and self.backward_minibatch_id % self.verbose_freq == 0:
            self.backward_stats.print_stats()
        self.backward_stats.reset_stats()
        self.backward_minibatch_id += 1

    def num_tokens(self):
        return self.tensors[-1]["target_length"][0].item()

    def run_ack(self):
        # No need for ack if running on a single worker.
        if self.rank is None:
            return

        # Receive ack from next stage. Send ack to previous stage.
        if self.stage < (self.num_stages-1):
            self.comm_handler.recv(
                "ack",
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)
        if self.stage > 0:
            self.comm_handler.send(
                "ack",
                torch.zeros(self.tensor_shapes["ack"],
                            dtype=torch.int64).cuda(),
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)

            # Used to track where to receive forward from.
            self.comm_handler.increment_messaging_index(sending=True)

        self.backward_minibatch_id += 1

    def wait(self):
        if self.comm_handler is not None:
            self.comm_handler.wait()

    def num_iterations(self, loader_size):
        """ Determines number of iterations for this stage

        TODO: don't currently support uneven configurations.
        """
        if self.stage == 0 or self.stage is None:
            return loader_size

        num_iterations = loader_size * self.num_ranks_in_first_stage
        assert num_iterations % self.num_ranks_in_stage == 0
        num_iterations = num_iterations // self.num_ranks_in_stage

        return num_iterations

    def get_adjusted_learning_rate(self, base_lr):
        if self.stage == 0:
            return base_lr

        adjusted_lr = float(base_lr) * float(self.num_ranks_in_stage) \
                      / float(self.num_ranks_in_first_stage)

        return adjusted_lr
