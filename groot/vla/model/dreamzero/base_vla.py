from dataclasses import dataclass, field
from typing import Tuple
import os
import json

from hydra.utils import instantiate
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


def _load_safetensors_dir(pretrained_model_name_or_path: str) -> dict:
    from safetensors.torch import load_file

    safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
    safetensors_index_path = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")

    state_dict = {}
    if os.path.exists(safetensors_index_path):
        print(f"Loading sharded safetensors using index: {safetensors_index_path}")
        with open(safetensors_index_path, "r") as f:
            index = json.load(f)
        for shard_file in sorted(set(index["weight_map"].values())):
            shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
            print(f"Loading shard: {shard_path}")
            state_dict.update(load_file(shard_path))
    elif os.path.exists(safetensors_path):
        print(f"Loading weights from safetensors: {safetensors_path}")
        state_dict.update(load_file(safetensors_path))
    else:
        raise FileNotFoundError(
            f"No valid checkpoint found at {pretrained_model_name_or_path}. "
            "Expected model.safetensors or model.safetensors.index.json."
        )
    return state_dict


def _resolve_pretrained_model_path_from_exp_cfg(checkpoint_path: str) -> str | None:
    conf_path = os.path.join(checkpoint_path, "experiment_cfg", "conf.yaml")
    if not os.path.exists(conf_path):
        return None
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(conf_path)
        pretrained_model_path = cfg.get("pretrained_model_path", None)
    except Exception as exc:
        print(f"Warning: failed to read pretrained_model_path from {conf_path}: {exc}")
        return None

    if not pretrained_model_path:
        return None
    if not os.path.isabs(pretrained_model_path):
        pretrained_model_path = os.path.abspath(pretrained_model_path)
    return pretrained_model_path


@dataclass
class VLAConfig(PretrainedConfig):
    model_type = "vla"
    backbone_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Backbone configuration."}
    )

    action_head_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Action head configuration."}
    )

    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})

    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class VLA(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = VLAConfig
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    see discussion at https://nvidia.slack.com/archives/C07T1V7L886/p1732550624654139
    """

    def __init__(
        self,
        config: VLAConfig,
    ):
        print(f"[CODE_CHECK] VLA.__init__ from {__file__}", flush=True)
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)
        super().__init__(config)
        self.backbone = instantiate(config.backbone_cfg)
        self.action_head = instantiate(config.action_head_cfg)
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def validate_inputs(self, inputs):
        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] % self.action_horizon == 0
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):

        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:

        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def joint_video_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.joint_video_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def lazy_joint_video_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.lazy_joint_video_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def lazy_joint_video_action_causal(
        self,
        inputs: dict,
        latent_video: torch.Tensor | None = None,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.lazy_joint_video_action(backbone_outputs, action_inputs, latent_video=latent_video)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def lazy_joint_video_action_causal_gt_cond(
        self,
        inputs: dict,
        latent_video: torch.Tensor | None = None,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)

        action_head_outputs = self.action_head.lazy_joint_video_action_causal_gt_cond(backbone_outputs, action_inputs, latent_video=latent_video)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def lazy_joint_video_action_efficient(
        self,
        inputs: dict,
        prompt_embs: torch.Tensor | None = None,
        prompt_emb_nega: torch.Tensor | None = None,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.lazy_joint_video_action_efficient(backbone_outputs, action_inputs, prompt_embs=prompt_embs, prompt_emb_nega=prompt_emb_nega)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def gt_video_action_pred(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.gt_video_action_pred(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def get_language(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone.generate(backbone_inputs)
        return backbone_outputs

    def get_video(
        self,
        inputs: dict,
    ) -> BatchFeature:
        _, video_inputs = self.prepare_input(inputs)
        video_outputs = self.action_head.get_video(video_inputs)
        return video_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs


    @classmethod
    def from_pretrained_for_tuning(
        cls, 
        pretrained_model_name_or_path: str,
        config: VLAConfig = None,  # This config will now be USED
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        offload_state_dict: bool = True,
        lora_weights_path: str | None = None,
    ):
        if config is None:
            raise ValueError(
                "A `config` object must be provided to build the model structure."
            )

        import os
        import json
        import gc
        from safetensors.torch import load_file

        model = cls(config)

        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        safetensors_index_path = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")

        if os.path.exists(safetensors_index_path):
            with open(safetensors_index_path, 'r') as f:
                index = json.load(f)
            missing_keys_accum = set()
            unexpected_keys_accum = set()
            shard_files = sorted(set(index["weight_map"].values()))
            for shard_file in shard_files:
                shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                print(f"Loading shard: {shard_path}")
                shard_state_dict = load_file(shard_path)
                missing_keys, unexpected_keys = model.load_state_dict(shard_state_dict, strict=False)
                if missing_keys:
                    missing_keys_accum.update(missing_keys)
                if unexpected_keys:
                    unexpected_keys_accum.update(unexpected_keys)
                # Free shard immediately
                del shard_state_dict
                gc.collect()
            if missing_keys_accum:
                print(f"Missing keys when loading sharded pretrained weights: {sorted(missing_keys_accum)} ... total={len(missing_keys_accum)}")
            if unexpected_keys_accum:
                print(f"Unexpected keys when loading sharded pretrained weights: {sorted(unexpected_keys_accum)} ... total={len(unexpected_keys_accum)}")
            if not missing_keys_accum and not unexpected_keys_accum:
                print("Successfully loaded pretrained base weights (sharded)")
        elif os.path.exists(safetensors_path):
            # Handle single safetensors file
            print(f"Loading weights from safetensors: {safetensors_path}")
            state_dict = load_file(safetensors_path)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
            if not missing_keys and not unexpected_keys:
                print("Successfully loaded pretrained base weights")
        else:
            raise FileNotFoundError(
                f"No weights found at '{pretrained_model_name_or_path}'. "
                "Expected 'model.safetensors' or 'model.safetensors.index.json'."
            )

        def maybe_inject_lora_after_loading(reason: str):
            if (
                hasattr(model, "action_head")
                and hasattr(model.action_head, "inject_lora_after_loading")
                and model.action_head.config.defer_lora_injection
            ):
                print(f"Injecting LoRA adapters into action_head {reason}")
                model.action_head.inject_lora_after_loading()

        if lora_weights_path is not None:
            # 修复点 2：defer_lora_injection=True 时，LoRA adapter 不会在 __init__ 创建。
            # 因此必须先注入 adapter，再加载 lora_A/lora_B 权重，否则 LoRA 权重会被当作 unexpected keys 丢掉。
            maybe_inject_lora_after_loading("before loading LoRA weights")
            print(f"Loading LoRA weights from: {lora_weights_path}")
            model.load_lora_weight(lora_weights_path)
        else:
            maybe_inject_lora_after_loading("after loading pretrained weights")
        
        print(f"{cls}\n")
        return model

    @classmethod
    def load_lora(
        cls, 
        pretrained_model_name_or_path: str
    ): 
        print("loading lora@@@@@")

        lora_state_dict = _load_safetensors_dir(pretrained_model_name_or_path)
        
        # Load config
        print("loading config@@")
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = VLAConfig(**config_dict)
        print("loading model")

        pretrained_model_path = _resolve_pretrained_model_path_from_exp_cfg(pretrained_model_name_or_path)

        # 修复点 1：LoRA-only checkpoint 本身不包含完整 base。
        # 如果 experiment_cfg 记录了 pretrained_model_path，需要复现训练链路：
        # 先跳过 raw Wan DiT 组件加载，加载完整 pretrained base，再注入并加载 LoRA。
        ah_cfg = config.action_head_cfg
        inner = ah_cfg.get('config', ah_cfg) if isinstance(ah_cfg.get('config'), dict) else ah_cfg
        if pretrained_model_path is not None:
            if 'defer_lora_injection' in inner:
                inner['defer_lora_injection'] = True
                print("defer_lora_injection enabled for pretrained base loading")
            if 'skip_component_loading' in inner:
                inner['skip_component_loading'] = True
                print("skip_component_loading enabled for pretrained base loading")
        else:
            # Backward-compatible path for LoRA checkpoints that do not record a
            # full pretrained base: create LoRA layers during init and load raw
            # Wan components as before.
            if 'defer_lora_injection' in inner:
                inner['defer_lora_injection'] = False
                print("defer_lora_injection disabled for load_lora")
            if 'skip_component_loading' in inner:
                inner['skip_component_loading'] = False
                print("skip_component_loading disabled for load_lora")

        # Instantiate model.
        model = cls(config)

        if pretrained_model_path is not None:
            # 先加载训练时使用的 full pretrained base，例如 DreamZero-AgiBot。
            # 之后再创建 LoRA adapter 并加载当前 LoRA checkpoint，避免退化成 raw Wan base + LoRA。
            print(f"Loading pretrained base weights before LoRA: {pretrained_model_path}")
            pretrained_state_dict = _load_safetensors_dir(pretrained_model_path)
            missing_keys, unexpected_keys = model.load_state_dict(pretrained_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys when loading pretrained base weights: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained base weights: {unexpected_keys}")
            print("Successfully loaded pretrained base weights")

            if (hasattr(model, 'action_head')
                    and hasattr(model.action_head, 'inject_lora_after_loading')
                    and model.action_head.config.defer_lora_injection):
                model.action_head.inject_lora_after_loading()

        # Remove .base_layer from keys if present
        state_dict = lora_state_dict
        has_base_layer = any(".base_layer." in key for key in state_dict.keys())
        if has_base_layer:
            print("Removing '.base_layer' from state dict keys")
            state_dict = {k.replace(".base_layer.", "."): v for k, v in state_dict.items()}

        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
        if missing_keys:
            print(f"Missing keys when loading LoRA weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading LoRA weights: {unexpected_keys}")
        
        print("Successfully loaded LoRA weights")

        print(f"{cls}\n")
        return model

    def load_lora_weight(self, pretrained_model_name_or_path: str):
        """Load only LoRA weights from a pretrained model without loading config."""
        from safetensors.torch import load_file
        import os
        import json
        
        print(f"Loading LoRA weights from {pretrained_model_name_or_path}")
        
        # Check for different checkpoint formats
        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        safetensors_index_path = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")

        state_dict = {}
        if os.path.exists(safetensors_index_path):
            # Handle sharded safetensors
            print(f"Loading sharded safetensors using index: {safetensors_index_path}")
            
            with open(safetensors_index_path, 'r') as f:
                index = json.load(f)
            
            # Load each shard
            for shard_file in set(index["weight_map"].values()):
                shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                print(f"Loading shard: {shard_path}")
                shard_state_dict = load_file(shard_path)
                state_dict.update(shard_state_dict)
                
        elif os.path.exists(safetensors_path):
            # Handle single safetensors file
            print(f"Loading weights from safetensors: {safetensors_path}")
            state_dict.update(load_file(safetensors_path))
        else:
            raise FileNotFoundError(f"No valid checkpoint found at {pretrained_model_name_or_path}")
        
        print("Loading LoRA weights into existing model")

        def rewrite_lora_state_dict_keys(state_dict, pattern, repl):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace(pattern, repl)
                new_state_dict[new_k] = v
            return new_state_dict

        has_target_pattern = any("action_head.model.base_model.model" in key for key in state_dict.keys())
        
        if not has_target_pattern:
            print("Rewriting LoRA state dict keys from 'action_head.model' to 'action_head.model.base_model.model'")
            state_dict = rewrite_lora_state_dict_keys(
                state_dict,
                pattern="action_head.model",
                repl="action_head.model.base_model.model",
            )
        else:
            print("State dict already has 'action_head.model.base_model.model' pattern, skipping key rewrite")
        
        # Load only the weights into the existing model
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        print("Successfully loaded LoRA state dict")
            
        if missing_keys:
            print(f"Missing keys when loading LoRA weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading LoRA weights: {unexpected_keys}")
        
        print("Successfully loaded LoRA weights")

    @classmethod
    def from_config_with_lora_weights(
        cls,
        config: VLAConfig,
        pretrained_model_path: str,
    ):
        """Create VLA model from config and then load LoRA weights from pretrained model."""
        print(f"Creating VLA model from config and loading LoRA weights from {pretrained_model_path}")
        
        # 1. Create model from config (similar to vla.yaml)
        model = cls(config)
        print("Model created from config")
        
        # 2. Load LoRA weights into the created model
        model.load_lora_weight(pretrained_model_path)
        
        return model

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str,
        config: VLAConfig = None
    ):
        del config

        from safetensors.torch import load_file
        import os
        import json
        print("loading pretrained@@@@@")
        # Check for different checkpoint formats
        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        safetensors_index_path = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")

        state_dict = {}
        if os.path.exists(safetensors_index_path):
            # Handle sharded safetensors
            print(f"Loading sharded safetensors using index: {safetensors_index_path}")
            
            with open(safetensors_index_path, 'r') as f:
                index = json.load(f)
            
            # Load each shard
            for shard_file in set(index["weight_map"].values()):
                shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                print(f"Loading shard: {shard_path}")
                shard_state_dict = load_file(shard_path)
                state_dict.update(shard_state_dict)
                
        elif os.path.exists(safetensors_path):
            # Handle single safetensors file
            print(f"Loading weights from safetensors: {safetensors_path}")
            state_dict.update(load_file(safetensors_path))
        
        # Load config
        print("loading config@@")
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = VLAConfig(**config_dict)
        print("loading model")
        print("config.action_head_cfg", config.action_head_cfg)
        # Always disable defer_lora_injection
        # config.action_head_cfg is a dict, and defer_lora_injection is nested in config.action_head_cfg['config']
        if 'config' in config.action_head_cfg and isinstance(config.action_head_cfg['config'], dict):
            if 'defer_lora_injection' in config.action_head_cfg['config']:
                config.action_head_cfg['config']['defer_lora_injection'] = False
                print("config.action_head_cfg['config']['defer_lora_injection'] disabled (set to False)")
        elif 'defer_lora_injection' in config.action_head_cfg:
            config.action_head_cfg['defer_lora_injection'] = False
            print("config.action_head_cfg['defer_lora_injection'] disabled (set to False)")

        # Instantiate model
        model = cls(config)
        print("model", model)
        # Remove .base_layer from keys (e.g., 'action_head.model.base_model.model.blocks.19.self_attn.v.base_layer.bias' -> 'action_head.model.base_model.model.blocks.19.self_attn.v.bias')
        has_base_layer = any(".base_layer." in key for key in state_dict.keys())
        if has_base_layer:
            print("Removing '.base_layer' from state dict keys")
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace(".base_layer.", ".")
                new_state_dict[new_k] = v
            state_dict = new_state_dict

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
        if missing_keys:
            print(f"Missing keys when loading pretrained weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        
        print("Successfully loaded pretrained weights")

        print(f"{cls}\n")
        return model

    def post_initialize(self):
        self.action_head.post_initialize()

    def parallelize(self, device_mesh: DeviceMesh):
        self.action_head.parallelize(device_mesh=device_mesh)


class CotrainVLA(VLA):

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        if "cotrain" in inputs and inputs["cotrain"]:
            return self.backbone.cotrain(inputs)
        return super().forward(inputs)


def create_vla_with_pretrained_action_head(pretrained_vla_path: str, config: VLAConfig):
    # 1. Instantiate a new VLAModel
    vla = VLA(config)

    # 2. Load the pretrained VLAModel
    pretrained_vla = VLA.from_pretrained(pretrained_vla_path)

    # 3. Replace the action head in the new VLAModel with the pretrained action head
    vla.action_head = pretrained_vla.action_head

    # 4. Replace the action head config in the new VLAModel with the pretrained action head config
    vla.config.action_head_cfg = pretrained_vla.config.action_head_cfg

    # 5. Return the new VLAModel
    return vla


# register
AutoConfig.register("vla", VLAConfig)
AutoModel.register(VLAConfig, VLA)
