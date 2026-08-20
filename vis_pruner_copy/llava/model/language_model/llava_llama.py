#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import time
import json
import os

# Where per-question timing lines get written. Overridable via the
# LLAVA_TIMING_FILE env var (e.g. by scripts/time_sqa_sweep.py, so each
# subprocess run writes to its own scratch file that the sweep script reads
# back and folds into its single combined output). Looked up at write time,
# not import time, since the env var may be set after this module is
# imported (imports happen before argparse runs in the eval scripts).
DEFAULT_TIMING_FILE = "/workspace/GPU_Profiling/results/timing/llava_llama_timing.json"

class LlavaLlamaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaLlamaConfig

    def __init__(self, config, visual_token_num, important_ratio):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # [VisPruner] Visual token pruning config
        self.visual_token_num = visual_token_num
        self.important_ratio = important_ratio

        # Timing: set (to a list) by generate() for the duration of one
        # question, so forward()/prepare_inputs_for_generation() have
        # somewhere to record into. None outside of a generate() call, so
        # calls made outside that path are silently not recorded.
        # forward() is called once per decode step, so its timing uses CUDA
        # events (record on the stream, no sync) rather than perf_counter +
        # torch.cuda.synchronize() per call -- syncing inside that loop would
        # drain the pipeline every step and inflate exactly the fast kernels
        # we're trying to measure. Events get read out with a single sync
        # after generate() returns.
        self._timing_forward_events = None
        self._timing_prepare_inputs_calls = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    # [VisPruner] Visual token number
    def get_visual_token_num(self):
        return self.visual_token_num

    # [VisPruner] Important ratio
    def get_important_ratio(self):
        return self.important_ratio

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        timing_active = self._timing_forward_events is not None
        if timing_active:
            prep_start_evt = torch.cuda.Event(enable_timing=True)
            prep_end_evt = torch.cuda.Event(enable_timing=True)
            fwd_start_evt = torch.cuda.Event(enable_timing=True)
            fwd_end_evt = torch.cuda.Event(enable_timing=True)
            prep_start_evt.record()

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        if timing_active:
            prep_end_evt.record()
            fwd_start_evt.record()

        result = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        if timing_active:
            fwd_end_evt.record()
            self._timing_forward_events.append(
                (prep_start_evt, prep_end_evt, fwd_start_evt, fwd_end_evt)
            )

        return result
        
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        # Open the recording lists for this one question. forward() and
        # prepare_inputs_for_generation() append into these for every step
        # of the generation loop below; we fold it all into one JSON line
        # once generate() finishes.
        self._timing_forward_events = []
        self._timing_prepare_inputs_calls = []

        torch.cuda.synchronize()
        start_time_prep = time.perf_counter()

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                visual_token_num
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            visual_token_num = 0

        torch.cuda.synchronize()
        end_time_prep = time.perf_counter()
        elapsed_time_prep = end_time_prep - start_time_prep

        start_time_generate = time.perf_counter()
        result = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        # Single sync for the whole question: forward()'s CUDA events were
        # recorded on the stream without syncing per call (see forward()),
        # so nothing is safe to read off them until now. elapsed_time() on
        # an event pair errors out if the corresponding work hasn't actually
        # completed on the GPU yet.
        torch.cuda.synchronize()
        end_time_generate = time.perf_counter()
        elapsed_time_generate = end_time_generate - start_time_generate

        # elapsed_time() returns milliseconds; convert to seconds to match
        # every other duration in this file.
        forward_calls = [
            {
                "multimodal_prep_time": prep_start_evt.elapsed_time(prep_end_evt) / 1000.0,
                "lm_forward_time": fwd_start_evt.elapsed_time(fwd_end_evt) / 1000.0,
            }
            for prep_start_evt, prep_end_evt, fwd_start_evt, fwd_end_evt in self._timing_forward_events
        ]

        output_dictionary = {
            "multimodal_prep_time_generate": elapsed_time_prep,
            "generate_time": elapsed_time_generate,
            "forward_calls": forward_calls,
            "prepare_inputs_calls": self._timing_prepare_inputs_calls,
        }

        timing_file_path = os.environ.get("LLAVA_TIMING_FILE", DEFAULT_TIMING_FILE)
        os.makedirs(os.path.dirname(timing_file_path), exist_ok=True)
        with open(timing_file_path, "a+") as f:
            json.dump(output_dictionary, f)
            f.write("\n")

        self._timing_forward_events = None
        self._timing_prepare_inputs_calls = None

        return result, visual_token_num

        

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        # Plain dict assignment below, no GPU kernels involved -- no sync
        # needed for correctness, and syncing here (once per decode step)
        # would just drain the pipeline for no timing benefit.
        start_time_prepare_inputs = time.perf_counter()

        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes

        end_time_prepare_inputs = time.perf_counter()
        elapsed_time_prepare_inputs = end_time_prepare_inputs - start_time_prepare_inputs

        if self._timing_prepare_inputs_calls is not None:
            self._timing_prepare_inputs_calls.append({
                "input_preparation_time": elapsed_time_prepare_inputs,
            })

        return inputs

AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModelForCausalLM.register(LlavaLlamaConfig, LlavaLlamaForCausalLM)
