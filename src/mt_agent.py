#! /usr/bin/env python

from typing import Dict, List, Union
from vllm import LLM, SamplingParams


class MTAgent:

	def __init__(self, gen_model: LLM, agent_name: str, agent_profile: Dict[str, Union[str, int, float]]):

		self._gen_model: LLM = gen_model
		self._agent_name: str = agent_name
		self._agent_profile: Dict[str, Union[str, int, float]] = agent_profile

	@property
	def agent_name(self) -> str:
		return self._agent_name

	@property
	def agent_profile(self) -> Dict[str, Union[str, int, float]]:
		return self._agent_profile

	@property
	def gen_model(self) -> LLM:
		return self._gen_model

	def get_context(self, history: List[List[str]], answer: str = ''):
		raise NotImplementedError("Method 'get_context' is not implemented in the base class, subclasses should implement it.")

	def generate(self, history: List[List[str]], gen_params: SamplingParams) -> str:

		generation_prompt = self.get_context(history)
		outputs = self.gen_model.generate(generation_prompt, sampling_params=gen_params)

		return outputs[0].outputs[0].text

