#! /usr/bin/env python
import re

from torch.nn.functional import softmax
from torch import Tensor
from typing import Dict, List, Union, Tuple
from vllm import LLM, SamplingParams
from pandas import DataFrame
from tqdm import tqdm


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

	def get_context(self, history: List[str], answer: str = ''):
		raise NotImplementedError("Method 'get_context' is not implemented in the base class, subclasses should implement it.")

	def generate(self, history: List[str], params: SamplingParams) -> str:
		raise NotImplementedError("Method 'generate' is not implemented in the base class, subclasses should implement it.")

