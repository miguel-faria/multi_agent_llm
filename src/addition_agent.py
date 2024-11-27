#! /usr/bin/env python
import re

from torch.nn.functional import softmax
from torch import Tensor
from typing import Dict, List, Union, Tuple
from vllm import LLM, SamplingParams
from societies_agents.doc_mt.mt_agent import MTAgent
from pandas import DataFrame
from tqdm import tqdm


class AdditionAgent(MTAgent):

	def get_context(self, history: List[str], answer: str = '') -> str:

		context = self.agent_profile['role_prompt']

		context += '\n'   # Agent specific information

		for elem in history:

			e_context = elem[0]
			e_instruct = elem[1]
			context += ''

		return context
