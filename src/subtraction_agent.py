#! /usr/bin/env python

from typing import List
from vllm import SamplingParams
from mt_agent import MTAgent


class SubtractionAgent(MTAgent):

	def get_context(self, history: List[List[str]], answer: str = ''):

		context = self.agent_profile['role_prompt']

		context += '\nConsidering your role as a %s, and using the provided context, provide feedback, revise and remove redundancies for the following answer.\n\nContext:\n' % self.agent_profile['profession']   # Agent specific information
		for elem in history:
			context += elem[0] + '\n' + elem[1] + '\n'

		context += '\nAnswer:\n %s\n' % answer
		return context

	def generate(self, history: List[List[str]], gen_params: SamplingParams, answer: str = '') -> str:

		generation_prompt = self.get_context(history, answer)
		outputs = self.gen_model.generate(generation_prompt, sampling_params=gen_params)

		return outputs[0].outputs[0].text
