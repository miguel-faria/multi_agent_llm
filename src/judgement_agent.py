#! /usr/bin/env python

from typing import List
from vllm import SamplingParams
from mt_agent import MTAgent


class JudgementAgent(MTAgent):

	def get_context(self, history: List[str], answer: str = '', instruction: str = ''):

		context = self.agent_profile['role_prompt']

		context += '\nConsidering your role as a %s, and the initial context and instruction, evaluate the quality of the following answer.\n\nContext:\n' % self.agent_profile['profession']   # Agent specific information
		context += history[0] + '\n\nInstruction:\n' + instruction + '\n\n' 'Answer:\n %s\n' % answer

		return context

	def generate(self, history: List[str], gen_params: SamplingParams, answer: str = '', instruction: str = '') -> str:

		generation_prompt = self.get_context(history, answer, instruction)
		outputs = self.gen_model.generate(generation_prompt, sampling_params=gen_params)

		return outputs[0].outputs[0].text
