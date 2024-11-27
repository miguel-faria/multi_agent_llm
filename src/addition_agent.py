#! /usr/bin/env python

from typing import List
from mt_agent import MTAgent


class AdditionAgent(MTAgent):

	def get_context(self, history: List[List[str]], answer: str = '') -> str:

		context = self.agent_profile['role_prompt']

		context += '\nConsidering your role as a %s, provide a detailed answer to the following instruction.\n\n' % self.agent_profile['profession']   # Agent specific information
		for elem in history:
			context += elem[0] + '\n' + elem[1] + '\n'

		return context

