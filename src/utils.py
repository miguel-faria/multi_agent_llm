#! /usr/bin/env python

from vllm import SamplingParams
from addition_agent import AdditionAgent
from subtraction_agent import SubtractionAgent
from critique_agent import CritiqueAgent
from judgement_agent import JudgementAgent


def addition_by_subtraction(addition_agent: AdditionAgent, subtraction_agent: SubtractionAgent, init_context: str, instruction: str, max_iterations: int,
                            generation_temp: int = 0.5, top_k: int = 5, max_tokens: int = 1000) -> str:

	history = [[init_context, instruction]]
	answer = ''
	gen_params = SamplingParams(
			temperature=generation_temp,
			top_k=top_k,
			max_tokens=max_tokens)

	for _ in range(max_iterations):

		new_answer = addition_agent.generate(history, gen_params)
		feedback = subtraction_agent.generate(history, gen_params, new_answer)
		history.append([new_answer, feedback])

		if answer == new_answer:
			break

		answer = new_answer

	return answer


def trilateral_collab(act_agent: AdditionAgent, critique_agent: CritiqueAgent, judge_agent: JudgementAgent, init_context: str, instruction: str, max_iterations: int,
                      generation_temp: int = 0.5, top_k: int = 5, max_tokens: int = 1000) -> str:

	history = [[init_context, instruction]]
	answer = ''
	gen_params = SamplingParams(
			temperature=generation_temp,
			top_k=top_k,
			max_tokens=max_tokens)

	for it in range(max_iterations):

		answer = act_agent.generate(history, gen_params)
		feedback = critique_agent.generate(history, gen_params, answer)
		history.append([answer, feedback])

		if it > 1:
			is_good_answer = judge_agent.generate([init_context], gen_params, answer, instruction)
			if is_good_answer:
				break

	return answer
