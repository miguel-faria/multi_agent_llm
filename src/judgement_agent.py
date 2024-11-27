#! /usr/bin/env python
import re

from torch.nn.functional import softmax
from torch import Tensor
from typing import Dict, List, Union, Tuple
from vllm import LLM, SamplingParams
from societies_agents.doc_mt.mt_agent import MTAgent
from pandas import DataFrame
from tqdm import tqdm


class JudgementAgent(MTAgent):

	pass
