import argparse
from datasets import Dataset
import pandas as pd
from openai import AsyncOpenAI
import asyncio
import nest_asyncio
import os
import yaml
import time
import re
from tqdm import tqdm

nest_asyncio.apply()

class vLLMProcessor:
    def __init__(self, api_key, base_url, model, batch=False, batch_size=5):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.batch = batch
        self.batch_size = batch_size

    async def chat_completion(self, client, input_data, **kwargs):
        messages = [{"role": "user", "content": input_data}]
        response = await client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return response

    async def run_chat_completions(self, client, prompts: list, **kwargs):
        tasks = [self.chat_completion(client, prompt, **kwargs) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        return responses

    async def process(self, prompts, **kwargs):
        async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            responses = await self.run_chat_completions(client, prompts, **kwargs)
        return responses

class OpenaiProcessor:
    def __init__(self, api_key, model, batch=False, batch_size=5):
        self.api_key = api_key
        self.model = model
        self.batch = batch
        self.batch_size = batch_size

    async def chat_completion(self, client, input_data, **kwargs):
        messages = [{"role": "user", "content": input_data}]
        response = await client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return response

    async def run_chat_completions(self, client, prompts: list, **kwargs):
        tasks = [self.chat_completion(client, prompt, **kwargs) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        return responses

    async def process(self, prompts, **kwargs):
        async with AsyncOpenAI(api_key=self.api_key) as client:
            responses = await self.run_chat_completions(client, prompts, **kwargs)
        return responses

def make_question_list(q):
    questions = q.strip().split('?')
    questions = [q.strip() for q in questions if q.strip()]
    return '\n'.join([f"Q{idx}: {question}?" for idx, question in enumerate(questions, 1)])

def extract_answers(response):
    matches = re.findall(r'Q\d+: (Yes|No)', response)
    return [1 if answer == 'Yes' else 0 for answer in matches]

def _split_into_batches(data, num_batches):
    batch_size = len(data) // num_batches
    batches = [data[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    if len(data) % num_batches != 0:
        batches.append(data[num_batches * batch_size:])
    return batches

summeval_template = '''### Task Overview ###\nYour task is to read a provided news article and its summary, then answer 'yes' or 'no' to specific questions. These questions will relate to a particular aspect of the summary.\n\n### Aspect Definition ###\n<aspect> - <definition>\n\n### Instructions ###\n1. Read these instructions thoroughly.\n2. Carefully read both the Article and the Summary.\n3. Understand the given questions and the definition of the <aspect>.\n4. Respond to each question with 'yes' or 'no'. Base your answers on a clear rationale.\n5. Follow the specified format for your answers.\n\n### Answer Format ###\nQ1: [Your Answer] \nQ2: [Your Answer] \n...\n\n# Article #\n"<source>"\n\n# Summary #\n"<summary>"\n\n# Questions # \n<questions>\n\n# Response\nProvide your answers to the given questions, following the specified Answer Format.\n'''

topical_chat_template = '''### Task Overview ###\nYou will be given a conversation between two individuals. You will then be given one potential response for the next turn in the conversation. The response concerns an interesting fact, which will be provided as well.\nYour task is to read a provided conversation history, corresponding fact and response, then answer 'yes' or 'no' to specific questions. These questions will relate to a particular aspect of the response.\n\n### Aspect Definition ###\n<aspect> - <definition>\n\n### Instructions ###\n1. Read these instructions thoroughly.\n2. Carefully read the Conversation History, the Corresponding Fact and the Response.\n3. Understand the given questions and the definition of the <aspect>.\n4. Respond to each question with 'yes' or 'no'. Base your answers on a clear rationale.\n5. Follow the specified format for your answers.\n\n### Answer Format ###\nQ1: [Your Answer] \nQ2: [Your Answer] \n...\n\n# Conversation History #\n"<document>"\n\n# Corresponding Fact #\n"<fact>"\n\n# Response #\n"<response>"\n\n# Questions # \n<questions>\n\n# Your answer\nProvide your answers to the given questions, following the specified Answer Format.\n'''

def main(data_path, base_url, model, aspect_list, question_version, save_dir, template_type, processor_type):
    processor = vLLMProcessor(api_key="", base_url=base_url, model=model) if processor_type == 'vllm' else OpenaiProcessor(api_key="", model=model)
    data = pd.read_csv(data_path)
    data = Dataset.from_pandas(data)

    template = summeval_template if template_type == "summeval" else topical_chat_template

    for aspect in tqdm(aspect_list):
        question_path = f"{template_type}/{aspect}_{question_version}.yaml"
        with open(question_path, 'r') as file:
            question_data = yaml.safe_load(file)

        for key, key_questions in question_data['sub_aspect'].items():
            question_list = make_question_list(key_questions)
            definition = question_data['definition'][template_type]

            def _mapping(x):
                prompt = template.replace("<aspect>", aspect).replace("<definition>", definition)
                prompt = prompt.replace("<source>", x.get('source', '')).replace("<summary>", x.get('system_output', ''))
                prompt = prompt.replace("<questions>", question_list)
                return {"prompt": prompt}

            data = data.map(_mapping)
            num_batches = 1
            data_batches = _split_into_batches(data, num_batches)
            responses = asyncio.run(processor.process(data["prompt"]))
            data = data.add_column(f"{aspect}_response", [r.choices[0].message.content for r in responses])
            data.to_csv(os.path.join(save_dir, f"{aspect}_responses.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some data using OpenaiProcessor or vLLMProcessor.')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--base_url', type=str, default="")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--aspects', nargs='+', required=True)
    parser.add_argument('--question_version', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--template_type', type=str, choices=['summeval', 'topical_chat'], required=True)
    parser.add_argument('--processor_type', type=str, choices=['openai', 'vllm'], required=True)
    args = parser.parse_args()
    main(args.data_path, args.base_url, args.model, args.aspects, args.question_version, args.save_dir, args.template_type, args.processor_type)
