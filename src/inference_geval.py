import argparse
from datasets import Dataset
import pandas as pd
from openai import AsyncOpenAI
import asyncio
import nest_asyncio
import os
import yaml
import time
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

def _split_into_batches(data, num_batches):
    batch_size = len(data) // num_batches
    batches = [data[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    if len(data) % num_batches != 0:
        batches.append(data[num_batches * batch_size:])
    return batches

def main(data_path, base_url, model, aspect_list, save_dir, template_path, processor_type):
    api_key = ""
    processor = vLLMProcessor(api_key=api_key, base_url=base_url, model=model) if processor_type == 'vllm' else OpenaiProcessor(api_key=api_key, model=model)

    data = pd.read_csv(data_path)
    data = Dataset.from_pandas(data)

    with open(template_path, 'r') as file:
        template = yaml.safe_load(file)

    for aspect in tqdm(aspect_list):
        def _mapping(x):
            prompt = template[f'{aspect}']
            prompt = prompt.replace('{{Document}}', x.get('source', ''))
            prompt = prompt.replace('{{Summary}}', x.get('system_output', ''))
            prompt = prompt.replace('{{Fact}}', x.get('context', ''))
            prompt = prompt.replace('{{Response}}', x.get('system_output', ''))
            return {"prompt": prompt}

        params = {"temperature": 0, "max_tokens": 500, "n": 1, "seed": 42}
        data_ds = data.map(_mapping)

        num_batches = 1
        data_batches = _split_into_batches(data_ds, num_batches)
        responses = []

        for i, batch in enumerate(data_batches):
            print(f"Processing batch {i+1}/{num_batches}")
            batch_prompts = batch['prompt']
            batch_responses = asyncio.run(processor.process(batch_prompts, **params))
            responses.extend(batch_responses)
            if i < num_batches - 1:
                time.sleep(0.2)

        original = [r.choices[0].message.content for r in responses]
        data_ds = data_ds.add_column(f"{aspect}_response", original)
        save_path = os.path.join(save_dir, f"{aspect}_responses.csv")
        data_ds.to_csv(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some data using OpenaiProcessor or vLLMProcessor.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input CSV data file.')
    parser.add_argument('--base_url', type=str, default="", help='Base URL for the OpenAI API.')
    parser.add_argument('--model', type=str, required=True, help='Model name or path for processing.')
    parser.add_argument('--aspects', nargs='+', required=True, help='List of aspects to process.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the output CSV files.')
    parser.add_argument('--template_path', type=str, required=True, help='Path to the prompt template YAML file.')
    parser.add_argument('--processor_type', type=str, choices=['openai', 'vllm'], required=True, help='Choose between OpenAI API or vLLM processor.')

    args = parser.parse_args()
    main(args.data_path, args.base_url, args.model, args.aspects, args.question_version, args.save_dir, args.template_path, args.processor_type)
