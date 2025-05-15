import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any

def parse_output(s):
    parsed_list = [1 if item.lower() == 'yes' else 0 for item in re.findall(r'Yes|No', s, re.IGNORECASE)]
    return parsed_list


class Aggregator:
    @staticmethod
    def calculate_metrics(binary_list: List[List[int]]) -> Dict[str, float]:
        averages = binary_list
        
        return {
            'avg': sum(averages) / len(averages) if averages else 0,
            'sum': sum(averages),
            'proportion': sum(averages) / len(averages) if len(averages) > 0 else 0,
            'std_dev': np.std(averages) if averages else 0,
            'mode': max(set(averages), key=averages.count) if averages else 0,
            'median': np.median(averages) if averages else 0,
            'cumulative_sum': np.cumsum(averages)[-1] if averages else 0
        }

    @classmethod
    def aggregate(cls, data: pd.DataFrame, aspect_list: List[str]) -> pd.DataFrame:
        df = data.copy()
        
        for aspect in aspect_list:
            df[f'{aspect}_binary'] = df[f'{aspect}_response'].apply(parse_output)
            
            metrics = df[f'{aspect}_binary'].apply(cls.calculate_metrics)
            metrics_list = list(metrics[0].keys())
            metrics = pd.DataFrame(metrics)
            
            for k in metrics_list:
                df[f'{aspect}_{k}'] = metrics[f'{aspect}_binary'].apply(lambda x: x[k])
        
        return df

def aggregation(data: pd.DataFrame, aspect_list: List[str]) -> pd.DataFrame:
    return Aggregator.aggregate(data, aspect_list)