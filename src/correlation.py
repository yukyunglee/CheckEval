import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau
from typing import List, Dict, Tuple
from prettytable import PrettyTable

class Correlation:
    @staticmethod
    def calculate_correlation(pred_score: List[float], human_score: List[float]) -> Tuple[float, float, float]:
        return (
            pearsonr(pred_score, human_score)[0],
            spearmanr(pred_score, human_score)[0],
            kendalltau(pred_score, human_score)[0]
        )

    @staticmethod
    def print_correlations(result: Dict[str, List[float]]) -> None:
        table = PrettyTable(['Dimensions', 'Pearson', 'Spearman', 'Kendall'])
        for dim, scores in result.items():
            table.add_row([dim] + [round(score, 6) for score in scores])
        print(table)

    @classmethod
    def correlation_for_summ(cls, data: pd.DataFrame, col_name: str, dimension: str) -> None:
        dimensions = [dimension]
        
        # Sample level correlation
        print('\n ********** Sample Level Correlations *********')
        result = {}
        for dim in dimensions:
            pred_score = data[col_name].tolist()
            human_score = data['scores'].apply(lambda x: eval(x)[dim]).tolist()
            result[dim] = cls.calculate_correlation(pred_score, human_score)
        cls.print_correlations(result)
        
        # Summary level correlation
        print('\n ********* Summary Level Correlations *********')
        result = {}
        docs = data['doc_id'].unique()
        for dim in dimensions:
            valid_cnt = 0
            corr_sum = [0, 0, 0]
            for doc_idx in docs:
                doc_data = data[data['doc_id'] == doc_idx]
                pred_score = doc_data[col_name].tolist()
                human_score = doc_data['scores'].apply(lambda x: eval(x)[dim]).tolist()
                if len(set(pred_score)) == 1 or len(set(human_score)) == 1:
                    continue
                corr = cls.calculate_correlation(pred_score, human_score)
                corr_sum = [sum(x) for x in zip(corr_sum, corr)]
                valid_cnt += 1
            result[dim] = [score / valid_cnt for score in corr_sum] if valid_cnt > 0 else [0, 0, 0]
        cls.print_correlations(result)
        
        # System level correlations
        print('\n ********** System Level Correlations *********')
        result = {}
        systems = data['system_id'].unique()
        for dim in dimensions:
            pred_score, human_score = [], []
            for system_idx in systems:
                system_data = data[data['system_id'] == system_idx]
                pred_score.append(system_data[col_name].mean())
                human_score.append(system_data['scores'].apply(lambda x: eval(x)[dim]).mean())
            result[dim] = cls.calculate_correlation(pred_score, human_score)
        cls.print_correlations(result)

def correlation_for_summ(data: pd.DataFrame, col_name: str, dimension: str) -> None:
    Correlation.correlation_for_summ(data, col_name, dimension)