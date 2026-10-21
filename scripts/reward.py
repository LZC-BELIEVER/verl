"""Score the final numeric answer in the generated arithmetic solution."""
import re
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
 answer=solution_str.split('</think>')[-1].strip()
 numbers=re.findall(r'-?\d+(?:\.\d+)?',answer.replace(',',''))
 return float(bool(numbers) and float(numbers[-1])==int(ground_truth))
