K = 60

def rerank_results(
    results: list[list[dict[any, any]]]
):
    weights = [1/len(results) for _ in range(len(results))]
    results_dict = {}
    for result_list, weight in zip(results, weights):
        for rank, result in enumerate(result_list):
            if result.text not in results_dict:
                score = weight/(K + rank + 1)
                results_dict[result.text] = score
            else:
                results_dict[result.text] += weight/(K + rank + 1)
    
    return sorted(results_dict.items(), key=lambda x: x[1], reverse=True)