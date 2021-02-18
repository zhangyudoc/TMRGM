from pycocoevalcap.eval import calculate_metrics
import json


def evaluation_captions(result):
    test = result
    datasetGTS = {'annotations': []}
    datasetRES = {'annotations': []}

    for i, image_id in enumerate(test):
        array = []
        for each in test[image_id]['Pred Sent']:
            array.append(test[image_id]['Pred Sent'][each])
        pred_sent = '. '.join(array)

        array = []
        for each in test[image_id]['Real Sent']:
            sent = test[image_id]['Real Sent'][each]
            if len(sent) != 0:
                array.append(sent)
        real_sent = '. '.join(array)
        datasetGTS['annotations'].append({
            'image_id': i,
            'caption': real_sent
        })
        datasetRES['annotations'].append({
            'image_id': i,
            'caption': pred_sent
        })

    rng = range(len(test))
    eva_scores = calculate_metrics(rng, datasetGTS, datasetRES)
    print(eva_scores)
    test_result_dir = './results/BLEUS.txt'
    with open(test_result_dir, 'a') as f:
        f.writelines(str(eva_scores))
        f.writelines('\n')
    Bleu = eva_scores['Bleu_1']
    return Bleu


if __name__ == '__main__':
    with open('./results/test_result.json', 'r') as load_f:
        load_dict = json.load(load_f)
    result = load_dict
    print(result.keys())
    for i in list(result.keys()):
        if result[i]['Pred Tags'][0] == 'normal':
            result[i]['Pred Sent']= {'0': 'no acute cardiopulmonary abnormality', '1': 'no active disease', '2': 'no focal consolidation pleural effusion or pneumothorax', '3': 'the lungs are clear', '4': 'the cardiomediastinal silhouette is within normal limits', '5': 'the heart is normal in size'}
   
    Bleu = evaluation_captions(result)

# results[id] = {
#     'Real Tags': self.tagger.inv_tags2array(real_tag),
#     'Pred Tags': self.tagger.array2tags(torch.topk(pred_tag, self.args.k)[1].cpu().detach().numpy()),
#     'Pred Sent': pred_sentences[id],
#     'Real Sent': real_sentences[id]
# }
