import sys, argparse, string
import csv
import warnings
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def fscore(prediction, ground_truth, modelname):
    # Concept stats
    min_concepts = sys.maxsize
    max_concepts = 0
    total_concepts = 0
    concepts_distrib = {}
    #使用readfile函数进行读取，将图片ID内容赋予candidate_pairs
    candidate_pairs = prediction
    gt_pairs = ground_truth
    # 定义最高分和当前分数，每个对的最高分为1，所以最高总分为长度值
    max_score = len(gt_pairs)
    current_score = 0
    pcurrent_score = 0
    rcurrent_score = 0
    if len(candidate_pairs) != len(gt_pairs):
        print('ERROR : Candidate does not contain the same number of entries as the ground truth!')
        exit(1)

    # print('Processing concept sets...\n********************************')

    i = 0
    for image_key in candidate_pairs:
        #print(image_key)
        candidate_concepts = candidate_pairs[image_key].upper()
        gt_concepts = gt_pairs[image_key].upper()
        if gt_concepts.strip() == '':
            gt_concepts = []
        else:
            gt_concepts = gt_concepts.split(';')

        if candidate_concepts.strip() == '':
            candidate_concepts = []
        else:
            candidate_concepts = candidate_concepts.split(';')

        if len(gt_concepts) == 0:
            max_score -= 1
        else:
            total_concepts += len(gt_concepts)
            all_concepts = sorted(list(set(gt_concepts + candidate_concepts)))

            # Calculate F1 score for the current concepts
            y_true = [int(concept in gt_concepts) for concept in all_concepts]
            y_pred = [int(concept in candidate_concepts) for concept in all_concepts]
            #print(y_true,y_pred)

            pscore = precision_score(y_true, y_pred, average='binary')
            rscore = recall_score(y_true, y_pred, average='binary')
            f1score = f1_score(y_true, y_pred, average='binary')

            # Increase calculated score
            pcurrent_score += pscore
            rcurrent_score += rscore
            current_score += f1score

            #print(f1_score,current_score)
        nb_concepts = str(len(gt_concepts))
       # print(nb_concepts,concepts_distrib)
        if nb_concepts not in concepts_distrib:
            concepts_distrib[nb_concepts] = 1
        else:
            concepts_distrib[nb_concepts] += 1

        if len(gt_concepts) > max_concepts:
            max_concepts = len(gt_concepts)

        if len(gt_concepts) < min_concepts:
            min_concepts = len(gt_concepts)
       # print(gt_concepts,len(gt_concepts),max_concepts,min_concepts)
        # Progress display
        i += 1
        if i % 1000 == 0:
            print(i, '/', len(gt_pairs), ' concept sets processed...')

    # Print stats
    print('Concept statistics\n********************************')
    print('Number of concepts distribution')
    print_dict_sorted_num(concepts_distrib)
    print('Least concepts in set :', min_concepts)
    print('Most concepts in set :', max_concepts)
    print('Average concepts in set :', total_concepts / len(candidate_pairs))

    # Print evaluation result
    print('Final result\n********************************')
    print('Obtained score :', current_score, '/', max_score)
    print('Mean score over all concept sets :', current_score / max_score)
    print('p:', pcurrent_score / max_score)
    print('r:', rcurrent_score / max_score)
    scores  = 'P_score:'+str(pcurrent_score / max_score) + '---R_score:' +str(rcurrent_score / max_score) + '---F1_score:' +str(current_score / max_score)
    out_path = './report_v4_models/v4/'+ modelname + '/' + modelname + '_prediction_gt.txt'
    score_path = './report_v4_models/v4/'+ modelname + '/' + modelname + '_PRFscores.txt'
    with open(out_path, 'a') as f:
        f.write('prediction:' + str(prediction))
        f.write('\n')
        f.write('ground_truth:' + str(ground_truth))
        f.write('\n')
    with open(score_path, 'a') as f:
        f.write('\n')
        f.write(scores)
    return current_score / max_score


def recall_k(prediction, ground_truth, modelname,k):
    #使用readfile函数进行读取，将图片ID内容赋予candidate_pairs
    candidate_pairs = prediction
    gt_pairs = ground_truth
    # 定义最高分和当前分数，每个对的最高分为1，所以最高总分为长度值
    max_score = len(gt_pairs)
    rcurrent_score = 0
    if len(candidate_pairs) != len(gt_pairs):
        print('ERROR : Candidate does not contain the same number of entries as the ground truth!')
        exit(1)

    i = 0
    for image_key in candidate_pairs:
        #print(image_key)
        candidate_concepts = candidate_pairs[image_key].upper()
        gt_concepts = gt_pairs[image_key].upper()
        if gt_concepts.strip() == '':
            gt_concepts = []
        else:
            gt_concepts = gt_concepts.split(';')

        if candidate_concepts.strip() == '':
            candidate_concepts = []
        else:
            candidate_concepts = candidate_concepts.split(';')

        if len(gt_concepts) == 0:
            max_score -= 1
        else:
            all_concepts = sorted(list(set(gt_concepts + candidate_concepts)))

            # Calculate F1 score for the current concepts
            y_true = [int(concept in gt_concepts) for concept in all_concepts]
            y_pred = [int(concept in candidate_concepts) for concept in all_concepts]
            #print(y_true,y_pred)
            rscore = recall_score(y_true, y_pred, average='binary')
            rcurrent_score += rscore
        # Progress display
        i += 1
        if i % 1000 == 0:
            print(i, '/', len(gt_pairs), ' concept sets processed...')
    print_str = 'Recall_at_' + str(k) +': '
    print(print_str, rcurrent_score / max_score)
    score_path = './report_v4_models/v4/' + modelname + '/' + modelname + '_PRFscores.txt'
    with open(score_path, 'a') as f:
        f.write('---Recall_at_' + str(k) +':' + str(rcurrent_score / max_score))
    return rcurrent_score / max_score


def readfile(path):
    try:
        pairs = {}
        with open(path) as csvfile:
            #以tab作为分割符，分为两个部分
            reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                # We have an ID and a set of concepts (possibly empty)
                if len(row) == 2:
                    #将图片ID对应的描述文字赋予变量pairs
                    pairs[row[0]] = row[1]
                # We only have an ID
                elif len(row) == 1:
                    pairs[row[0]] = ''
                else:
                    print('File format is wrong, please check your run file')
                    exit(1)

        return pairs
    except FileNotFoundError:
        print('File "' + path + '" not found! Please check the path!')
        exit(1)

# Print 1-level key-value dictionary, sorted (with numeric key)
def print_dict_sorted_num(obj):
    keylist = [int(x) for x in list(obj.keys())]
    keylist.sort()
    for key in keylist:
        print(key, ':', obj[str(key)])
