import numpy as np
import bottleneck as bn
import torch
import math
import time
import csv
def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
        
        precision.append(round(sumForPrecision / len(predictedIndices), 4))
        recall.append(round(sumForRecall / len(predictedIndices), 4))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
        MRR.append(round(sumForMRR / len(predictedIndices), 4))
        
    return precision, recall, NDCG, MRR

import numpy as np
import bottleneck as bn
import torch
import math
import time
import csv
import os
def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
        
        precision.append(round(sumForPrecision / len(predictedIndices), 4))
        recall.append(round(sumForRecall / len(predictedIndices), 4))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
        MRR.append(round(sumForMRR / len(predictedIndices), 4))
        
    return precision, recall, NDCG, MRR


def print_results(loss, valid_result, test_result, epoch=None, csv_path="results_log.csv"):
    """Output the evaluation results and save them to CSV."""

    # 打印训练损失
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))

    # 打印验证集结果
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
            '-'.join([str(x) for x in valid_result[0]]), 
            '-'.join([str(x) for x in valid_result[1]]), 
            '-'.join([str(x) for x in valid_result[2]]), 
            '-'.join([str(x) for x in valid_result[3]])))

    # 打印测试集结果
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
            '-'.join([str(x) for x in test_result[0]]), 
            '-'.join([str(x) for x in test_result[1]]), 
            '-'.join([str(x) for x in test_result[2]]), 
            '-'.join([str(x) for x in test_result[3]])))

    # 组织CSV数据
    if valid_result is not None and test_result is not None:
        headers = [
            "Epoch", "Train Loss",
            "Valid_P@1", "Valid_P@5", "Valid_P@10", "Valid_P@20",
            "Valid_R@1", "Valid_R@5", "Valid_R@10", "Valid_R@20",
            "Valid_NDCG@1", "Valid_NDCG@5", "Valid_NDCG@10", "Valid_NDCG@20",
            "Valid_MRR@1", "Valid_MRR@5", "Valid_MRR@10", "Valid_MRR@20",
            "Test_P@1", "Test_P@5", "Test_P@10", "Test_P@20",
            "Test_R@1", "Test_R@5", "Test_R@10", "Test_R@20",
            "Test_NDCG@1", "Test_NDCG@5", "Test_NDCG@10", "Test_NDCG@20",
            "Test_MRR@1", "Test_MRR@5", "Test_MRR@10", "Test_MRR@20"
        ]
        row = [
            epoch if epoch is not None else "",
            round(loss, 4) if loss is not None else ""
        ] + valid_result[0] + valid_result[1] + valid_result[2] + valid_result[3] \
          + test_result[0] + test_result[1] + test_result[2] + test_result[3]

        # 检查文件是否存在并写入
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)
