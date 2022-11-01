import pdb
import numpy as np
import pandas as pd

ntag = ['pcawgSNV150MNVPos']

allresults = []

for tag in ntag:
    for i in range(1,11):
        try:
            print(i)
            pd_readfile = pd.read_csv('./' + tag +'/'+ 'class_report_fold' + str(i) + '.csv',index_col = 0)
            allresults.append(pd_readfile[['accuracy','top3','top5','precision','recall','f1']].values)
        except:
            print('skip')
    np_allresults = np.asarray(allresults)

    #pdb.set_trace()
    #allresults.columns=['accuracy','top3','top5','precision','recall','f1']

    #meanall = np.mean(np_allresults,axis=1)
    meanall = np.mean(np_allresults,axis=1)
    #pdb.set_trace()
    #meanall = np.mean(np_allresults,axis=1)
    stdall = np.std(meanall,axis=0)
    meanall = np.mean(meanall,axis=0)

    pd_allmean = pd.DataFrame(meanall).T
    pd_allmean.columns=['accuracy','top3','top5','precision','recall','f1']

    pd_allstd = pd.DataFrame(stdall).T
    pd_allstd.columns=['accuracy','top3','top5','precision','recall','f1']

    pd_allmean.to_csv('./' + tag +'/' + 'allmean.csv')
    pd_allstd.to_csv('./' + tag +'/' + 'allstd.csv')


for tag in ntag:
    for i in range(1,5):
        try:
            print(i)
            pd_readfile = pd.read_csv('./' + tag +'/'+ 'RFclass_report_fold' + str(i) + '.csv',index_col = 0)
            allresults.append(pd_readfile[['accuracy','top3','top5','precision','recall','f1']].values)
        except:
            print('skip')

    np_allresults = np.asarray(allresults)

    #pdb.set_trace()
    #allresults.columns=['accuracy','top3','top5','precision','recall','f1']

    #meanall = np.mean(np_allresults,axis=1)
    meanall = np.mean(np_allresults,axis=1)
    #pdb.set_trace()
    #meanall = np.mean(np_allresults,axis=1)
    stdall = np.std(meanall,axis=0)
    meanall = np.mean(meanall,axis=0)

    pd_allmean = pd.DataFrame(meanall).T
    pd_allmean.columns=['accuracy','top3','top5','precision','recall','f1']

    pd_allstd = pd.DataFrame(stdall).T
    pd_allstd.columns=['accuracy','top3','top5','precision','recall','f1']

    pd_allmean.to_csv('./' + tag +'/' + 'allmeanRF.csv')
    pd_allstd.to_csv('./' + tag +'/' + 'allstdRF.csv')

