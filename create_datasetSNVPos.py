import argparse

def get_args():
    parser = argparse.ArgumentParser(description='create dataset DNN')
    parser.add_argument('--temp-dir', type=str, default=None,
                        help='temp inputpath')
    parser.add_argument('--output-path', type=str, default=None,
                        help='output path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    import glob
    import pdb
    import pandas as pd
    import numpy as np

    from collections import Counter

    args = get_args()

    all_samples = glob.glob(args.temp_dir + '*.gc.genic.exonic.cs.tsv.gz*')

    template = pd.read_csv('SNVPos_template.csv')

    pd_dataset = pd.DataFrame()

    all_samp = []
    for i in range(len(all_samples)):
        samples = all_samples[i]
        read_samples = pd.read_csv(samples,sep='\t')

        clean_samples = samples.split('/')[-1]
        clean_samples = clean_samples[:-26]
        all_samp.append(clean_samples)

        read_samples = read_samples.loc[read_samples['seq'].isin(template.columns.to_list())]
        
        ps = (read_samples['pos'] / 1000000).apply(np.floor).astype(int).astype(str)
        chrom = read_samples['chrom'].astype(str)
        chrompos = chrom + '_' + ps
        read_samples['chrompos'] = chrompos
            
        read_samples_seq = read_samples['seq']
        read_samples_pos = read_samples['chrompos']

        counter = Counter(read_samples_seq)
        counter_pos = Counter(read_samples_pos)

        all_temp = []
        all_value = []
        for j in range(len(template.columns.to_list())):
            try:
                if j <=96:
                    temp_col = template.columns.to_list()[j]
                    value_col = counter.get(temp_col,0)
                else:
                    temp_col = template.columns.to_list()[j]
                    value_col = counter_pos.get(str(temp_col),0)
            except:
                pdb.set_trace()

            all_temp.append(temp_col)
            all_value.append(value_col)

        pd_samp = pd.DataFrame(all_value).T
        pd_samp.columns = all_temp
        pd_dataset = pd_dataset.append(pd_samp)
    
    pd_dataset['samples'] = all_samp
    pd_dataset = pd_dataset.reset_index(drop=True)
    pd_dataset.to_csv(args.output_path)

    print('creating dataset is successful')
   

    





    

    






