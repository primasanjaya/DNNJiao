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

    from collections import Counter

    args = get_args()

    all_samples = glob.glob(args.temp_dir + '*.gc.genic.exonic.cs.tsv.gz*')

    template = pd.read_csv('SNV_template.csv')

    pd_dataset = pd.DataFrame()

    all_samp = []
    for i in range(len(all_samples)):
        samples = all_samples[i]
        read_samples = pd.read_csv(samples,sep='\t')

        clean_samples = samples.split('/')[-1]
        clean_samples = clean_samples[:-26]
        all_samp.append(clean_samples)

        read_samples = read_samples.loc[read_samples['seq'].isin(template.columns.to_list())]
        read_samples = read_samples['seq']

        counter = Counter(read_samples)

        template.columns.to_list()

        all_temp = []
        all_value = []
        for j in range(len(template.columns.to_list())):
            temp_col = template.columns.to_list()[j]
            value_col = counter.get(temp_col,0)

            all_temp.append(temp_col)
            all_value.append(value_col)

        pd_samp = pd.DataFrame(all_value).T
        pd_samp.columns = all_temp
        pd_dataset = pd_dataset.append(pd_samp)
    
    pd_dataset['samples'] = all_samp
    pd_dataset = pd_dataset.reset_index(drop=True)
    pd_dataset.to_csv(args.output_path)

    print('creating dataset is successful')
   

    





    

    






