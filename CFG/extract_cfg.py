# The following code is used to extract the raw control flow graph from the assembly code of the binary file under asm_sample directory.
# The raw control flow graph is stored in the json format under the raw_cfg directory.
# We only inclue one example asm file from Big-15 dataset to show the code is functional. But the code can be used to extract the raw control flow graph from all the asm files in the asm_sample directory.
# The CFG construction code is built from MCBG codebase which is available here https://github.com/Bowen-n/MCBG. 

import os
from asm import *


if __name__ == '__main__':
    big2015_dir = '../data/examples/inputs/asm_sample'
    store_dir = '../data/examples/outputs/raw_cfg'
    file_format = 'json'

    with open('empty_code.err', 'r') as f:
        empty_code_ids = f.read().split('\n')

    count = 0
    file_list = os.listdir(big2015_dir)
    file_list = list(filter(lambda x: '.asm' in x, file_list))
    for filepath in file_list:
        count += 1
        # if '.asm' not in filepath:
        #     continue

        print('{}/{} File: {}. Info: '.format(count, len(file_list), filepath), end='')

        binary_id = filepath.split('.')[0]
        if binary_id in empty_code_ids:
            print('Empty code.')
            continue

        store_path = os.path.join(store_dir, '{}.{}'.format(binary_id, file_format))
        if os.path.exists(store_path):
            print('Already parsed.')
            continue

        parser = AsmParser(directory=big2015_dir, binary_id=binary_id)
        success = parser.parse()
        if success:
            parser.store_blocks(store_path, fformat=file_format)
            print('Success.')
        else:
            print('Empty code or block after parse.')

