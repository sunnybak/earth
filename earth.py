import re
import pandas as pd
import sys
import os

def extract_params(file_path, df):

    index = len(df)

    with open(file_path, 'r') as file:
        data = file.read()

        nodes = re.search(r'nodes\s+(\d+)', data)
        _, value = nodes.group(0).split(' ')
        df.at[index, 'nodes'] = value

        bsim4 = re.search(r'bsim4\s+(\d+)', data)
        _, value = bsim4.group(0).split(' ')
        df.at[index, 'bsim4'] = value

        pattern_extract = r"Opening the PSFXL file \.\./psf/tran\.tran\.tran \.\.\.(.*?)\n\n"
        matches_extract = re.findall(pattern_extract, data, re.DOTALL)
        extracted_content = matches_extract[0].strip().split('\n') if matches_extract else "No match found"
        extracted_content = [s.strip(' ') for s in extracted_content][1:]


        def process_unit(s):
            s = s.strip(' ').replace(' ', '')
            if s == '':
                return float(1)
            elif s == 'M':
                return float(1e6)
            elif s == 'm':
                return float(1e-3)
            elif 'u' in s:
                return float(1e-6)
            elif 'n' in s:
                return float(1e-9)
            elif 'p' in s:
                return float(1e-12)
            elif 'f' in s:
                return float(1e-15)
            elif 'k' in s:
                return float(1e3)
            else:
                raise Exception('unknown unit')

        for s in extracted_content:
            key, val = s.split('=')
            key = key.strip(' ')
            val = val.strip(' ')

            if key == 'gmin':
                val = val[:-1] # remove S
                num, unit = val.split(' ')
                val = float(num) * process_unit(unit)
                df.at[index, 'gmin(S)'] = val
            elif key == 'maxstep':
                val = val[:-1] # remove s
                num, unit = val.split(' ')
                val = float(num) * process_unit(unit)
                df.at[index, 'maxstep(s)'] = val
            elif key == 'abstol(V)':
                val = val[:-1] # remove V
                num, unit = val.split(' ')
                val = float(num) * process_unit(unit)
                df.at[index, 'abstolV(V)'] = val
            elif key == 'abstol(I)':
                val = val[:-1] # remove V
                num, unit = val.split(' ')
                val = float(num) * process_unit(unit)
                df.at[index, 'abstolI(A)'] = val
            elif key == 'reltol':
                val = float(val)
                df.at[index, 'reltol'] = val
            elif key == 'method':
                df.at[index, key] = val
            elif key == 'errpreset':
                df.at[index, key] = val

        post_trans_data = data.split('Post-Transient Simulation Summary')[1]

        pattern_cpu = r"Total time required for tran analysis `tran': CPU\s*=\s*([\d\.]+)\s*([A-Za-z]?s)"
        cpu_time_matches = re.findall(pattern_cpu, post_trans_data)[0]
        unit = cpu_time_matches[1].replace('s', '')
        df.at[index, 'transtotaltime(s)'] = float(cpu_time_matches[0]) * process_unit(unit)
        
        pattern_mem = r"Peak resident memory used *= ([\d\.]+)\s*(\w+)"
        mem_matches = re.findall(pattern_mem, post_trans_data)[0]
        unit = mem_matches[1].replace('bytes', '')
        df.at[index, 'peakresmem(MBytes)'] = float(mem_matches[0]) * process_unit(unit) / 1e6

        cpu_info_match_str = "During simulation, the CPU load for active processors is :"
        pattern_cpu_info_start = post_trans_data.index(cpu_info_match_str) + len(cpu_info_match_str)
        pattern_cpu_info_end = post_trans_data.index("Total", pattern_cpu_info_start)
        info = post_trans_data[pattern_cpu_info_start:pattern_cpu_info_end].replace('\n', ' ').replace('Spectre', '')
        pattern_data = r"(\d+)\s*\(([\d\.]+)\s*%\)"
        matches = re.findall(pattern_data, info)
        df.at[index, 'numcpu'] = len(matches)
        
        if 'Total: ' in post_trans_data:
            pattern_cpu_info_start = post_trans_data.index('Total: ') + len('Total: ')
            pattern_cpu_info_end = post_trans_data.index('%', pattern_cpu_info_start)
            df.at[index, 'totalused(%)'] = float(post_trans_data[pattern_cpu_info_start:pattern_cpu_info_end])

            # pattern_cpu_info_start = post_trans_data.index('util. = ') + len('util. = ')
            # pattern_cpu_info_end = post_trans_data.index('%', pattern_cpu_info_start)
            # df.at[index, 'totalused(%)'] = float(post_trans_data[pattern_cpu_info_start:pattern_cpu_info_end])
        elif 'Other' in post_trans_data:
            df.at[index, 'totalused(%)'] = sum([float(m[1]) for m in matches])
        else:
            raise Exception('total cpu usage not found')
        
        pattern_cpu = r"CPU Type: (.*)"
        matches = re.findall(pattern_cpu, data)
        if len(matches) == 0:
            raise Exception('cpu type not found')
        df.at[index, 'coreType'] = matches[0]
        
        # get multithreading info
        pattern_mt = r"Multithreading enabled: (\d+) threads"
        n_threads = re.findall(pattern_mt, data)
        if len(n_threads) == 0:
            n_threads = [1]
        df.at[index, 'nthreads'] = int(n_threads[0])
        
        # extract cpu util
        pattern_cpu = r"util. = (\d+\.?\d*)%"
        cpu_util = re.findall(pattern_cpu, data)
        df.at[index, 'cpu_util'] = cpu_util[0]

    return df

if __name__ == '__main__':
    
    eecs = True
    
    df = pd.DataFrame(columns=[
        'transtotaltime(s)', 'peakresmem(MBytes)', 'numcpu', 'totalused(%)',
        'nthreads', 'cpu_util',
        'nodes', 'bsim4', 'coreType', 'errpreset',
        'gmin(S)', 'maxstep(s)', 'abstolV(V)', 'abstolI(A)', 'reltol', 'method',   
    ])

    # collect file names
    print('Paste the filenames from google sheets:\n')
    given_filenames = []
    count_newlines = 0
    while count_newlines < 1:
        line = sys.stdin.readline().strip()
        if line == "":
            count_newlines += 1
        else:
            given_filenames.append(line)
            count_newlines = 0
            
    # find the files
    base_path = os.path.join(os.getcwd(), 'out')
    filenames = {}
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".out"):
                filenames[file] = os.path.join(root, file)

    # extraction code
    if len(given_filenames) == 0:
        for filename in filenames.keys():
            print('Reading file: ', filename)
            df = extract_params(filenames[filename], df)
    else:
        for filename in given_filenames:
            print('Reading file: ', filename)
            # if there is a space in the filename, change it to the right format
            # filename = filename.replace(' ', '\ /')
            if filename not in filenames.keys():
                print(filename, ' not found')
                df = df._append(pd.Series(), ignore_index=True)
            else:
                extract_params(filenames[filename], df)

    # name the index column
    df.index.name = 'exp_no'

    # save df as csv
    df.to_csv('data.csv', index=False)
    print('\n\nData saved as data.csv!')