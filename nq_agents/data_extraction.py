import json
import random
import os
from tqdm import tqdm


TRAIN_FILE = '.data/simplified-nq-train.jsonl'
DEV_FILE = '.data/v1.0-simplified_nq-dev-all.jsonl'
TRAIN_FILE_2000 = '.data/train_top2000.jsonl'
DEV_FILE_2000 = '.data/train_top2000_dev.jsonl' 


def count_lines(file_path):
    """Count the number of lines in a large file."""
    def blocks(file, size=65536):
        while True:
            b = file.read(size)
            if not b: break
            yield b
    with open(file_path, "r",encoding="utf-8",errors='ignore') as f:
        return sum(bl.count("\n") for bl in blocks(f))

def read_first_lines(file_path, num_lines=5):
    """Read the first num_lines lines from a file."""
    with open(file_path, 'r') as file:
        for _ in range(num_lines):
            print(file.readline().strip())

def parse_first_lines(file_path, num_lines=5):
    output = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for _ in range(num_lines):
            data = json.loads(file.readline().strip())
            output.append(data)
    return output

def write_jsonl(examples, output_path):
    """Write examples to a jsonl file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

def sample_examples(file_path, output_path=None, k=10, seed=42):
    """Sample k examples from input jsonl and write to output jsonl."""
    random.seed(seed)
    output = []
    # Open the file and iterate with an unbounded progress bar
    with open(file_path, 'r', encoding='utf-8') as file:
        # Initialize tqdm without a total, to track processed lines
        for i, line in enumerate(tqdm(file, desc="Sampling Lines", unit="line")):
            if i < k:
                output.append(json.loads(line.strip()))
            else:
                # Replace elements in the reservoir with decreasing probability
                j = random.randint(0, i)
                if j < k:
                    output[j] = json.loads(line.strip())
    # Set the output path if not provided
    if output_path is None:
        output_path = file_path.replace('.jsonl', f'_sample{k}_seed{seed}.jsonl')
    # Write sampled output to the output JSONL file
    write_jsonl(output, output_path)
    return output

def keep_short_ans_examples(file_path, output_path=None):
    """Keep examples with short answer."""
    output = []
    with open(file_path, 'r', encoding='utf-8') as file:
        # Initialize tqdm without a total count
        for line in tqdm(file, desc="Filtering Short Answer Examples", unit="line"):
            data = json.loads(line.strip())
            if (data['annotations'][0]['short_answers'] 
                or data['annotations'][0]['yes_no_answer'] != 'NONE'):
                output.append(data)
                
    # Set the output path if not provided
    if output_path is None:
        output_path = file_path.replace('.jsonl', '_short_ans.jsonl')
    
    # Write the filtered output to the output JSONL file
    write_jsonl(output, output_path)
    return output

def sample_short_ans_examples(file_path, output_path=None, k=100, seed=42):
    """Sample k examples with short answer from input jsonl and write to output jsonl."""
    temp_path = file_path.replace('.jsonl', '_temp.jsonl')
    keep_short_ans_examples(file_path, temp_path)
    output = sample_examples(temp_path, output_path, k, seed)
    os.remove(temp_path)
    return output
