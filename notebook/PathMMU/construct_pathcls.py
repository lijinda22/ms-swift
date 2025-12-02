import os
import json
import re

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

# Configuration
# Use the path relative to this script for portability
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(script_dir, 'data.json')
output_file_path = os.path.join(script_dir, 'data_exist.json')

# Image root directory
img_root_dir = '/data/dataset/vqa/PathMMU/images/'

def process_data():
    if not os.path.exists(data_file_path):
        print(f"Error: {data_file_path} not found.")
        return

    print(f"Loading data from {data_file_path}...")
    data = load_data(data_file_path)
    data_exist = {}
    
    total_count = 0
    exist_count = 0
    
    for source, source_data in data.items():
        data_exist[source] = {}
        for split, items in source_data.items():
            data_exist[source][split] = []
            for item in items:
                total_count += 1
                img_filename = item.get('img')
                if not img_filename:
                    continue
                
                img_abs_path = os.path.join(img_root_dir, img_filename)
                
                # Check if image exists
                if os.path.exists(img_abs_path):
                    # Construct question
                    question = item.get('question', '')
                    options = item.get('options', [])
                    if options:
                        question += '\n' + '\n'.join(options)
                    
                    # Construct answer
                    answer = item.get('answer', '')
                    if answer and isinstance(answer, str):
                        answer = answer[0] # Take the first letter
                    # assert answer 是 A,B,C,.......Z 中的首字母, 使用正则表达式
                    assert re.match(r'^[A-Z]$', answer)
                    
                    new_item = {
                        'img': img_abs_path,
                        'question': question,
                        'answer': answer
                    }
                    
                    data_exist[source][split].append(new_item)
                    exist_count += 1
                
                if total_count % 1000 == 0:
                    print(f"Processed {total_count} items...")
    
    save_data(data_exist, output_file_path)
    print(f"Total items processed: {total_count}")
    print(f"Items with existing images saved: {exist_count}")
    print(f"Saved to {output_file_path}")

if __name__ == '__main__':
    process_data()
