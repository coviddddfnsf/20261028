import json
import numpy as np
from pathlib import Path

def trans(base_dict:dict):
    extents = base_dict['extents']
    for name in ['target_pose', 'contact_points_pose', 'functional_matrix']:
        for tmat in base_dict[name]:
            tmat[1][3] += extents[1] / 2
    if len(base_dict['orientation_point']) > 0:
        base_dict['orientation_point'][1][3] += extents[1] / 2
    return base_dict

base = Path('.')
for dir in base.iterdir():
    if not dir.is_dir(): continue
    try:
        idx, name = dir.name.split('_', 1)
        idx = int(idx)
    except ValueError:
        continue
    
    if int(idx) > 42: continue
    # if idx != 1: continue
    try:
        for file in dir.iterdir():
            if file.suffix != '.json': continue
            if not file.name.startswith('model_data'): continue
            
            print(f'Processing {file}')
            with open(file, 'r') as f:
                data = json.load(f)
                
            data = trans(data)
            
            with open(file, 'w') as f:
                json.dump(data, f, indent=4)
        print(f'Processed {file}')
    except Exception as e:
        print(f'Error processing {file}: {e}')
        continue