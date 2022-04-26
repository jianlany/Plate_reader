#!/usr/bin/env python3
import json
from glob import glob
fid = open('data.csv', 'w')
with open('./plate_image_results.json', 'r') as f:
    count = 0
    
    fid.write('filename, score, platenumber\n')
    for line in f:
        data = json.loads(line)
        try:
            result = data['results'][0]
            filename = data['filename'].split('_')[-1].replace('jpg', 'png')
            # print(filename, result['score'], result['plate'])
            if result['score'] < 0.8: print(filename, result['plate'], result['score'])
            fid.write('{}, {}, {}\n'.format(filename, result['score'],result['plate']))
        except:
            print(data)
            count += 1
fid.close()
print(count)


