#!/usr/bin/env python3
import time
import sys
from glob import glob
from subprocess import run, PIPE
import shlex
import json

with open('plate_image_results.json', 'w') as fid:
    for f in glob('./*.png'):
        if '--dry-run' in sys.argv:
            print('Image {} needs to run.'.format(f))
            continue
        cmd = """curl -F "upload=@./{}" -F regions=us-ca   -H "Authorization: Token 5cd9f6554f28797faac4572ff26f3199bcae247a"   https://api.platerecognizer.com/v1/plate-reader""".format(f)
        output = run(shlex.split(cmd), stdout = PIPE)
        time.sleep(1)
        fid.write('{}\n'.format(output.stdout.decode()))
