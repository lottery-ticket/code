#!/usr/bin/env python3

import tempfile
import subprocess
import os
import tqdm

DEST = tempfile.mkdtemp()
DIRNAME = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

print(DEST)

def call(coms):
    for com in coms:
        subprocess.check_call(
            com,
            shell=True,
            cwd=DEST,
            stdout=subprocess.DEVNULL
        )

replacements = [
    b'REDACTED',
    b'REDACTED',
    b'REDACTED',
    b'REDACTED',
]

exclude_prefixes_from_results = [
    'ibm/',
    'REDACTED/',
    'experiments/',
    'Experiment Status',
    'gpu-src/official/transformer/test_data',
]

call([
    'git init',
    'git remote add origin https://github.com/comparing-rewinding-finetuning/comparing-rewinding-codebase.git',
    'git config user.name "Anonymous ICLR Submission"',
    'git config user.email "anonymous@example.com"',
    # 'git pull origin master',
])

files = sorted(subprocess.check_output(['git', 'ls-files'], cwd=DIRNAME, universal_newlines=True).strip().split('\n'))

for fname in tqdm.tqdm(files):
    if any(fname.startswith(e) for e in exclude_prefixes_from_results):
        continue
    if os.path.isdir(fname):
        os.makedirs(os.path.join(DEST, os.path.dirname(fname)), exist_ok=True)
    elif os.path.isfile(fname):
        with open(fname, 'rb') as f:
            os.makedirs(os.path.join(DEST, os.path.dirname(fname)), exist_ok=True)
            with open(os.path.join(DEST, fname), 'wb') as new:
                data = f.read()
                for repl in replacements:
                    data = data.replace(repl, b'REDACTED')
                new.write(data)

call([
    'git add .',
    'git commit -m "anonymized code updates"',
    'git push --set-upstream origin master',
])
