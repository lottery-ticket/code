#!/usr/bin/env python3

import utils
import sys
import os

for expt in utils.experiments_of_directories(sys.argv[1:]):
    for trial in expt.trial_data:
        for it in trial.iter_data:
            print(os.path.join(expt.experiment_dir, trial.trial, it.iter))
