from __future__ import absolute_import

import os
from got10k.experiments import *
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


from siamfc import TrackerSiamFC


if __name__ == '__main__':
    net_path = 'pretrained/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    root_dir = os.path.expanduser('~/data/VOT')
    e = ExperimentVOT(root_dir)
    e.run(tracker)
    e.report([tracker.name])
