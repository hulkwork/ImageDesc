import os

dir_path = os.path.dirname(os.path.realpath(__file__))

conf = {"output_dir": os.path.join(dir_path, '../out/'),
        "cascade_file": os.path.join(dir_path, '../data/cascade/lbpcascade_animeface.xml'),
        "clusters" : 10}
