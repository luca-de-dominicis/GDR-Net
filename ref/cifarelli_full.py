# encoding: utf-8
"""This file includes necessary params, info."""
import os
import mmcv
import os.path as osp

import numpy as np

# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
# directory storing experiment data (result, model checkpoints, etc).
output_dir = osp.join(root_dir, "output")

data_root = osp.join(root_dir, "datasets")
custom_root = osp.join(data_root, "")

# ---------------------------------------------------------------- #
# EPOSE DATASET
# ---------------------------------------------------------------- #
dataset_root = osp.join(custom_root, "cifarelli_DS_gdrn")
train_dir = osp.join(dataset_root, "train")
test_dir = osp.join(dataset_root, "test")
real_dir = osp.join(dataset_root, "real")
model_dir = osp.join(dataset_root, "models")
vertex_scale = 0.001
model_eval_dir = osp.join(dataset_root, "models_eval")

# object info
objects =["tubetto_m1200", "ugello_l80_99", "dado_m5", "vite_65", "vite_20", "chiave_brugola_6", "deviatore_boccaglio", "chiave_candela_19", "chiave_fissa_8_10", "fascetta_68_73"]
id2obj = {
    1: "tubetto_m1200",
    2: "ugello_l80_99",
    3: "dado_m5",
    4: "vite_65",
    5: "vite_20",
    6: "chiave_brugola_6",
    7: "deviatore_boccaglio",
    8: "chiave_candela_19",
    9: "chiave_fissa_8_10",
    10: "fascetta_68_73",
}
obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}

model_paths = [osp.join(model_dir, "obj_{:06d}.ply").format(_id) for _id in id2obj]
texture_paths = None
model_colors = [((i + 1) * 10, (i + 1) * 10, (i + 1) * 10) for i in range(obj_num)]  # for renderer

diameters = (
    np.array(
        [
            55.47972602672079,
            50.1422975141746,
            13.748576598846908,
            72.38720467541849,
            27.739863013360395,
            101.25709822486309,
            151.77867863223565,
            178.5739376691493,
            151.09723406163698,
            125.20319578083749
        ]
    )
    / 1000.0
)

# Camera info
width = 640
height = 480
center = (height / 2, width / 2)



def get_models_info():
    """key is str(obj_id)"""
    models_info_path = osp.join(model_dir, "models_info.json")
    assert osp.exists(models_info_path), models_info_path
    models_info = mmcv.load(models_info_path)  # key is str(obj_id)
    return models_info


def get_fps_points():
    fps_points_path = osp.join(model_dir, "fps_points.pkl")
    assert osp.exists(fps_points_path), fps_points_path
    fps_dict = mmcv.load(fps_points_path)
    return fps_dict
