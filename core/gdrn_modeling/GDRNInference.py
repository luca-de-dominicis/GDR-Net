import logging
from loguru import logger as loguru_logger
import os
import os.path as osp
import sys
from setproctitle import setproctitle
import torch
from mmcv import Config
import cv2
from pytorch_lightning import seed_everything
from pytorch_lightning.lite import LightningLite  # import LightningLite
from torch.utils.data import Dataset
import json
from detectron2.evaluation import inference_context
from torch.cuda.amp import autocast
import itertools
import time
import numpy as np
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../"))

from core.utils.data_utils import crop_resize_by_warp_affine, get_2d_coord_np, read_image_cv2
from core.utils.default_args_setup import my_default_argument_parser, my_default_setup
from core.utils.my_setup import setup_for_distributed
from core.utils.my_checkpoint import MyCheckpointer
from core.gdrn_modeling.models import GDRN
from lib.utils.utils import iprint
from lib.utils.time_utils import get_time_str
from lib.pysixd import inout
from detectron2.structures import Boxes, BoxMode

from external.bRend.renderer import renderer
from external.bRend.renderer import misc
from external.bRend.renderer import inout as rendered_inout
import glob

class GDRNInference(LightningLite):
    def __init__(self, cfg, weights_path, info_path):
        super().__init__(
            accelerator="gpu",
            devices=1,
            num_nodes=1,
            precision=16,
            strategy=None,
        )
        self.cfg = self._init_cfg(cfg, weights_path)
        self.set_my_env()
        self.model = self._init_model()
        self.info = self._init_info(info_path)
        self.models = [osp.join(self.info["model_dir"], f"obj_{obj_id+1:06d}.ply") for obj_id in self.info["obj2id"].values()]
        self.extents = self._get_extents() 

    def _init_cfg(self, config_file, weights_path):
        """Create configs and perform basic setups."""
        cfg = Config.fromfile(config_file)
        if weights_path is not None:
            cfg.MODEL.WEIGHTS = weights_path

        if cfg.SOLVER.AMP.ENABLED:
            if torch.cuda.get_device_capability() <= (6, 1):
                iprint("Disable AMP for older GPUs")
                cfg.SOLVER.AMP.ENABLED = False

        # NOTE: pop some unwanted configs in detectron2
        # ---------------------------------------------------------
        cfg.SOLVER.pop("STEPS", None)
        cfg.SOLVER.pop("MAX_ITER", None)
        # NOTE: get optimizer from string cfg dict
        if cfg.SOLVER.OPTIMIZER_CFG != "":
            if isinstance(cfg.SOLVER.OPTIMIZER_CFG, str):
                optim_cfg = eval(cfg.SOLVER.OPTIMIZER_CFG)
                cfg.SOLVER.OPTIMIZER_CFG = optim_cfg
            else:
                optim_cfg = cfg.SOLVER.OPTIMIZER_CFG
            iprint("optimizer_cfg:", optim_cfg)
            cfg.SOLVER.OPTIMIZER_NAME = optim_cfg["type"]
            cfg.SOLVER.BASE_LR = optim_cfg["lr"]
            cfg.SOLVER.MOMENTUM = optim_cfg.get("momentum", 0.9)
            cfg.SOLVER.WEIGHT_DECAY = optim_cfg.get("weight_decay", 1e-4)
        # -------------------------------------------------------------------------
        exp_id = "{}".format(osp.splitext(osp.basename(config_file))[0])

        cfg.EXP_ID = exp_id
        ####################################
        return cfg

    def set_my_env(self):
        seed_everything(0, workers=True)
        setup_for_distributed(is_master=self.is_global_zero)

    def _init_model(self):
        model, optimizer = eval(self.cfg.MODEL.CDPN.NAME).build_model_optimizer(self.cfg)
        self.setup(model, optimizer)
        MyCheckpointer(model, save_dir=self.cfg.OUTPUT_DIR).resume_or_load(self.cfg.MODEL.WEIGHTS, resume=False)
        return model

    def _init_info(self, info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
        return info

    def _get_extents(self):
        cur_extents = {}
        for i, obj_name in enumerate(self.info["objects"]):
            obj_id = self.info["obj2id"][obj_name]
            model_path = osp.join(self.info["model_dir"], f"obj_{obj_id+1:06d}.ply")
            model = inout.load_ply(model_path, vertex_scale=self.info["vertex_scale"])
            pts = model["pts"]
            xmin, xmax = np.amin(pts[:, 0]), np.amax(pts[:, 0])
            ymin, ymax = np.amin(pts[:, 1]), np.amax(pts[:, 1])
            zmin, zmax = np.amin(pts[:, 2]), np.amax(pts[:, 2])
            size_x = xmax - xmin
            size_y = ymax - ymin
            size_z = zmax - zmin
            cur_extents[i] = np.array([size_x, size_y, size_z], dtype="float32")

        return cur_extents

    def _normalize_image(self, image):
        # image: CHW format
        pixel_mean = np.array(self.cfg.MODEL.PIXEL_MEAN).reshape(-1, 1, 1)
        pixel_std = np.array(self.cfg.MODEL.PIXEL_STD).reshape(-1, 1, 1)
        return (image - pixel_mean) / pixel_std

    def _pre_process(self, image, image_path, annotations):
        dataset_dict = {}
        im_H, im_W = image.shape[:2]
        dataset_dict["file_name"] = image_path
        dataset_dict["cam"] = torch.tensor(self.info["cam"])

        input_res = self.cfg.MODEL.CDPN.BACKBONE.INPUT_RES
        out_res = self.cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES

        # Prepare 2D coordinates and initialize empty lists for batched data
        coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)
        roi_keys = [
            "roi_img", "inst_id", "roi_coord_2d", "roi_cls", "roi_extent",
            "bbox_est", "bbox_mode", "bbox_center", "roi_wh",
            "scale", "resize_ratio",
        ]
        for key in roi_keys:
            dataset_dict[key] = []

        for inst_i, inst_infos in enumerate(annotations):
            dataset_dict["inst_id"].append(inst_i)
            roi_cls = inst_infos["obj_id"]
            dataset_dict["roi_cls"].append(roi_cls)
            roi_extent = self.extents[roi_cls - 1]
            dataset_dict["roi_extent"].append(roi_extent)

            bbox = BoxMode.convert(inst_infos["bbox_est"], 1, BoxMode.XYXY_ABS)
            x1, y1, x2, y2 = bbox
            bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
            bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
            scale = min(max(bh, bw) * self.cfg.INPUT.DZI_PAD_SCALE, max(im_H, im_W))

            dataset_dict["bbox_est"].append(bbox)
            dataset_dict["bbox_mode"].append(BoxMode.XYXY_ABS)
            dataset_dict["bbox_center"].append(bbox_center)
            dataset_dict["roi_wh"].append(np.array([bw, bh], dtype=np.float32))
            dataset_dict["scale"].append(scale)
            dataset_dict["resize_ratio"].append(out_res / scale)

            # Process and normalize ROI image
            roi_img = crop_resize_by_warp_affine(
                image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR,
                bbox=bbox, rnd_bg=self.random_backs[np.random.randint(0, len(self.random_backs))] if self.cfg.INPUT.RND_BG else None
            ).transpose(2, 0, 1)
            roi_img = self._normalize_image(roi_img)
            dataset_dict["roi_img"].append(roi_img.astype("float32"))

            # Process 2D coordinates for ROI
            roi_coord_2d = crop_resize_by_warp_affine(
                coord_2d, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)
            dataset_dict["roi_coord_2d"].append(roi_coord_2d.astype("float32"))

        for key in roi_keys:
            if key in ["roi_img", "roi_coord_2d"]:
                # Convert list of numpy arrays to a single numpy array before converting to tensor
                dataset_dict[key] = torch.as_tensor(np.stack(dataset_dict[key])).contiguous()
            elif key != "file_name":
                # Similarly, convert lists of numeric types to numpy array first
                dataset_dict[key] = torch.as_tensor(np.array(dataset_dict[key]))

        # Prepare final batch dictionary
        batch = {key: torch.cat([d[key] for d in [dataset_dict]], dim=0).to(device="cuda", dtype=torch.long if key == "roi_cls" else torch.float32, non_blocking=True) for key in roi_keys if key != "file_name"}
        batch["roi_cam"] = dataset_dict["cam"].reshape(1, 3, 3).to("cuda", non_blocking=True)
        batch["roi_center"] = torch.cat([dataset_dict["bbox_center"]], dim=0).to("cuda", non_blocking=True)
        batch["file_name"] = dataset_dict["file_name"]

        return batch

    def render(self, image, predictions):
        K = self.info["cam"]
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]
        rendere = renderer.create_renderer(640, 480, 'vispy', mode='rgb')
        (h, w) = image.shape[:2]
        # Initialize an empty overlay
        overlay = np.zeros_like(image)
        for j in range(len(predictions)):
            pose_R = np.array(predictions[j]['R']).reshape((3, 3))
            pose_t = np.array(predictions[j]['t']).reshape((3, 1)) * 1000 # we rescale to mm
            class_name = predictions[j]['class'] - 1
            rendere.add_object(j, self.models[class_name])  # Using unique object ID for each model
            rgb_dict = rendere.render_object(j, pose_R, pose_t, fx, fy, cx, cy)
            ren_img = rgb_dict['rgb']

            # Update the overlay
            mask = ren_img[:, :, 1] > 0  # Assuming green channel is used for object masks
            overlay[mask] = ren_img[mask]
            overlay[mask, 0] = 0
            overlay[mask, 1] = 128
            overlay[mask, 2] = 0

        # Combine the overlay with the original image
        final_image = cv2.addWeighted(image, 1, overlay, 1, 0)

        cv2.imshow("All Predictions", np.concatenate((image, final_image), axis=1))
        cv2.waitKey(0)
    
    def _post_process(self, out_dict, img_dict):
        out = []
        for i in range(len(out_dict['trans'])):
            ele = {}
            ele['class'] = img_dict['roi_cls'][i].item()
            ele['t'] = out_dict['trans'][i].cpu().numpy().tolist()
            ele['R'] = out_dict['rot'][i].cpu().numpy().tolist()
            ele['file_name'] = img_dict['file_name']
            out.append(ele)
    
        return out

    def run(self, image, image_path, annotations=None):
        start_preprocess_time = time.time()
        
        img_dict = self._pre_process(image, image_path, annotations)
        
        pre_process_time = time.time() - start_preprocess_time
        start_inference_time = time.time()


        with inference_context(self.model), torch.no_grad():
            with autocast(enabled=self.cfg.TEST.AMP_TEST):
                out_dict = self.model(
                    img_dict["roi_img"],
                    roi_classes=img_dict["roi_cls"],
                    roi_cams=img_dict["roi_cam"],
                    roi_whs=img_dict["roi_wh"],
                    roi_centers=img_dict["roi_center"],
                    resize_ratios=img_dict["resize_ratio"],
                    roi_coord_2d=img_dict.get("roi_coord_2d", None),
                    roi_extents=img_dict.get("roi_extent", None),
                )
        
        inference_time = time.time() - start_inference_time

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_postprocess_time = time.time()
        out = self._post_process(out_dict, img_dict)
        post_process_time = time.time() - start_postprocess_time
        print(f"preprocess time: {pre_process_time}, inference time: {inference_time}, postprocess time: {post_process_time}")
        return out, pre_process_time, inference_time, post_process_time


if __name__ == "__main__":    
    LM = False
    if not LM:
        model = GDRNInference(cfg="/data/repos/GDR-Net/configs/gdrn/epose/config_epose_48_s0_sym_r32.py",
        weights_path="/data/repos/GDR-Net/output/gdrn/epose/config_epose_48_s0_sym_r32/model_final.pth",
        info_path="/media/goldberg/T9/lucadedominicis/inference/info.json")
        files = glob.glob("/media/goldberg/T9/lucadedominicis/inference/images/*.png")
        with open("/media/goldberg/T9/lucadedominicis/inference/annotations.json", 'r') as f:
            annotations = json.load(f)
        pre_process_times, inference_times, post_process_times = [], [], []
        for file in files:
            image = cv2.imread(file)
            image_path = file
            annot = annotations[image_path]
            out, pre_process_time, inference_time, post_process_time = model.run(image, image_path, annotations=annot)
            model.render(image, out)
            pre_process_times.append(pre_process_time)
            inference_times.append(inference_time)
            post_process_times.append(post_process_time)
    else:
        model = GDRNInference(cfg="/data/repos/GDR-Net/configs/gdrn/lm/a6_cPnP_lm13_paper_s0.py",
        weights_path="/data/repos/GDR-Net/output/gdrn/lm/a6_cPnP_lm13_paper/gdrn_lm.pth",
        info_path="/media/goldberg/T9/lucadedominicis/inference/info_lm.json")
        files = ['/media/goldberg/T9/lucadedominicis/datasets/BOP_DATASETS/lm/test/000001/rgb/' + str(i.strip('\n')).zfill(6) + '.png' for i in open("/data/repos/GDR-Net/datasets/BOP_DATASETS/lm/image_set/ape_test.txt", "r").readlines()]
        with open("/data/repos/GDR-Net/datasets/BOP_DATASETS/lm/test/test_bboxes/bbox_faster_all.json", 'r') as f:
            annotations = json.load(f)
        pre_process_times, inference_times, post_process_times = [], [], []
        for file in files:
            num_img = file.split("/")[-1].split(".")[0].strip("0")
            if num_img == "":
                num_img = 0
            num_img = int(num_img)
            if f"{1}/{num_img}" in annotations:
                annot = annotations[f"{1}/{num_img}"]
                image = cv2.imread(file)
                out, pre_process_time, inference_time, post_process_time = model.run(image, file, annot)
                pre_process_times += [pre_process_time]
                inference_times += [inference_time]
                post_process_times += [post_process_time]
                #model.render(image, out)
    print("\n\n")
    print(f"Best preprocess time: {min(pre_process_times)}, Best inference time: {min(inference_times)}, Best postprocess time: {min(post_process_times)}, Best total time: {min([x + y + z for x, y, z in zip(pre_process_times, inference_times, post_process_times)])}")
    print(f"Best preprocess time: {min(pre_process_times[1:])}, Best inference time: {min(inference_times[1:])}, Best postprocess time: {min(post_process_times[1:])}, Best total time: {min([x + y + z for x, y, z in zip(pre_process_times[1:], inference_times[1:], post_process_times[1:])])}")
    print("\n\n")
    print(f"Worst preprocess time: {max(pre_process_times)}, Worst inference time: {max(inference_times)}, Worst postprocess time: {max(post_process_times)}, Worst total time: {max([x + y + z for x, y, z in zip(pre_process_times, inference_times, post_process_times)])}")
    print(f"Worst preprocess time: {max(pre_process_times[1:])}, Worst inference time: {max(inference_times[1:])}, Worst postprocess time: {max(post_process_times[1:])}, Worst total time: {max([x + y + z for x, y, z in zip(pre_process_times[1:], inference_times[1:], post_process_times[1:])])}")
    print("\n\n")
    print(f"Mean preprocess time: {np.mean(pre_process_times)}, mean inference time: {np.mean(inference_times)}, mean postprocess time: {np.mean(post_process_times)}, mean total time: {np.mean([x + y + z for x, y, z in zip(pre_process_times, inference_times, post_process_times)])}")
    print(f"Mean preprocess time: {np.mean(pre_process_times[1:])}, mean inference time: {np.mean(inference_times[1:])}, mean postprocess time: {np.mean(post_process_times[1:])}, mean total time: {np.mean([x + y + z for x, y, z in zip(pre_process_times[1:], inference_times[1:], post_process_times[1:])])}")

