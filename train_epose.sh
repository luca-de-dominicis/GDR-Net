#!/usr/bin/env bash
# Ask the user if they prepared the dataset
while true; do
    read -p "Have you prepared the dataset? [y/n]" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) 
            echo "Ok it may take some time, I will ask you the necessary informations"
            read -p "Please provide the dataset path: " userInput
            echo "You entered: $userInput"
            echo "1/4 compute fps"
            sh core/csrc/compile.sh
            python tools/epose/compute_fps_epose.py --path $userInput
            echo "2/4 generate image set"
            python tools/epose/generate_image_set.py --path $userInput
            echo "3/4 generate xyz"
            python tools/epose/generate_xyz_from_pbr.py --path $userInput
            echo "4/4 split xyz"
            python tools/epose/split_xyz.py --path $userInput
            echo "Done preparing the dataset"
            break;; 
        * ) echo "Please answer yes or no.";;
    esac
done
./core/gdrn_modeling/train_gdrn.sh /data/repos/GDR-Net/configs/gdrn/epose/config_epose_16_s0_sym_r32_40_scalepos_sr_dado.py 0
