import xarray as xr
import requests
import netCDF4
import boto3
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import math
from FlightRadar24 import FlightRadar24API
from datetime import datetime, date, time, timezone

import pandas as pd
import sys
import glob
import time
import yaml
from tqdm.auto import tqdm
# from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

import torchvision.transforms as T
import albumentations as A
import segmentation_models_pytorch as smp
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import cv2


model_path = "model_7_3_2024_1.pytorch"

s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
fr_api = FlightRadar24API()
bucket_name = 'noaa-goes16'
product_name = 'ABI-L1b-RadF'

_T11_BOUNDS = (244, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)


def normalize_range(data, bounds):
    """Maps data to the range [0,1]"""
    normalized_data = (data - bounds[0]) / (bounds[1] - bounds[0])
    return normalized_data


def get_s3_keys(bucket, s3_client, prefix):
    """
    """
    filenames = []

    while True:
        resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in resp['Contents']:
            key = obj['Key']
            filenames.append(key)  # optional if you have more filefolders to got through.
        return filenames[-1]


def cord_to_planes(lat, lon, DS):
    """
    """
    H = 42164160
    A = DS.nominal_satellite_height.values
    r_eq = 6378137
    r_pol = 6356752.31414
    lat = lat * (math.pi / 180)
    lon = lon * (math.pi / 180)
    lat_C = math.atan(((r_pol * r_pol) / (r_eq * r_eq)) * math.tan(lat))
    e = 0.0818191910435
    r_c = (r_pol) / math.sqrt(1 - (e * e) * (math.cos(lat_C) * math.cos(lat_C)))
    lon_C = DS.nominal_satellite_subpoint_lon.values * (math.pi / 180)
    s_x = H - r_c * math.cos(lat_C) * math.cos(lon - lon_C)
    s_y = -r_c * math.cos(lat_C) * math.sin(lon - lon_C)
    s_z = r_c * math.sin(lat_C)
    x = math.asin((-s_y) / (math.sqrt((s_x * s_x) + (s_y * s_y) + (s_z * s_z))))
    y = math.atan((s_z) / (s_x))
    return np.array([A * math.tan(x) / 2, A * math.tan(y) / 2])


def do_stuff(resp, file_name):
    """
    """
    nc4_ds = netCDF4.Dataset(file_name, memory=resp.content)
    store = xr.backends.NetCDF4DataStore(nc4_ds)
    DS = xr.open_dataset(store)
    L = DS.Rad.values
    fk1 = DS.planck_fk1.values
    fk2 = DS.planck_fk2.values
    bc1 = DS.planck_bc1.values
    bc2 = DS.planck_bc2.values
    T = (fk2 / (np.log((fk1 / L) + 1)) - bc1) / bc2
    return T


def do_stuff_DS(resp):
    """
    """
    nc4_ds = netCDF4.Dataset("lol", memory=resp.content)
    store = xr.backends.NetCDF4DataStore(nc4_ds)
    return xr.open_dataset(store)

def refresh():
    """
    :return:
    """
    # Load time
    datetime.now(timezone.utc)
    year = datetime.now(timezone.utc).year
    day_of_year =datetime.now(timezone.utc).timetuple().tm_yday
    # day_of_year=63
    hour = datetime.now(timezone.utc).hour-1
    print(hour)
    # load DataStore
    req_DS=requests.get(f'https://noaa-goes16.s3.amazonaws.com/ABI-L1b-RadF/2024/067/14/OR_ABI-L1b-RadF-M6C15_G16_s20240671430206_e20240671439519_c20240671439572.nc')
    DS=do_stuff_DS(req_DS)
    print("Data Store Loaded")
    # Load Flights
    bounds = fr_api.get_bounds_by_point(DS.nominal_satellite_subpoint_lat.values, DS.nominal_satellite_subpoint_lon.values, 5000000)
    flights = fr_api.get_flights(bounds = bounds)
    lox = np.array([cord_to_planes(flight.latitude,flight.longitude,DS)for flight in flights ])
    print("Flights Loaded")
    # Load Stalite Images
    key= get_s3_keys(bucket_name,
                       s3_client,
                       prefix = f'{product_name}/{year}/{day_of_year:03.0f}/{hour:02.0f}/OR_{product_name}-M6C{15}'
                      )
    resp= requests.get(f'https://noaa-goes16.s3.amazonaws.com/{key}')
    third_band=do_stuff(resp,key.split('/')[-1][:-3])
    print("15th Band Loaded")
    key= get_s3_keys(bucket_name,
                       s3_client,
                       prefix = f'{product_name}/{year}/{day_of_year:03.0f}/{hour:02.0f}/OR_{product_name}-M6C{14}'
                      )
    resp = requests.get(f'https://noaa-goes16.s3.amazonaws.com/{key}')
    second_band=do_stuff(resp,key.split('/')[-1][:-3])
    print("14th Band Loaded")

    key= get_s3_keys(bucket_name,
                       s3_client,
                       prefix = f'{product_name}/{year}/{day_of_year:03.0f}/{hour:02.0f}/OR_{product_name}-M6C{11}'
                      )
    resp= requests.get(f'https://noaa-goes16.s3.amazonaws.com/{key}')

    first_band=do_stuff(resp,key.split('/')[-1][:-3])
    print("11th Band Loaded")



    # b - Applying normalization functions
    normalized_r =torch.from_numpy(normalize_range(third_band - second_band, _TDIFF_BOUNDS))
    normalized_g =torch.from_numpy(normalize_range(second_band - first_band, _CLOUD_TOP_TDIFF_BOUNDS))
    normalized_b =torch.from_numpy(normalize_range(second_band, _T11_BOUNDS))
    false_color=1-torch.stack([normalized_r, normalized_g, normalized_b], axis=0)

    return false_color,lox
def normalize_2d_array(arr):
    min_val = -15
    max_val =-5
    return (arr - min_val) / (max_val - min_val)
def set_below_threshold(arr, threshold):
    arr[arr < threshold] = 0
    arr[arr > threshold] = 1
    return arr
cfg = yaml.safe_load("""
data:
  resize: 512          # maxvit_tiny_tf_512
  augment: rotation    

augment_prob: 0.95    

model:
  encoder: tu-maxvit_tiny_tf_512.in1k   # timm-resnest26d
  pretrained: False    
  decoder_channels: [256, 128, 64, 32, 16]

kfold:
  k: 10
  folds: 0  # 0,1,2,3,4

train:
  batch_size: 4        
  weight_decay: 1e-2
  clip_grad_norm: 1000.0
  num_workers: 2

val:
  per_epoch: 1        
  th: 0.45         

test:
  th: 0.45           

scheduler:
  - linear:
      lr_start: 1e-8
      lr_end: 8e-4
      epoch_end: 0.5
  - cosine:
      lr_end: 1e-6
      epoch_end: 15   
""")


def get_asym_conv(nc):
    """
    Final tiny convolution from y_sym_pred to y_pred
    - expected to shift 0.5 pixel back
    - also reduce from 512 to 256 when y_sym is 512x512
    """
    if nc == 256:
        hidden_size = 9
        asym_conv = nn.Sequential(
            nn.Conv2d(1, hidden_size, kernel_size=(3, 3), padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, 1, kernel_size=1),
        )
    elif nc == 512:
        hidden_size = 25
        asym_conv = nn.Sequential(
            nn.Conv2d(1, hidden_size, kernel_size=(5, 5), padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, 1, kernel_size=1),
        )
    else:
        raise NotImplementedError

    return asym_conv


class Model(nn.Module):
    def __init__(self, cfg, pretrained=True, tta=None):
        super().__init__()
        name = cfg['model']['encoder']
        pretrained = 'imagenet' if (pretrained and cfg['model']['pretrained']) else None
        decoder_channels = cfg['model']['decoder_channels']  # (256, 128, 64, 32, 16)

        self.unet = smp.Unet(name,
                             encoder_weights=pretrained,
                             classes=1,
                             decoder_channels=decoder_channels,
                             )

        self.asym_conv = get_asym_conv(cfg['data']['resize'])
        self.tta = TTA(tta) if tta is not None else None

    def forward(self, x):
        #         if self.tta is not None:
        #             x = self.tta.stack(x)  # TTA input

        y_sym = self.unet(x)

        #         if self.tta is not None:
        #             y_sym = self.tta.average(y_sym)

        #         # Tiny conv from y_sym_pred -> y_pred
        y_pred = self.asym_conv(y_sym)
        return y_sym, y_pred

def pred_on_model(fal_image):
    images = []
    img_size = 256
    resize = T.Resize(512, antialias=False)
    for x in range(int(3328 / img_size)):
        for y in range(int(3328 / img_size)):
            images.append(resize(fal_image[:, img_size * x:img_size * (x + 1), img_size * y:img_size * (y + 1)]))
    images = torch.from_numpy(np.array(images))

    model = Model(cfg, pretrained=True)
    model.to(device)  # Initialize model variable
    if torch.cuda.is_available():
        model_state_dict = torch.load(model_path)
    else:
        model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state_dict)
    model.eval()
    y_syms = []
    y_preds = []
    for x in images:
        inputx = torch.from_numpy(np.array([x])).to(device)
        with torch.no_grad():
            y_sym, y_pred = model(inputx)
            y_syms.append(y_sym)
            y_preds.append(y_pred)
    y_preds = [y_pred.detach().cpu().numpy() for y_pred in y_preds]
    y_syms = [y_sym.detach().cpu().numpy() for y_sym in y_syms]

    resize = T.Resize(512, antialias=False)
    outimage = np.zeros((img_size, 3328))
    t = 0.95
    for x in range(int(3328 / img_size)):
        #     layer=np.array(y_preds[(x*13)][0][0])
        layer = set_below_threshold(normalize_2d_array(np.array(y_preds[(x * 13)][0][0])), t)
        for y in range(1, int(3328 / img_size)):
            #         slide=np.array(y_preds[(x*13)+y][0][0])
            slide = set_below_threshold(normalize_2d_array(np.array(y_preds[(x * 13) + y][0][0])), t)
            layer = np.concatenate((layer, slide), axis=1)
        outimage = np.concatenate((outimage, layer), axis=0)
    outimage = outimage[img_size:]
    outimage = np.pad(outimage, 1024 + 24, mode='constant')
    return outimage


while True:
    false_color, lox = refresh()
    # ax,fig = plt.subplots(figsize=(12, 12))
    # plt.scatter([false_color.shape[0]/2],[false_color.shape[0]/2],c="red")
    # plt.scatter((false_color.shape[1]/2)+lox[...,0],(false_color.shape[1]/2)-lox[...,1],c="yellow",s=3)
    # plt.imshow(np.flipud(np.rot90(1-false_color.T)))
    #
    # plt.axis('off')
    #
    # plt.savefig('foo.png')
    #
    # plt.show()

    a = 24+(512*2)
    b = 5424-24-(512*2)
    fal_image = false_color[:, a:b, a:b]

    outimage = pred_on_model(fal_image)

    maskX = []
    maskY = []
    for x in tqdm(range(int(outimage.shape[0]))):
        for y in range(int(outimage.shape[1])):
            if outimage[x][y] == 1:
                maskX.append(x)
                maskY.append(y)

    
    ax, fig = plt.subplots(figsize=(12, 12))
    # plt.imshow(np.flipud(np.rot90(1-false_color.T)))
    plt.scatter(maskY, maskX, c="red", s=0.1)

    plt.scatter((false_color.shape[1]/2)+lox[..., 0], (false_color.shape[1]/2)-lox[..., 1], c="yellow", s=3)
    # plt.savefig('output.png')
    image = np.flipud(np.rot90(1-false_color.T))
    for x in range(len(maskX)):
        cv2.circle(image, (maskX[x], maskY[x]), 2, (255, 0, 0), 2)
    cv2.imshow('RGB Array', image)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
