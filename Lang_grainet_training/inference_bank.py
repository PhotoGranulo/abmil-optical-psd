import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Replaced 'from osgeo import gdal' with PIL
from PIL import Image
import numpy as np
from tf_keras.layers import Input

import preprocessing as prepro
from resnet_architecture import FCN_grainsize
import helper


def read_rgb(ds, rgb_indices=(1, 2, 3)):
    """
    Read RGB bands and returns a 3D numpy array with shape: [height, width, channels]
    :param ds: PIL Image object (replaces gdal raster dataset)
    :param rgb_indices: Tuple of indices (1-based to match original logic)
    :return: numpy rgb bands
    """
    # Convert PIL image to numpy array [H, W, C]
    img_array = np.array(ds, dtype=np.float32)
    
    # Select specific bands based on rgb_indices (original was 1-based)
    # Most standard images are already RGB, so indices 1,2,3 map to 0,1,2
    bands_all = []
    for i in rgb_indices:
        band_array = img_array[:, :, i-1] 
        print('band_array.shape: ', band_array.shape)
        bands_all.append(band_array)
        
    bands_all = np.array(bands_all, dtype=np.float32)
    bands_all = np.moveaxis(bands_all, source=0, destination=2) # channels last
    return bands_all


def get_tiles(ds, tile_rows=500, tile_cols=200):

    img = read_rgb(ds=ds)
    img_rows, img_cols = img.shape[0:2]

    print('img_rows, img_cols:', img_rows, img_cols)
    print('tile_rows: {}, tile_cols: {}'.format(tile_rows, tile_cols))
    
    rows_range = np.arange(0, int(img_rows / tile_rows) * tile_rows, tile_rows)
    cols_range = np.arange(0, int(img_cols / tile_cols) * tile_cols, tile_cols)

    # round to integers for selecting number of pixels
    rows_range = np.array(np.round(rows_range), dtype=np.int32)
    cols_range = np.array(np.round(cols_range), dtype=np.int32)

    print('num tiles row: ', len(rows_range))
    print('num tiles col: ', len(cols_range))
    print('total tiles :', len(rows_range) * len(cols_range))

    tiles = []
    for i in rows_range:
        for j in cols_range:
            tile = img[i:i + int(tile_rows), j:j + int(tile_cols), :]
            if tile.shape[:2] == (int(tile_rows), int(tile_cols)):
                tiles.append(tile)
    tiles = np.array(tiles)
    print('tiles.shape:', tiles.shape)
    pred_shape = (len(rows_range), len(cols_range))
    print('pred_shape: ', pred_shape)
    return tiles, pred_shape


def save_array_as_geotif(out_path, ref_ds, array, x_res, y_res, out_width, out_height):
    """
    Saves the array as a standard TIFF since GeoTIFF metadata requires GDAL.
    """
    # Normalized array for image saving if necessary, or just save raw values
    output_img = Image.fromarray(array.astype(np.float32))
    output_img.save(out_path)
    print(f"Saved prediction map to {out_path}")


def run_prediction_orthophoto(args):
    GSD_orig = 0.0025
    print('downsample_factor: ', args.downsample_factor)

    if not os.path.exists(args.inference_path):
        os.makedirs(args.inference_path)

    # Replaced gdal.Open with PIL Image.open
    ds = Image.open(args.image_path).convert('RGB')

    # adjust original tile size (GSD 0.0025) for downsampling factor
    args.img_rows /= args.downsample_factor
    args.img_cols /= args.downsample_factor

    tiles, pred_shape = get_tiles(ds=ds, tile_rows=args.img_rows, tile_cols=args.img_cols)

    # load preprocessing statistics
    train_MEAN = np.load(os.path.join(args.experiment_dir, 'train_MEAN.npy'))
    train_STD = np.load(os.path.join(args.experiment_dir, 'train_STD.npy'))

    X_test_prepro = prepro.normalize_images_per_channel(images=tiles, mean_train=train_MEAN, std_train=train_STD,
                                                        out_dtype='float32')

    # initialize the model with input of proper shape
    input_shape = (int(args.img_rows), int(args.img_cols), args.channels)  # for tensorflow: channels last
    img_input = Input(shape=input_shape)
    # load model
    model = FCN_grainsize(img_input=img_input, bins=args.bins, output_scalar=args.output_dm)

    # load trained weights
    weights_filepath_val = os.path.join(args.experiment_dir, 'weights_best_val.h5')
    model.load_weights(weights_filepath_val)

    # predict
    predictions = model.predict(X_test_prepro)
    print('predictions.shape: ', predictions.shape)

    if not args.output_dm:
        # get dms
        dm_preds = []
        for pred in predictions:
            dm_preds.append(helper.get_dm(pred))
        dm_preds = np.array(dm_preds)
    else:
        # copy predictions (dm output)
        dm_preds = np.array(predictions)

    print('dm_preds.shape:', dm_preds.shape)

    # reshape predictions
    dm_pred_reshaped = np.reshape(dm_preds, newshape=pred_shape)
    print('dm_pred_reshaped.shape: ', dm_pred_reshaped.shape)

    # Maintained function call; saves as standard .tif
    save_array_as_geotif(out_path=os.path.join(args.inference_path, 'dm_pred.tif'), ref_ds=ds, array=dm_pred_reshaped,
                         x_res=GSD_orig * args.downsample_factor * args.img_cols,
                         y_res=GSD_orig * args.downsample_factor * args.img_rows,
                         out_width=pred_shape[1],
                         out_height=pred_shape[0])

    np.save(os.path.join(args.inference_path, 'predictions.npy'), predictions)
    np.save(os.path.join(args.inference_path, 'dm_pred_2D.npy'), dm_pred_reshaped)

    return predictions, dm_pred_reshaped, read_rgb(ds=ds)


if __name__ == "__main__":

    # set parameters
    parser = helper.setup_parser()
    args, unknown = parser.parse_known_args()

    run_prediction_orthophoto(args=args)