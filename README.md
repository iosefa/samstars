# sam

## Setup

### Create Conda Env and Install Dependencies

```bash
conda env create -f environment-wsl.yml -n sam_star
conda activate sam_star
```

## Step 1: Create tiles

```bash
python make_tiles.py --img image.tif --chm chm.tif --den density.tif 
```

## Step 2: Create Predictions Maps

```bash
# process everything (unchanged behaviour)
./run_predict_tiles.sh

# process only a few specific tiles
./run_predict_tiles.sh tile_000_001 tile_000_042
```

## Step 3: Create Crown Polygons


