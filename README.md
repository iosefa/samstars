# sam

## Step 1: Create tiles

```bash
python make_tiles.py
```

## Step 2: Create Predictions Maps

```bash
# process everything (unchanged behaviour)
./run_predict_tiles.sh

# process only a few specific tiles
./run_predict_tiles.sh tile_000_001 tile_000_042
```

