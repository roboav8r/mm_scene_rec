# mm_scene_rec

# Usage

## Recording Scene Data
```
ros2 launch mm_scene_rec laptop_logi_demo.launch.py
ros2 bag record -o <filename> -s mcap /image_raw /audio_info /audio_data
```

## Recording p(z|x) model data - bag
```
ros2 launch mm_scene_rec scene_rec.launch.py # window 1
cd mm_scene_rec
python3 scripts/est_scene_model.py 10 'home_audio_test_10' 'home_test' 'audio_scene_category'
ros2 bag play <filename>
# kill the scene rec nodes
```

python3 scripts/est_scene_model.py 10 'home_clip_test_10' 'home_test' 'clip_scene_category'
ros2 bag play -p <filename>
```

## Recording p(z|x) model data - live
```
ros2 launch mm_scene_rec laptop_logi_demo.launch.py
cd mm_scene_rec
python3 scripts/est_scene_model.py 1000 'home_music_1000' 'home_music' 'audio_scene_category'
```

# Future Work
## Improvements
- Fix checkpoint path in audio_scene_rec
- Replace cpu() and .cuda() in audio_scene_rec with self.device as appropriate