# mm_scene_rec

# Usage
## Recording p(z|x) model data
```
cd mm_scene_rec
python3 scripts/est_scene_model.py 1000 'home_music_1000' 'home_music' 'audio_scene_category'
```

# Future Work
## Improvements
- Fix checkpoint path in audio_scene_rec
- Replace cpu() and .cuda() in audio_scene_rec with self.device as appropriate