# AquaExp_Brutalist_Liminal_City

A 3:29 film of a grey brutalist city under Metabolist towers, generated entirely in code in AquariusLang on
the graphics library's WebGPU module (`WGPU`) and path traced offline. No models, textures or photographs.

| Path | Contents |
| --- | --- |
| `main.aqua` | modes (`still`, `sheet`, `video`), sample accumulation in bands, post pass, frame output |
| `shots.aqua` | lighting presets (morning, day, sun break, dusk, night), the 10 shots, dissolves, grade |
| `shaders/scene.wgsl` | the world as distance functions (towers with tapered beam crowns, lift cabins, city blocks, decks with trams), materials (board-marked concrete, glazing, wet paving) and one path-traced sample per pixel |
| `shaders/post.wgsl` | filmic curve, grade, vignette, grain, fade |
| `tools/run.sh` | runs the program with the repository's CLI and runtime packs (`AQUARIUS_QUERY` settings) |
| `tools/render.sh` | renders all 5016 frames to `out/frames` (resumable; ~4 s per frame on an RTX 4070 laptop GPU) |
| `tools/music.py` | the score (numpy/scipy synthesis), `python -I tools/music.py out/music.wav` |
| `tools/compose.sh` | frames + music + the closing line into `out/JointCore.mp4`, exactly 3:29 |

```sh
tools/run.sh 'mode=still&t=156&spp=28&out=out/still.png'
tools/run.sh 'mode=sheet&times=10,60,128,156,180&out=out/sheet.png'
tools/render.sh 28 && python -I tools/music.py out/music.wav && sh tools/compose.sh
```
