# climb-cut

`climb-cut` creates a time-synchronised composite of several attempts on one climbing route. Its JSON file is the sole editable project data: it is small enough to review and version-control, and generates fast OpenCV previews plus a fixed-30-FPS OpenCV render.

## Install and make a route

Install the declared dependency once with `uv sync`. If `uv` is not on your PowerShell PATH, invoke it as `& "$env:USERPROFILE\.local\bin\uv.exe" sync` (the setup in this folder uses that location), then run:

```powershell
uv run python climb_cut.py init yellow-arete.json 20260915_101609.mp4 20260915_101819.mp4 --route "Yellow arete"
uv run python climb_cut.py preview yellow-arete.json
```

Before reviewing, build the fast preview cache once. This creates a `clips/<video-name>/` folder next to your descriptor, containing a small JPEG for every decoded input frame. JPEG names are their source presentation timestamps, and `index.json` maps timestamps to files. Missing source videos are cached concurrently in threads; the main process draws one stable tqdm bar per video. It may take a little while and use disk space, but all later seeking is cache-backed:

```powershell
uv run python climb_cut.py cache yellow-arete.json --height 720
```

The seed has just `start`; its `output.duration` sets the end of the final video. Add a hold at the same meaningful position in every attempt, and give it one shared output time:

```json
"holds": [
  {"id": "start", "at": 0},
  {"id": "right-crimp", "at": 5.4}
],
"tracks": [{
  "id": "attempt-1",
  "source": "20260915_101609.mp4",
  "outcome": "flash",
  "anchors": {
    "start": {"source": 8.2, "opacity": 1},
    "right-crimp": {"source": 14.7, "opacity": 0}
  },
  "translation": [-32, 18],
  "enabled": true
}]
```

Set `"duration": 12.0` inside `output` to choose the final cut length. Each track anchor carries its source timestamp and its unnormalised opacity. Both source timing and opacity interpolate linearly between shared holds. Adjacent source timestamps may decrease: that interval plays backward, and later intervals follow their own mapping. Before the first and after the last hold source playback is always original speed. After the last hold, each track’s opacity fades linearly from its final anchor value to zero exactly as that source clip reaches its natural end. At every output instant the tool divides enabled tracks’ non-negative opacity values by their sum, so visible tracks always add to exactly 1. Use an anchor opacity of zero to hide an attempt after that hold. Mark failed attempts with `"outcome": "not_topped"`; this is descriptive metadata and does not remove video.

`translation` is `[x, y]` in final-output pixels: positive `x` moves right and positive `y` moves down; omit it for `[0, 0]`. `enabled` defaults to `true`; disabled tracks are excluded and the remaining visible tracks are renormalised to 100% opacity in preview and final render. The preview opens at 50% display scale while keeping the edit calculations full-resolution. Use `-`/`=` to zoom out/in, or launch it with `--scale 0.35` for a smaller window. Press `g` to snap exactly to the nearest shared hold. `n`/`p` selects the next/previous track without changing visibility; uppercase `N`/`P` selects it and solos it, disabling every other track. Press `v` to toggle the selected track. Press `i`/`j`/`k`/`l` to move the selected attempt up/left/down/right by 1 pixel (uppercase moves 10 pixels), then `s` to save. Press `h` at the playhead to add a shared hold (each attempt inherits its current source time and opacity). Seek to a hold, then use `a`/`d` to move that attempt’s source time by one exact decoded source frame (`A`/`D` moves 0.5 source seconds). `z`/`x` moves the selected shared hold in output time. `0` sets the selected track’s nearest-anchor opacity to `0`; `1` sets it to `1`. The one-line keyboard help is also shown in the controls window.

## Output

```powershell
uv run python climb_cut.py validate yellow-arete.json
uv run python climb_cut.py render yellow-arete.json --output renders/yellow-arete-preview.mp4
```

The OpenCV renderer is the final deterministic renderer. It emits exactly 30 output frames per second; frame `n` has output time `n / 30`. For each track it maps that time through the hold anchors, chooses the most recent input frame at or before that decoded source timestamp, and blends it with the normalised opacity. It does not create interpolated source frames, so variable/uneven source frame rates are preserved rather than flattened into nominal-FPS timing. Final renders use one bounded decode/transform worker per track and a separate bounded encoder worker; no track can run more than one output frame ahead of the compositor.

`init` does not guess where a climb begins: its `start` value is deliberately a whole-video placeholder. Set it to the real climbing moment before judging alignment. Add shared holds only where you want a retimed interval; no finish anchor is needed.
