"""FLA-0 v2 render tooling (the independent evaluation, ticket B10): sole panel v2, foot-rig clips, sole close-ups, course layout.

Offscreen MuJoCo with MUJOCO_GL=egl (works on the compute server). Videos: H.264 MP4, 1280x720, 30 fps, via /usr/bin/ffmpeg.
Every value drawn is a recorded sensor value (fla_terrain.sim.recorder.EpisodeRecorder output at 50 Hz, nominal
sole v2 with 0.2 mm layer compliance): no smoothing, no interpolation; the latest finished 50 Hz frame is shown.
"""
