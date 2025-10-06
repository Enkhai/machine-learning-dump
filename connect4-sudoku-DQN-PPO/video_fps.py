from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx

in_loc = 'C:\\Users\\Ceyx\\Videos\\Movie Maker and Video Editor for Windows\\KingloftVideoEditor_9_27_2022_11_39_15_PM.mp4'
out_loc = 'dummy_out.mp4'

# Import video clip
clip = VideoFileClip(in_loc)
print("fps: {}".format(clip.fps))

# Modify the FPS
clip = clip.set_fps(clip.fps)

# Apply speed up
final = clip.fx(vfx.speedx, 2)
print("fps: {}".format(final.fps))

# Save video clip
final.write_videofile(out_loc)
