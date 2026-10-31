#!/bin/sh
# tools/compose.sh  out/frames + out/music.wav -> out/JointCore.mp4, exactly 3:29 (5016 frames at 24 fps).
# The only text, "Visuals made with AquariusLang", fades in at 196.5 s and out by 202.5 s (the picture fades to
# black from 203 s).
cd "$(dirname "$0")/.."
FONT="C\\:/Windows/Fonts/segoeuil.ttf"
A="if(lt(t,196.5),0,if(lt(t,198),(t-196.5)/1.5,if(lt(t,201),1,if(lt(t,202.5),(202.5-t)/1.5,0))))"
ffmpeg -v error -stats -y -framerate 24 -i out/frames/%05d.png -i out/music.wav \
    -filter_complex "[0:v]drawtext=fontfile='$FONT':text='Visuals made with AquariusLang':fontsize=34:fontcolor=0xE6E8EA:shadowcolor=0x000000@0.35:shadowy=1:alpha='$A':x=(w-tw)/2:y=h*0.80,format=yuv420p[v];[1:a]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[a]" \
    -map "[v]" -map "[a]" -frames:v 5016 -t 209 \
    -c:v libx264 -preset slow -crf 16 -tune film -r 24 -c:a aac -b:a 256k -ar 48000 -movflags +faststart out/JointCore.mp4
ffprobe -v error -show_entries format=duration:stream=codec_type,nb_frames,duration -of compact out/JointCore.mp4
