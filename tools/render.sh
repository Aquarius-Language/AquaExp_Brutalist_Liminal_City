#!/bin/sh
# tools/render.sh [spp]  renders every frame of the film to out/frames (resumes: existing frames are skipped)
cd "$(dirname "$0")/.."
exec tools/run.sh "mode=video&spp=${1:-28}&out=out/frames" > out/render.log 2>&1
