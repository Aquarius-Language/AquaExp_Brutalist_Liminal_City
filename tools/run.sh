#!/bin/sh
# tools/run.sh 'mode=still&...'  runs the film program with the repository's CLI and runtime packs
export AQUARIUS_HOME=/c/OfficialProjects/AquariusLangTW
export AQUARIUS_QUERY="$1"
cd "$(dirname "$0")/.." && exec /c/OfficialProjects/AquariusLangTW/compiler/AquariusCli/bin/Release/net8.0/aqua.exe run .
