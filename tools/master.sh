#!/bin/bash
# 渲染 4K 60 帧的 SDR 母版（无声），给 HDR 编码或直接混音封装用。
# 分成 N 段依次渲染，已经渲好、帧数对得上的段直接跳过：中断以后重跑同一条命令就能续上。
# 在 Remotion 项目的文件夹里运行（画面按 1920×1080 写，--scale=2 放大成 3840×2160，线条和字都是矢量，不会糊）。
# 用法：bash tools/master.sh [合成名 Main] [段数 8] [并发 8] [时间表 src/data/timeline.json]
# 输出：out/master/base_4k.mp4（H.264 4:4:4，CRF 10）
set -e
COMP=${1:-Main}; NSEG=${2:-8}; CONC=${3:-8}; TL=${4:-src/data/timeline.json}
mkdir -p out/master/parts
FPS=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('fps',60))" "$TL")
TOTAL=$(python3 -c "import json,math,sys;d=json.load(open(sys.argv[1]));print(math.ceil(d['total']*$FPS))" "$TL")
STEP=$(( (TOTAL + NSEG - 1) / NSEG ))
: > out/master/list.txt
for ((i = 0; i < NSEG; i++)); do
  A=$(( i * STEP )); B=$(( (i + 1) * STEP - 1 )); (( B > TOTAL - 1 )) && B=$(( TOTAL - 1 ))
  (( A > B )) && break
  F="out/master/parts/base_$(printf %02d $i).mp4"
  WANT=$(( B - A + 1 ))
  HAVE=$( [ -f "$F" ] && ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$F" || echo 0 )
  if [ "$HAVE" != "$WANT" ]; then
    echo "$(date +%H:%M:%S) 第 $((i + 1))/$NSEG 段：帧 $A-$B"
    # 片子里有视频素材时，几路视频同时抽帧偶尔会超时：等待上限放到 2 分钟，失败了这一段重渲，最多 3 次
    for TRY in 1 2 3; do
      if npx remotion render src/index.ts "$COMP" "$F" --frames=$A-$B --scale=2 --codec=h264 --crf=10 --pixel-format=yuv444p \
        --concurrency=$CONC --muted --log=error --timeout=120000; then break; fi
      echo "$(date +%H:%M:%S) 第 $((i + 1)) 段第 $TRY 次失败"
      [ "$TRY" = 3 ] && exit 1
    done
  else
    echo "第 $((i + 1))/$NSEG 段已经有了，跳过"
  fi
  echo "file 'parts/$(basename "$F")'" >> out/master/list.txt
done
ffmpeg -v error -y -f concat -safe 0 -i out/master/list.txt -c copy out/master/base_4k.mp4
N=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 out/master/base_4k.mp4)
echo "$(date +%H:%M:%S) 母版完成：$N 帧（应为 $TOTAL）"
[ "$N" = "$TOTAL" ] && echo MASTER_DONE
