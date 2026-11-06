#!/bin/zsh
# SDR 母版 → HLG 10 位 HEVC → 杜比视界 8.4（HLG 基础层）MP4，另出一份不带杜比视界的 HLG MP4 备用。
# 用法：zsh tools/hdr/hdr_encode.sh BASE.mp4 MIX.wav OUT_DIR [CRF 16] [x265 预设 slow]
# 环境变量：PY（Python，默认 python3）、ENC（x265 或 vt）、VTRATE（vt 的码率，默认 50M）、
#           HDR_MASK_ARGS（传给 hdr_mask.py 的参数，比如 "--glow '#7FE6D8' --exclude '0,896,1920,1080'"）、HDR_PAPER（纸色）
# 要装：ffmpeg（带 libx265）、dovi_tool、MP4Box（GPAC）
# 步骤：
#   1. hdr_lut.py 出两张 LUT（base 普通映射、boost 提亮 GAIN 倍）；
#   2. hdr_mask.py 从成片算增益遮罩和每个镜头的亮度（已经有就跳过，删掉 OUT_DIR/mask.mkv 才会重算）；
#   3. ffmpeg：两张 LUT 各算一遍，按遮罩逐像素混合（maskedmerge），转成 BT.2020 10 位 4:2:0，x265 编码；
#   4. dovi_tool 按镜头亮度生成 RPU 并插进码流；MP4Box 封装，标成杜比视界 8.4（兼容 HLG）。
set -e
HERE=${0:A:h}
PY=${PY:-python3}
BASE=$1; AUD=$2; OUT=$3; CRF=${4:-16}; PRESET=${5:-slow}
[ -f "$BASE" ] && [ -f "$AUD" ] && [ -n "$OUT" ] || { echo "用法：hdr_encode.sh BASE.mp4 MIX.wav OUT_DIR [CRF] [预设]"; exit 1; }
mkdir -p "$OUT"
"$PY" "$HERE/hdr_lut.py" "$OUT/luts"
if [ ! -f "$OUT/mask.mkv" ] || [ ! -f "$OUT/dv_generator.json" ]; then
  "$PY" "$HERE/hdr_mask.py" "$BASE" "$OUT" ${(Q)${(z)HDR_MASK_ARGS:-}}   # (z) 按 shell 规则切开，(Q) 去掉引号
fi
WH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "$BASE")
W=${WH%x*}; H=${WH#*x}
NF=$("$PY" -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$OUT/frames.json")
echo "$(date +%H:%M:%S) 编码 ${W}x${H}，$NF 帧，CRF $CRF，预设 $PRESET"
# 两路都按帧编号重排时间戳：遮罩是 mkv（毫秒时间戳），画面是 1/60 秒，直接混会错开一帧
FC="[0:v]settb=1/60,setpts=N,scale=in_color_matrix=bt709:in_range=tv,format=gbrp16le,split[a][b];"
FC+="[a]lut3d=file='$OUT/luts/base.cube':interp=tetrahedral[A];"
FC+="[b]lut3d=file='$OUT/luts/boost.cube':interp=tetrahedral[B];"
FC+="[1:v]settb=1/60,setpts=N,scale=${W}:${H}:flags=bicubic:in_range=pc,format=gbrp16le[M];"
FC+="[A][B][M]maskedmerge[o];"
FC+="[o]scale=out_color_matrix=bt2020nc:out_range=tv:out_v_chr_pos=0:out_h_chr_pos=0,format=yuv420p10le[v]"
X265="colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited:chromaloc=2:repeat-headers=1:aud=1"
X265+=":hrd=1:vbv-maxrate=60000:vbv-bufsize=120000:keyint=120:min-keyint=12:open-gop=0:hdr10-opt=0"
# ENC=x265（默认，慢，同样画质体积小）或 ENC=vt（苹果硬件编码，快四倍左右，按码率 VTRATE 编，默认 50M）
if [ "${ENC:-x265}" = vt ]; then
  VCODEC=(-c:v hevc_videotoolbox -profile:v main10 -b:v "${VTRATE:-50M}" -maxrate "${VTRATE:-50M}" -g 120)
else
  VCODEC=(-c:v libx265 -preset "$PRESET" -crf "$CRF" -profile:v main10 -x265-params "$X265")
fi
ffmpeg -v error -stats -y -i "$BASE" -i "$OUT/mask.mkv" -filter_complex "$FC" -map "[v]" -frames:v "$NF" "${VCODEC[@]}" \
  -color_primaries bt2020 -color_trc arib-std-b67 -colorspace bt2020nc -color_range tv \
  -bsf:v hevc_metadata=colour_primaries=9:transfer_characteristics=18:matrix_coefficients=9:video_full_range_flag=0:chroma_sample_loc_type=2 \
  -f hevc "$OUT/hlg.hevc"   # 码流里的颜色标记直接写死：硬件编码器会忽略上面那几个参数
dovi_tool generate -j "$OUT/dv_generator.json" -o "$OUT/rpu.bin"
dovi_tool inject-rpu -i "$OUT/hlg.hevc" --rpu-in "$OUT/rpu.bin" -o "$OUT/dv.hevc"
ffmpeg -v error -y -i "$AUD" -c:a aac -b:a 320k -ar 48000 "$OUT/audio.m4a"
rm -f "$OUT/kazi_dv84.mp4" "$OUT/kazi_hlg.mp4"
MP4Box -quiet -add "$OUT/dv.hevc:dvp=8.hlg2100:fps=60" -add "$OUT/audio.m4a" -new "$OUT/kazi_dv84.mp4"
MP4Box -quiet -add "$OUT/hlg.hevc:fps=60" -add "$OUT/audio.m4a" -new "$OUT/kazi_hlg.mp4"
dovi_tool info -i "$OUT/dv.hevc" -f 0 --summary 2>/dev/null | head -20 || true
echo "$(date +%H:%M:%S) 完成：$OUT/kazi_dv84.mp4（杜比视界 8.4），$OUT/kazi_hlg.mp4（HLG）"
echo HDR_DONE
