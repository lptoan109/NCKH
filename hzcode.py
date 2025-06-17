#!/bin/bash

# Thư mục chứa các file WAV gốc
INPUT_DIR="./input_wavs"
# Thư mục lưu file đã chuyển đổi
OUTPUT_DIR="./output_wavs"


# Lặp qua tất cả các file .wav trong thư mục
for file in "$INPUT_DIR"/*.wav; do
    filename=$(basename "$file")
    output_file="$OUTPUT_DIR/${filename%.wav}_16khz.wav"
    
    # Chuyển sampling rate về 16000 Hz, mono, PCM 16-bit
    ffmpeg -y -i "$file" -ar 16000 -ac 1 -sample_fmt s16 "$output_file"
    
    echo "✅ Đã xử lý: $filename → $output_file"
done