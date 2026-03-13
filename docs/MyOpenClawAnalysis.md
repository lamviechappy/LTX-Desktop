📊 Phân tích: LTX Desktop + Local Models trên Mac M4
🔴 Điểm nghẽn chính
1. Video Generation Pipeline (ltx_pipelines)

Code hiện tại chỉ hỗ trợ CUDA (NVIDIA GPU)
macOS/MPS (Apple Silicon) không được hỗ trợ cho generation
Yêu cầu ≥32GB VRAM để chạy local
2. Text Encoder (đã có local option nhưng...)

Đã có settings use_local_text_encoder
Nhưng chỉ là phần text encoding, không phải video generation
Vẫn cần CUDA GPU
3. Prompt Enhancement (Gemini)

File: handlers/suggest_gap_prompt_handler.py
Hardcoded sử dụng Google Gemini API
Không có local fallback
📋 Tóm tắt các API đang dùng:
Chức năng	API	File
Text encoding	LTX API	ltx_text_encoder.py
Prompt enhancement	Gemini API	suggest_gap_prompt_handler.py
Video generation	LTX API	handlers/video_generation_handler.py
Retake	LTX API	handlers/retake_handler.py
Image generation (tùy chọn)	fal.ai	services/zit_api_client/
⚠️ Thực tế:
Theo README chính thức của project:

macOS (Apple Silicon builds) → API-only | LTX API key required

Project được thiết kế với mục tiêu không hỗ trợ local generation trên Mac. Lý do:

Video generation models cần CUDA
Apple Silicon MPS chưa được tối ưu/dịch sang để chạy các pipelines này
✅ Những gì CÓ THỂ làm local:
Thay thế Gemini bằng Ollama - Để chạy prompt enhancement locally
Text encoder - Đã có local option nhưng cần model weights
❌ Những gì KHÓ làm local:
Video generation - Cần port toàn bộ ltx_pipelines sang MPS hoặc dùng CPU (rất chậm)