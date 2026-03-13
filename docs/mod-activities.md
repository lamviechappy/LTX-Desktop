# Quá trình mod
1. Tìm file python có chứa "device" trong thư mục backend
    terminal
    find backend -name "*.py" | xargs grep -l "device"
2. Kết quả ở b1 => Tìm ra các file quan trọng

   1. File quyết định việc nhận diện GPU (Quan trọng nhất)
backend/services/gpu_info/gpu_info_impl.py
backend/services/gpu_info/gpu_info.py
=> Đây là nơi app định nghĩa thế nào là một GPU "hợp lệ". Chúng ta cần sửa để nó chấp nhận Apple M4 (MPS) thay vì chỉ đòi NVIDIA (CUDA).
   1. File quản lý dọn dẹp bộ nhớ GPU
backend/services/gpu_cleaner/torch_cleaner.py
=> File này sẽ chứa các lệnh như torch.cuda.empty_cache(). Trên Mac M4, chúng ta sẽ cần đổi hoặc bổ sung thêm torch.mps.empty_cache().
   1. File xử lý Pipeline (Cách chạy model)
backend/services/services_utils.py
=> File này thường chứa các hàm bổ trợ (utils) để quyết định dùng device nào cho toàn bộ ứng dụng.


3. Sửa file trong /Volumes/WD500/dev/LTX-2/packages/ltx-pipelines và trong /Volumes/WD500/dev/LTX-2/packages/ltx-core
   
4. Cài đặt gói mới sửa vào LTX-Desktop

uv pip install -e /Volumes/WD500/dev/LTX-2/packages/ltx-pipelines
uv pip install -e /Volumes/WD500/dev/LTX-2/packages/ltx-core

5. Sửa file /Volumes/WD500/dev/LTX-Desktop/backend/runtime_config/runtime_policy.py
   - hàm def decide_force_api_generations
   if system == "Darwin":
         # return True
         return False

6. Download model nhân công


curl -L "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled.safetensors" -o test-download.safetensors --progress-bar

mv test-download.safetensors "/Users/donald/Library/Application Support/LTXDesktop/models/ltx-2-19b-distilled.safetensors"

curl -L "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors" \
  -o "/Users/donald/Library/Application Support/LTXDesktop/models/ltx-2-spatial-upscaler-x2-1.0.safetensors" --progress-bar




  # Xem GEMMA_LLM_KEY_OPS mapping hiện tại transform keys như thế nào
/Volumes/WD500/dev/LTX-Desktop/backend/.venv/bin/python -c "
from ltx_core.text_encoders.gemma.encoders.encoder_configurator import GEMMA_LLM_KEY_OPS
print(GEMMA_LLM_KEY_OPS)
"